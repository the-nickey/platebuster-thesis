"""
Обучалка RF-DETR (Roboflow transformer-based detector) на COCO-датасете.

Перед запуском надо собрать COCO-структуру через
`scripts/training/convert_yolo_to_coco.py` — она положит датасет в
`data/processed/coco/finetune/{train,valid}/` и `data/processed/coco/test_per_region/<region>/`.

После train-а проходит per-region eval через pycocotools COCOeval, формат метрик
совместим с YOLO/classical (mAP50, mAP50-95, precision, recall).

Запуск:
    python scripts/training/train_rfdetr.py
    python scripts/training/train_rfdetr.py --model medium --epochs 100
    python scripts/training/train_rfdetr.py --model nano --batch-size 32 --grad-accum 1

Доступные модели: nano | small | medium | large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import REGIONS, REPO_ROOT, timestamp_dir, pick_device


COCO_ROOT = REPO_ROOT / "data" / "processed" / "coco"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="medium", choices=["nano", "small", "medium", "large"],
                   help="вариант RF-DETR (default: medium ≈ старый rfdetr-base)")
    p.add_argument("--dataset-dir", default=str(COCO_ROOT / "finetune"),
                   help="COCO-папка с train/ и valid/ (default: data/processed/coco/finetune)")
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=8,
                   help="на A100 8-16 ок, на T4 4-8")
    p.add_argument("--grad-accum", type=int, default=2,
                   help="effective batch = batch_size × grad_accum (рекомендуется 16)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-encoder", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=None,
                   help="кратно 14, default — для модели")
    p.add_argument("--device", default=None, help="cuda | cpu | mps (default: pick_device)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--no-early-stopping", action="store_true")
    p.add_argument("--tensorboard", action="store_true",
                   help="включить TB-логгер (default: выключен — не тащим dep в облако)")
    p.add_argument("--checkpoint-interval", type=int, default=20)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


MODEL_CLASSES = {
    "nano":   "RFDETRNano",
    "small":  "RFDETRSmall",
    "medium": "RFDETRMedium",
    "large":  "RFDETRLarge",
}


def load_model_class(name: str):
    import importlib
    rfdetr = importlib.import_module("rfdetr")
    cls_name = MODEL_CLASSES[name]
    if not hasattr(rfdetr, cls_name):
        raise SystemExit(
            f"в установленной версии rfdetr нет класса {cls_name}. "
            f"проверь `pip show rfdetr` и доступные классы: "
            f"{[a for a in dir(rfdetr) if a.startswith('RFDETR')]}"
        )
    return getattr(rfdetr, cls_name)


def per_region_eval(model, regions_root: Path, device: str) -> dict:
    """Гоняем predict() по каждому test_per_region/<region>/ и считаем COCOeval."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = {}
    for region in REGIONS:
        region_dir = regions_root / region
        gt_json = region_dir / "_annotations.coco.json"
        if not gt_json.exists():
            print(f"  {region}: пропуск (нет {gt_json})")
            continue

        print(f"\n[{region}] eval...")
        coco_gt = COCO(str(gt_json))
        img_ids = coco_gt.getImgIds()

        predictions = []
        for img_id in img_ids:
            info = coco_gt.loadImgs(img_id)[0]
            img_path = region_dir / info["file_name"]
            try:
                detections = model.predict(str(img_path), threshold=0.05)
            except Exception as e:
                print(f"    {info['file_name']}: predict failed — {e}")
                continue

            # detections.xyxy : (N, 4), .confidence : (N,), .class_id : (N,)
            if not hasattr(detections, "xyxy") or len(detections.xyxy) == 0:
                continue
            for box, score, cls in zip(detections.xyxy, detections.confidence, detections.class_id):
                x1, y1, x2, y2 = [float(v) for v in box]
                predictions.append({
                    "image_id": img_id,
                    "category_id": int(cls) if int(cls) > 0 else 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                })

        if not predictions:
            print(f"  {region}: пустые предсказания")
            results[region] = {
                "mAP50": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0,
                "n_images": len(img_ids), "n_predictions": 0,
            }
            continue

        coco_dt = coco_gt.loadRes(predictions)
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        # stats: [AP@.5:.95, AP@.5, AP@.75, AP_S, AP_M, AP_L,
        #         AR@1, AR@10, AR@100, AR_S, AR_M, AR_L]
        results[region] = {
            "mAP50": float(ev.stats[1]),
            "mAP50_95": float(ev.stats[0]),
            "precision": None,
            "recall": float(ev.stats[8]),
            "n_images": len(img_ids),
            "n_predictions": len(predictions),
        }
        print(f"  {region}: mAP50={results[region]['mAP50']:.3f} "
              f"mAP50-95={results[region]['mAP50_95']:.3f} R={results[region]['recall']:.3f}")
    return results


def main():
    args = parse_args()
    device = args.device or pick_device(prefer_mps=False)  # rfdetr на mps не очень дружит

    out_dir = timestamp_dir(args.name or f"rfdetr_{args.model}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== train_rfdetr ({args.model}) ===")
    print(f"  device:       {device}")
    print(f"  dataset_dir:  {args.dataset_dir}")
    print(f"  out_dir:      {out_dir}")
    print(f"  epochs:       {args.epochs}")
    print(f"  batch_size:   {args.batch_size} × grad_accum={args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print()

    ModelCls = load_model_class(args.model)
    model = ModelCls()

    train_kwargs = dict(
        dataset_dir=args.dataset_dir,
        output_dir=str(out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        lr=args.lr,
        lr_encoder=args.lr_encoder,
        weight_decay=args.weight_decay,
        device=device,
        early_stopping=not args.no_early_stopping,
        early_stopping_patience=args.patience,
        tensorboard=args.tensorboard,
        wandb=False,
        checkpoint_interval=args.checkpoint_interval,
    )
    if args.resolution is not None:
        train_kwargs["resolution"] = args.resolution

    model.train(**train_kwargs)

    # порядок предпочтения для inference: total (averaged) > ema > regular > свежайший
    best_ckpt = None
    for name in ("checkpoint_best_total.pth", "checkpoint_best_ema.pth", "checkpoint_best_regular.pth"):
        cand = out_dir / name
        if cand.exists():
            best_ckpt = cand
            break
    if best_ckpt is None:
        candidates = sorted(out_dir.glob("checkpoint*.pth"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            best_ckpt = candidates[0]
            print(f"\nfallback на свежайший: {best_ckpt}")
        else:
            print(f"\n!!! не нашли checkpoint в {out_dir}, eval пропускаем")
            return

    if args.skip_eval:
        print("--skip-eval, выходим")
        return

    print(f"\n=== per-region eval (weights: {best_ckpt}) ===")
    # `from_checkpoint` устойчивее, чем ModelCls(pretrain_weights=...) — он сам резолвит вариант модели
    from rfdetr import from_checkpoint
    eval_model = from_checkpoint(str(best_ckpt))
    regions_root = COCO_ROOT / "test_per_region"
    results = per_region_eval(eval_model, regions_root, device)

    (out_dir / "per_region_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ИТОГ ===")
    for r, m in results.items():
        print(f"  {r:<10} mAP50={m['mAP50']:.3f}")


if __name__ == "__main__":
    main()
