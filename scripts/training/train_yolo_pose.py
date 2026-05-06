"""
Single-stage YOLO11n-pose: bbox + 4 угла одновременно (одна модель, одна голова).

В отличие от 2-stage pipeline (detect + ResNet18 keypoint head на crop'е),
здесь pose-голова вшита в YOLO. Архитектурно это пятая модель в зоопарке
сравнения главы 2 §2.3.

ВАЖНО: на MPS YOLO-pose сломан (Ultralytics issue #18413, gradient scaler RuntimeError).
На CUDA работает нормально. Этот скрипт НЕ имеет MPS-пресета — на M3 Pro
просто упадёт.

Pretrained yolo11n-pose.pt обучен на COCO с 17 keypoints; при изменении
kpt_shape в data.yaml на [4, 3] Ultralytics переинициализирует pose-голову
под 4 точки, backbone остаётся pretrained.

Запуск:
    # CUDA (Yandex Cloud V100/A100):
    python scripts/training/train_yolo_pose.py --epochs 300

    # с overrides:
    python scripts/training/train_yolo_pose.py --batch 32 --lr0 0.0005
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    REPO_ROOT, FINETUNE_YAML, REGIONS, TEST_REGIONS_DIR,
    pick_device, timestamp_dir, make_test_yaml_for_region, assert_unified_exists,
)


PRESETS = {
    "cuda": dict(
        epochs=300, batch=64, imgsz=640, workers=8,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        freeze=None, patience=25, amp=True, mosaic=1.0,
        cache="ram",
        # pose-loss веса (default Ultralytics: pose=12.0, kobj=2.0).
        # На небольшом датасете 11K фото keypoints обычно требуют усиления —
        # оставляем default, при просадке pose mAP можно поднять pose= до 18-24.
        pose=12.0, kobj=2.0,
    ),
    "cpu": dict(
        epochs=50, batch=8, imgsz=640, workers=0,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        freeze=10, patience=10, amp=False, mosaic=0.3,
        cache="ram",
        pose=12.0, kobj=2.0,
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n-pose.pt",
                   help="pose-pretrained (yolo11n-pose.pt | yolo11s-pose.pt)")
    p.add_argument("--preset", choices=["cuda", "cpu", "auto"], default="auto",
                   help="набор гиперпараметров (auto = подобрать по pick_device())")
    p.add_argument("--name", default=None)
    p.add_argument("--device", default=None, help="cuda | cpu (mps НЕ поддерживается для pose)")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--cache", default=None, choices=["ram", "disk", "false"])

    p.add_argument("--optimizer", default=None, choices=["AdamW", "SGD", "auto"])
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--lrf", type=float, default=None)
    p.add_argument("--freeze", type=int, default=None)
    p.add_argument("--no-freeze", action="store_true")
    p.add_argument("--mosaic", type=float, default=None)
    p.add_argument("--pose-weight", type=float, default=None,
                   help="вес pose-loss (Ultralytics default 12.0)")
    p.add_argument("--kobj-weight", type=float, default=None,
                   help="вес keypoint objectness loss (default 2.0)")

    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp_explicit", action="store_const", const=True)
    amp_group.add_argument("--no-amp", dest="amp_explicit", action="store_const", const=False)
    p.set_defaults(amp_explicit=None)

    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def resolve_preset(args) -> tuple[str, str, dict]:
    device = args.device or pick_device(prefer_mps=False)
    if device == "mps":
        raise SystemExit(
            "YOLO-pose сломан на MPS (Ultralytics issue #18413). "
            "Запускай с --device cuda или --device cpu."
        )
    preset_name = args.preset
    if preset_name == "auto":
        preset_name = device if device in PRESETS else "cpu"
    preset = dict(PRESETS[preset_name])

    overrides = {
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "workers": args.workers, "patience": args.patience, "cache": args.cache,
        "optimizer": args.optimizer, "lr0": args.lr0, "lrf": args.lrf,
        "mosaic": args.mosaic, "pose": args.pose_weight, "kobj": args.kobj_weight,
    }
    if args.amp_explicit is not None:
        overrides["amp"] = args.amp_explicit
    if args.no_freeze:
        overrides["freeze"] = None
    elif args.freeze is not None:
        overrides["freeze"] = args.freeze

    for k, v in overrides.items():
        if v is not None:
            preset[k] = v

    if preset.get("cache") == "false":
        preset["cache"] = False

    return device, preset_name, preset


def main():
    args = parse_args()
    assert_unified_exists()

    from ultralytics import YOLO

    device, preset_name, params = resolve_preset(args)

    run_name = args.name or f"yolo_pose_{Path(args.model).stem}_{preset_name}"
    out_dir = timestamp_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== train_yolo_pose ===")
    print(f"  model:    {args.model}")
    print(f"  device:   {device}")
    print(f"  preset:   {preset_name}")
    print(f"  out_dir:  {out_dir}")
    print(f"  params:")
    for k, v in sorted(params.items()):
        print(f"    {k}: {v}")
    print()

    train_kwargs = dict(
        data=str(FINETUNE_YAML),
        device=device,
        plots=True,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        **params,
    )
    if train_kwargs.get("freeze") is None:
        train_kwargs.pop("freeze")

    model = YOLO(args.model)
    model.train(**train_kwargs)

    best = out_dir / "weights" / "best.pt"
    print(f"\nbest weights: {best}\n")

    if args.skip_eval:
        print("--skip-eval, выходим")
        return

    print("=== per-region eval ===")
    results = {}
    eval_model = YOLO(str(best))
    for region in REGIONS:
        ry = make_test_yaml_for_region(region, kpt_shape=[4, 3], flip_idx=[1, 0, 3, 2])
        if ry is None:
            print(f"  {region}: пропуск (нет {TEST_REGIONS_DIR / region / 'images'})")
            continue
        try:
            metrics = eval_model.val(
                data=str(ry), split="val",
                imgsz=params["imgsz"], batch=params["batch"],
                workers=params["workers"], device=device,
                plots=False, save_json=False, verbose=False,
            )
            row = {
                "mAP50": float(getattr(metrics.box, "map50", 0)),
                "mAP50_95": float(getattr(metrics.box, "map", 0)),
                "precision": float(getattr(metrics.box, "mp", 0)),
                "recall": float(getattr(metrics.box, "mr", 0)),
            }
            if hasattr(metrics, "pose") and metrics.pose is not None:
                row["pose_mAP50"] = float(getattr(metrics.pose, "map50", 0))
                row["pose_mAP50_95"] = float(getattr(metrics.pose, "map", 0))
            results[region] = row
            print(f"  {region}: bbox mAP50={row['mAP50']:.3f} pose mAP50={row.get('pose_mAP50', '—')}")
        except Exception as e:
            print(f"  {region}: FAILED — {e}")
            results[region] = {"error": str(e)}
        finally:
            ry.unlink(missing_ok=True)

    (out_dir / "per_region_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ИТОГ ===")
    for r, m in results.items():
        if "error" in m:
            print(f"  {r:<10} ERROR")
        else:
            pose_str = f"{m['pose_mAP50']:.3f}" if "pose_mAP50" in m else "—"
            print(f"  {r:<10} bbox={m['mAP50']:.3f}  pose={pose_str}")


if __name__ == "__main__":
    main()
