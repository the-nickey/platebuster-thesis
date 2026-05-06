"""
Финальный eval двухстадийного pipeline: bbox-detector + keypoint-head.

Логика:
  1. Прогоняем bbox-модель (Ultralytics YOLO/yolo11n.pt и др.) на каждом фото из
     test_per_region/<region>/images/.
  2. Каждый predicted bbox crop'аем (с тем же padding, что в build_keypoint_crops).
  3. Crop кладём в keypoint-head → 4 локальных угла.
  4. Транформируем в global нормализованные координаты.
  5. Матчим predicted bbox с GT bbox по IoU≥0.5 (greedy).
  6. Для matched пар считаем pixel-error на 4 углах против GT keypoints из labels.

Метрики per-region:
  - bbox: precision, recall, mAP@0.5 (через model.val(), отдельно)
  - keypoint: mean / median / p90 pixel error на matched парах
  - end-to-end: какая доля GT plates получила valid 4 corners

Запуск:
    python scripts/training/eval_two_stage_pipeline.py \\
        --bbox-model runs/yolo_detect_<ts>/weights/best.pt \\
        --keypoint-head runs/keypoint_head_<ts>/best.pt \\
        --crop-padding 0.25 --crop-size 192
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch.nn as nn

from common import pick_device, TEST_REGIONS_DIR, REGIONS, RUNS_ROOT
from train_keypoint_head import KeypointHead


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox-model", required=True, help="путь к .pt весам bbox-детектора (Ultralytics)")
    p.add_argument("--keypoint-head", required=True, help="путь к .pt весам keypoint head (наш)")
    p.add_argument("--crop-size", type=int, default=192,
                   help="должно совпадать с обучением keypoint head")
    p.add_argument("--crop-padding", type=float, default=0.25,
                   help="должно совпадать с build_keypoint_crops.py")
    p.add_argument("--bbox-conf", type=float, default=0.25,
                   help="confidence threshold для bbox-детектора")
    p.add_argument("--iou-match", type=float, default=0.5,
                   help="порог IoU для матчинга predicted ↔ GT bbox")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default=None)
    p.add_argument("--out-name", default=None)
    return p.parse_args()


def parse_pose_label(label_path: Path) -> list[dict]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 17:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            kpts = [(float(parts[5 + i*3]), float(parts[5 + i*3 + 1]),
                     int(float(parts[5 + i*3 + 2]))) for i in range(4)]
            out.append({"cx": cx, "cy": cy, "w": w, "h": h, "kpts": kpts})
        except (ValueError, IndexError):
            continue
    return out


def iou_xywh(a, b) -> float:
    """a, b — (cx, cy, w, h) нормализованные [0..1]."""
    ax1, ay1 = a[0] - a[2]/2, a[1] - a[3]/2
    ax2, ay2 = a[0] + a[2]/2, a[1] + a[3]/2
    bx1, by1 = b[0] - b[2]/2, b[1] - b[3]/2
    bx2, by2 = b[0] + b[2]/2, b[1] + b[3]/2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def crop_bbox_to_square(img: np.ndarray, bbox_norm, padding: float, target_size: int):
    """Кропает квадратный crop вокруг bbox + padding. Возвращает (crop, origin_x, origin_y, side)."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox_norm[0]*w, bbox_norm[1]*h, bbox_norm[2]*w, bbox_norm[3]*h
    side = max(bw, bh) * (1 + 2 * padding)
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w, x2), min(h, y2)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return None, None, None, None

    crop = img[y1c:y2c, x1c:x2c]
    cw_actual = x2c - x1c
    ch_actual = y2c - y1c
    if cw_actual != ch_actual:
        max_side = max(cw_actual, ch_actual)
        canvas = np.full((max_side, max_side, 3), 114, dtype=np.uint8)
        ox = (max_side - cw_actual) // 2
        oy = (max_side - ch_actual) // 2
        canvas[oy:oy + ch_actual, ox:ox + cw_actual] = crop
        crop = canvas
        origin_x = x1c - ox
        origin_y = y1c - oy
        crop_side = max_side
    else:
        origin_x = x1c
        origin_y = y1c
        crop_side = cw_actual

    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return crop_resized, origin_x, origin_y, crop_side


def load_keypoint_head(weights_path: Path, device: str) -> nn.Module:
    model = KeypointHead().to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd if isinstance(sd, dict) and "state_dict" not in sd else sd["state_dict"])
    model.eval()
    return model


def preprocess_crop_for_head(crop_bgr: np.ndarray, device: str) -> torch.Tensor:
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std
    return t.unsqueeze(0).to(device)


def evaluate_region(region: str, bbox_model, kpt_head, args, device: str) -> dict:
    images_dir = TEST_REGIONS_DIR / region / "images"
    labels_dir = TEST_REGIONS_DIR / region / "labels"

    if not images_dir.exists():
        return {"error": "no images dir"}

    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    pred_results = bbox_model.predict(
        [str(p) for p in images], imgsz=args.imgsz, conf=args.bbox_conf,
        verbose=False,
    )

    matched_pixel_errs = []  # list of mean pixel error per matched pair
    matched_count = 0
    gt_with_kpts_total = 0
    pred_total = 0

    for img_path, pred in zip(images, pred_results):
        gts = parse_pose_label(labels_dir / f"{img_path.stem}.txt")
        gts_with_kpts = [g for g in gts if all(v == 2 for _, _, v in g["kpts"])]
        gt_with_kpts_total += len(gts_with_kpts)

        if pred.boxes is None or len(pred.boxes) == 0:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h_img, w_img = img.shape[:2]

        # predicted bboxes в нормализованных xywh
        boxes_xywhn = pred.boxes.xywhn.cpu().numpy()  # (N, 4) cx cy w h normalized
        confs = pred.boxes.conf.cpu().numpy()
        pred_total += len(boxes_xywhn)

        # greedy IoU match: сортируем по conf, для каждого ищем GT с max IoU
        used_gt = set()
        order = np.argsort(-confs)
        for i in order:
            pred_box = boxes_xywhn[i]
            best_iou = 0.0
            best_j = -1
            for j, g in enumerate(gts_with_kpts):
                if j in used_gt:
                    continue
                gt_box = (g["cx"], g["cy"], g["w"], g["h"])
                iou = iou_xywh(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_j = j
            if best_iou < args.iou_match or best_j < 0:
                continue

            # match
            used_gt.add(best_j)
            matched_count += 1

            # crop по predicted bbox + keypoint head
            crop, ox, oy, cside = crop_bbox_to_square(img, pred_box,
                                                      args.crop_padding, args.crop_size)
            if crop is None:
                continue

            with torch.no_grad():
                inp = preprocess_crop_for_head(crop, device)
                pred_kpts_local = kpt_head(inp).cpu().numpy()[0]  # (4, 2) в [0..1]

            # transform в global pixel space
            pred_kpts_px = []
            for lx, ly in pred_kpts_local:
                ax = ox + lx * cside
                ay = oy + ly * cside
                pred_kpts_px.append((ax, ay))

            # GT в pixel space
            gt = gts_with_kpts[best_j]
            gt_kpts_px = [(kx * w_img, ky * h_img) for kx, ky, _ in gt["kpts"]]

            # mean pixel error по 4 углам
            errs = [
                ((pkx - gkx) ** 2 + (pky - gky) ** 2) ** 0.5
                for (pkx, pky), (gkx, gky) in zip(pred_kpts_px, gt_kpts_px)
            ]
            matched_pixel_errs.append(float(np.mean(errs)))

    if matched_pixel_errs:
        arr = np.array(matched_pixel_errs)
        return {
            "n_images": len(images),
            "gt_with_kpts": gt_with_kpts_total,
            "pred_bboxes": pred_total,
            "matched": matched_count,
            "match_rate": matched_count / max(1, gt_with_kpts_total),
            "mean_px_err": float(arr.mean()),
            "median_px_err": float(np.median(arr)),
            "p90_px_err": float(np.percentile(arr, 90)),
            "p95_px_err": float(np.percentile(arr, 95)),
        }
    return {
        "n_images": len(images),
        "gt_with_kpts": gt_with_kpts_total,
        "pred_bboxes": pred_total,
        "matched": 0,
        "match_rate": 0.0,
        "mean_px_err": float("inf"),
    }


def main():
    args = parse_args()
    device = args.device or pick_device()
    print(f"Device: {device}")

    from ultralytics import YOLO
    bbox_model = YOLO(args.bbox_model)
    print(f"BBox model: {args.bbox_model}")

    kpt_head = load_keypoint_head(Path(args.keypoint_head), device)
    print(f"Keypoint head: {args.keypoint_head}")

    out_name = args.out_name or f"pipeline_eval_{Path(args.bbox_model).parent.parent.name}"
    out_dir = RUNS_ROOT / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Two-stage pipeline eval ===")
    results = {}
    for region in REGIONS:
        print(f"\n--- {region} ---")
        m = evaluate_region(region, bbox_model, kpt_head, args, device)
        results[region] = m
        if "error" in m:
            print(f"  {m['error']}")
            continue
        print(f"  images={m['n_images']}  gt_kpts={m['gt_with_kpts']}  "
              f"pred={m['pred_bboxes']}  matched={m['matched']} "
              f"({m['match_rate']*100:.1f}%)")
        if m['matched']:
            print(f"  pixel error mean={m['mean_px_err']:.2f}px median={m['median_px_err']:.2f}px "
                  f"p90={m['p90_px_err']:.2f}px")

    out_path = out_dir / "two_stage_metrics.json"
    out_path.write_text(json.dumps({
        "bbox_model": args.bbox_model,
        "keypoint_head": args.keypoint_head,
        "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nМетрики: {out_path}")

    print("\n=== СВОДНАЯ ТАБЛИЦА ===")
    print(f"{'Region':<12} {'matched%':>10} {'mean px':>10} {'median px':>10} {'p90 px':>10}")
    for region, m in results.items():
        if "error" in m or m.get("matched", 0) == 0:
            print(f"{region:<12} {'—':>10}")
            continue
        print(f"{region:<12} {m['match_rate']*100:>9.1f}% "
              f"{m['mean_px_err']:>10.2f} {m['median_px_err']:>10.2f} {m['p90_px_err']:>10.2f}")


if __name__ == "__main__":
    main()
