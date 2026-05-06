"""
Контурный baseline: grayscale → Canny → morphological closing → findContours →
approxPolyDP до 4 углов → фильтр по aspect ratio.

Eval-only, без обучения. Гоняется на всех регионах test_per_region/ и пишет
per_region_metrics.json в формате, совместимом с YOLO/RF-DETR/keypoint head.

Метрики:
- precision / recall / F1 при IoU≥0.5 (без mAP — классика без confidence)
- mean pixel error на 4 углах для TP-матчей (Hungarian-сматчиваем углы pred↔gt)

Запуск:
    python scripts/training/eval_classical.py
    python scripts/training/eval_classical.py --visualize 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import REGIONS, TEST_REGIONS_DIR, REPO_ROOT, timestamp_dir


# гиперпараметры классического детектора
ASPECT_MIN = 2.0       # h:w плашки русской 4.5, китайской 3.5, бразильской 3.0; нижний порог щадящий
ASPECT_MAX = 6.5
MIN_AREA_FRAC = 1e-4   # bbox должен быть >= 0.01% площади кадра
MAX_AREA_FRAC = 0.5    # и <= 50%, иначе это не плашка
APPROX_EPS_FRAC = 0.02
TOP_CONTOURS = 50
IOU_TP = 0.5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None, help="имя run-а; default — classical_<timestamp>")
    p.add_argument("--visualize", type=int, default=0,
                   help="сохранить N случайных визуализаций pred vs gt для каждого региона")
    return p.parse_args()


def detect_plates_classical(img_bgr: np.ndarray) -> list[dict]:
    """Возвращает список кандидатов: [{'bbox': (x,y,w,h), 'corners': np.ndarray(4,2), 'conf': float}]."""
    H, W = img_bgr.shape[:2]
    img_area = H * W

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 75, 75)

    # Canny с авто-порогом по медиане
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:TOP_CONTOURS]

    candidates = []
    for rank, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        if peri < 20:
            continue

        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        if w < 8 or h < 4:
            continue

        ar = w / max(h, 1)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX):
            continue

        area_frac = (w * h) / img_area
        if not (MIN_AREA_FRAC <= area_frac <= MAX_AREA_FRAC):
            continue

        corners = approx.reshape(4, 2).astype(np.float32)
        candidates.append({
            "bbox": (x, y, w, h),
            "corners": corners,
            "conf": 1.0 - rank / TOP_CONTOURS,
        })

    return candidates


def parse_label_line(line: str, W: int, H: int) -> tuple[tuple[int, int, int, int], np.ndarray | None]:
    """YOLO-pose формат: class cx cy w h x1 y1 v1 x2 y2 v2 ... возвращает (bbox_xywh_px, corners_px or None)."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    cx, cy, w, h = map(float, parts[1:5])
    bx = int((cx - w / 2) * W)
    by = int((cy - h / 2) * H)
    bw = int(w * W)
    bh = int(h * H)

    corners = None
    if len(parts) >= 5 + 4 * 3:
        kpts = list(map(float, parts[5:5 + 4 * 3]))
        if all(kpts[2 + i * 3] >= 1 for i in range(4)):
            corners = np.array([
                [kpts[i * 3] * W, kpts[i * 3 + 1] * H] for i in range(4)
            ], dtype=np.float32)
    return (bx, by, bw, bh), corners


def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def hungarian_corner_match(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """4×4 минимальное среднее расстояние без scipy. Перебор 4! = 24 перестановок."""
    from itertools import permutations
    best_perm = None
    best_cost = float("inf")
    for perm in permutations(range(4)):
        cost = float(np.linalg.norm(pred[list(perm)] - gt, axis=1).mean())
        if cost < best_cost:
            best_cost = cost
            best_perm = perm
    return pred[list(best_perm)]


def evaluate_region(region: str, viz_count: int, viz_dir: Path) -> dict:
    region_dir = TEST_REGIONS_DIR / region
    if not (region_dir / "images").exists():
        return None

    img_paths = sorted([p for p in (region_dir / "images").iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    tp = fp = fn = 0
    pixel_errors = []
    bbox_count_gt = 0
    bbox_count_pred = 0
    samples_with_kpt = 0

    viz_left = viz_count
    rng = np.random.default_rng(42)
    viz_indices = set(rng.choice(len(img_paths), size=min(viz_count, len(img_paths)), replace=False).tolist()) if viz_count else set()

    for idx, img_path in enumerate(img_paths):
        label_path = region_dir / "labels" / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]

        # GT
        gt_boxes = []
        gt_corners_list = []
        for line in label_path.read_text().splitlines():
            bbox, corners = parse_label_line(line, W, H)
            if bbox is None:
                continue
            gt_boxes.append(bbox)
            gt_corners_list.append(corners)
        bbox_count_gt += len(gt_boxes)

        # Pred
        preds = detect_plates_classical(img)
        bbox_count_pred += len(preds)

        # greedy IoU матчинг по убыванию IoU
        used_gt = set()
        local_tp = 0
        for pred in sorted(preds, key=lambda p: -p["conf"]):
            best_iou = 0
            best_gt = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in used_gt:
                    continue
                v = iou(pred["bbox"], gb)
                if v > best_iou:
                    best_iou = v
                    best_gt = gi
            if best_iou >= IOU_TP and best_gt >= 0:
                used_gt.add(best_gt)
                local_tp += 1
                gt_c = gt_corners_list[best_gt]
                if gt_c is not None:
                    matched = hungarian_corner_match(pred["corners"], gt_c)
                    err = float(np.linalg.norm(matched - gt_c, axis=1).mean())
                    pixel_errors.append(err)
                    samples_with_kpt += 1
            else:
                fp += 1
        tp += local_tp
        fn += len(gt_boxes) - len(used_gt)

        # визуализация
        if idx in viz_indices and viz_left > 0:
            viz = img.copy()
            for gb in gt_boxes:
                x, y, w, h = gb
                cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for pred in preds:
                x, y, w, h = pred["bbox"]
                cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 0, 255), 2)
                pts = pred["corners"].astype(int)
                for px, py in pts:
                    cv2.circle(viz, (px, py), 4, (255, 0, 0), -1)
            (viz_dir / region).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(viz_dir / region / img_path.name), viz)
            viz_left -= 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_images": len(img_paths),
        "n_gt_boxes": bbox_count_gt,
        "n_pred_boxes": bbox_count_pred,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_pixel_error_corners": float(np.mean(pixel_errors)) if pixel_errors else None,
        "n_kpt_samples": samples_with_kpt,
    }


def main():
    args = parse_args()
    out_dir = timestamp_dir(args.name or "classical")
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "visualizations"
    print(f"=== eval_classical ===")
    print(f"  out_dir: {out_dir}")

    results = {}
    for region in REGIONS:
        print(f"\n[{region}]")
        m = evaluate_region(region, args.visualize, viz_dir)
        if m is None:
            print(f"  пропуск (нет {TEST_REGIONS_DIR / region})")
            continue
        results[region] = m
        pe = f"{m['mean_pixel_error_corners']:.1f}px" if m['mean_pixel_error_corners'] is not None else "—"
        print(f"  P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"px_err={pe} "
              f"(n={m['n_images']}, gt={m['n_gt_boxes']}, pred={m['n_pred_boxes']})")

    (out_dir / "per_region_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ИТОГ ===")
    for r, m in results.items():
        pe = f"{m['mean_pixel_error_corners']:.1f}px" if m['mean_pixel_error_corners'] is not None else "—"
        print(f"  {r:<10} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} px_err={pe}")


if __name__ == "__main__":
    main()
