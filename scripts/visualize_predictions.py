"""
Визуализация предсказаний YOLO-pose модели поверх GT-разметки.

Прогоняет модель на случайных N изображениях из указанной папки,
рисует:
  - GT-bbox magenta + GT-углы цветными точками с подписями TL/TR/BR/BL
  - Predicted bbox cyan + predicted-углы белыми крестиками с числом confidence
  - Сверху подпись: "GT: N bbox / Pred: M bbox @ conf>=0.25"

Запуск:
    python scripts/visualize_predictions.py \\
        --model runs/yolo_pose_<ts>/stage_b/weights/best.pt \\
        --images data/processed/unified/test_per_region/russian/images \\
        --n 30

или просто прогнать на отдельной картинке без GT:
    python scripts/visualize_predictions.py \\
        --model runs/yolo_pose_<ts>/stage_b/weights/best.pt \\
        --images path/to/image.jpg
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent

POINT_NAMES = ["TL", "TR", "BR", "BL"]
GT_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 215, 255)]  # BGR
PRED_COLOR = (255, 255, 255)  # белый


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="путь к .pt весам")
    p.add_argument("--images", required=True, help="папка с картинками или путь к одной картинке")
    p.add_argument("--out", default=str(REPO_ROOT / "qc_output" / "predictions"))
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--conf", type=float, default=0.25, help="confidence threshold для отрисовки")
    p.add_argument("--imgsz", type=int, default=640)
    return p.parse_args()


def parse_pose_label(label_path: Path):
    """YOLO-keypoints: class cx cy w h (x y vis)*4."""
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
            kpts = [(float(parts[5 + i*3]), float(parts[5 + i*3 + 1]), int(float(parts[5 + i*3 + 2])))
                    for i in range(4)]
            out.append({"class": cls, "cx": cx, "cy": cy, "w": w, "h": h, "kpts": kpts})
        except (ValueError, IndexError):
            continue
    return out


def find_labels_dir(images_dir: Path) -> Path | None:
    """По yolo-конвенции: .../images/... → .../labels/..."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    return None


def draw_gt(canvas, instances):
    h, w = canvas.shape[:2]
    for inst in instances:
        x1 = int((inst["cx"] - inst["w"]/2) * w)
        y1 = int((inst["cy"] - inst["h"]/2) * h)
        x2 = int((inst["cx"] + inst["w"]/2) * w)
        y2 = int((inst["cy"] + inst["h"]/2) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)

        has_kp = any(v == 2 for _, _, v in inst["kpts"])
        if has_kp:
            pts = [(int(x*w), int(y*h)) for x, y, _ in inst["kpts"]]
            for i, (px, py) in enumerate(pts):
                cv2.circle(canvas, (px, py), 5, GT_COLORS[i], -1)
            for i in range(4):
                cv2.line(canvas, pts[i], pts[(i+1) % 4], (255, 255, 255), 1)


def draw_predictions(canvas, result, conf_thr):
    """result — Ultralytics Results object для одной картинки."""
    h, w = canvas.shape[:2]
    n_drawn = 0
    if result.boxes is None:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()  # абсолютные координаты
    confs = result.boxes.conf.cpu().numpy()
    kpts_arr = None
    if result.keypoints is not None:
        kpts_arr = result.keypoints.xy.cpu().numpy()  # (N, 4, 2) абсолютные

    for i, (box, conf) in enumerate(zip(boxes, confs)):
        if conf < conf_thr:
            continue
        n_drawn += 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(canvas, f"{conf:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        if kpts_arr is not None:
            for px, py in kpts_arr[i]:
                px, py = int(px), int(py)
                cv2.line(canvas, (px-6, py-6), (px+6, py+6), PRED_COLOR, 2)
                cv2.line(canvas, (px-6, py+6), (px+6, py-6), PRED_COLOR, 2)
    return n_drawn


def add_caption(canvas, text):
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(overlay, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def main():
    args = parse_args()
    from ultralytics import YOLO

    model = YOLO(args.model)
    print(f"Loaded: {args.model}")

    images_path = Path(args.images)
    if images_path.is_file():
        image_files = [images_path]
        labels_dir = None
    else:
        all_imgs = sorted(p for p in images_path.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        rng = random.Random(args.seed)
        image_files = rng.sample(all_imgs, min(args.n, len(all_imgs)))
        labels_dir = find_labels_dir(images_path)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    print(f"Labels: {labels_dir}")

    # batch predict
    results = model.predict(
        [str(p) for p in image_files],
        imgsz=args.imgsz,
        conf=0.001,  # берём все, фильтруем при рисовании
        verbose=False,
    )

    total_pred_above_conf = total_gt = total_predicted_any = 0
    for img_path, result in zip(image_files, results):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gt_count = 0
        if labels_dir:
            instances = parse_pose_label(labels_dir / f"{img_path.stem}.txt")
            draw_gt(img, instances)
            gt_count = len(instances)
        n_pred = draw_predictions(img, result, args.conf)
        total_predicted_any += len(result.boxes) if result.boxes is not None else 0
        total_pred_above_conf += n_pred
        total_gt += gt_count

        caption = f"GT: {gt_count} | Pred above conf>={args.conf}: {n_pred} | Pred total: {len(result.boxes)}"
        out_img = add_caption(img, caption)
        cv2.imwrite(str(out_dir / img_path.name), out_img)

    print(f"\nProcessed {len(image_files)} images")
    print(f"Total GT bboxes:                   {total_gt}")
    print(f"Total predictions (any conf):      {total_predicted_any}")
    print(f"Total predictions (conf>={args.conf}): {total_pred_above_conf}")
    print(f"\nОткрой: {out_dir}")


if __name__ == "__main__":
    main()
