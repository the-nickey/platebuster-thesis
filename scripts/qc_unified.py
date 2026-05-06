"""
QC для финального unified-датасета: визуальный sanity check labels перед обучением.

Парсит YOLO-keypoints формат (`class cx cy w h x1 y1 vis x2 y2 vis x3 y3 vis x4 y4 vis`),
рендерит bbox + 4 точки с разной отрисовкой по visibility:
  vis=2 (размечен) — закрашенный кружок с подписью TL/TR/BR/BL
  vis=0 (нет углов, bbox-only) — серый крестик
И отчёт со статистикой: сколько bbox с keypoints, сколько без, сколько per-class.

Запуск:
    python scripts/qc_unified.py --data data/processed/unified/finetune/data.yaml --split train --n 50
    python scripts/qc_unified.py --data data/processed/unified/pretrain/data.yaml --split train --n 30
    python scripts/qc_unified.py --data data/processed/unified/finetune/data.yaml --split test --n 100
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import cv2


REPO_ROOT = Path(__file__).resolve().parent.parent

POINT_NAMES = ["TL", "TR", "BR", "BL"]
POINT_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 215, 255)]  # BGR


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="путь к data.yaml")
    p.add_argument("--split", default="train", choices=["train", "val", "test"])
    p.add_argument("--n", type=int, default=50, help="сколько случайных фото отрендерить")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=str(REPO_ROOT / "qc_output"))
    return p.parse_args()


def parse_data_yaml(yaml_path: Path) -> dict:
    """Минимальный парсер: ловит path/train/val/test/names."""
    text = yaml_path.read_text(encoding="utf-8")
    out = {}
    for key in ("path", "train", "val", "test"):
        m = re.search(rf"^{key}:\s*(.+)$", text, re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    # имена
    names_block = re.search(r"^names:\s*\n((?:\s+.+\n?)+)", text, re.MULTILINE)
    out["names"] = {}
    if names_block:
        for line in names_block.group(1).splitlines():
            m = re.match(r"\s+(\d+):\s*(.+)", line)
            if m:
                out["names"][int(m.group(1))] = m.group(2).strip()
    return out


def parse_pose_label(label_path: Path) -> list[dict]:
    """YOLO-keypoints: class cx cy w h (x y vis)*4 в каждой строке."""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        # 5 (cls + bbox) + 4*3 (kpt) = 17
        if len(parts) < 17:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            kpts = []
            for i in range(4):
                x = float(parts[5 + i * 3])
                y = float(parts[5 + i * 3 + 1])
                vis = int(float(parts[5 + i * 3 + 2]))
                kpts.append((x, y, vis))
            out.append({"class": cls, "cx": cx, "cy": cy, "w": w, "h": h, "kpts": kpts})
        except (ValueError, IndexError):
            continue
    return out


def render(img, instances, class_names):
    h, w = img.shape[:2]
    canvas = img.copy()

    n_with_kp, n_bbox_only = 0, 0
    for inst in instances:
        x1 = int((inst["cx"] - inst["w"] / 2) * w)
        y1 = int((inst["cy"] - inst["h"] / 2) * h)
        x2 = int((inst["cx"] + inst["w"] / 2) * w)
        y2 = int((inst["cy"] + inst["h"] / 2) * h)

        # bbox: magenta = with kpts, gray = bbox-only
        has_kpts = any(v == 2 for _, _, v in inst["kpts"])
        bbox_color = (255, 0, 255) if has_kpts else (128, 128, 128)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), bbox_color, 2)

        cls_name = class_names.get(inst["class"], str(inst["class"]))
        cv2.putText(canvas, cls_name, (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 1, cv2.LINE_AA)

        if has_kpts:
            n_with_kp += 1
            pts_px = [(int(x * w), int(y * h)) for x, y, _ in inst["kpts"]]
            for i, (px, py) in enumerate(pts_px):
                vis = inst["kpts"][i][2]
                if vis == 2:
                    cv2.circle(canvas, (px, py), 6, POINT_COLORS[i], -1)
                    cv2.putText(canvas, POINT_NAMES[i], (px + 6, py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, POINT_COLORS[i], 1, cv2.LINE_AA)
            for i in range(4):
                cv2.line(canvas, pts_px[i], pts_px[(i + 1) % 4], (255, 255, 255), 1)
        else:
            n_bbox_only += 1
            # серая ✕ в центре bbox
            cx_px, cy_px = int(inst["cx"] * w), int(inst["cy"] * h)
            cv2.line(canvas, (cx_px - 8, cy_px - 8), (cx_px + 8, cy_px + 8), (128, 128, 128), 2)
            cv2.line(canvas, (cx_px - 8, cy_px + 8), (cx_px + 8, cy_px - 8), (128, 128, 128), 2)

    return canvas, n_with_kp, n_bbox_only


def main():
    args = parse_args()
    yaml_path = Path(args.data)
    if not yaml_path.exists():
        raise SystemExit(f"не найден {yaml_path}")

    cfg = parse_data_yaml(yaml_path)
    base = Path(cfg.get("path", yaml_path.parent))
    rel = cfg.get(args.split)
    if not rel:
        raise SystemExit(f"split={args.split} нет в {yaml_path}")
    images_dir = (base / rel) if not Path(rel).is_absolute() else Path(rel)
    labels_dir = Path(str(images_dir).replace("/images/", "/labels/").replace("/images", "/labels"))

    if not images_dir.exists() or not labels_dir.exists():
        raise SystemExit(f"не найдены: {images_dir} или {labels_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    print(f"Found {len(images)} images in {images_dir}")

    # глобальная статистика по всем labels
    total_inst = total_kp = total_bbox_only = 0
    cls_counts: dict[int, int] = {}
    for img in images:
        for inst in parse_pose_label(labels_dir / f"{img.stem}.txt"):
            total_inst += 1
            cls_counts[inst["class"]] = cls_counts.get(inst["class"], 0) + 1
            if any(v == 2 for _, _, v in inst["kpts"]):
                total_kp += 1
            else:
                total_bbox_only += 1

    print(f"\n=== STATS ({yaml_path.parent.name}/{args.split}) ===")
    print(f"Instances total: {total_inst}")
    print(f"  with keypoints (vis=2): {total_kp} ({total_kp*100/max(1,total_inst):.1f}%)")
    print(f"  bbox-only (vis=0):      {total_bbox_only} ({total_bbox_only*100/max(1,total_inst):.1f}%)")
    print(f"Classes: {cls_counts}  (names: {cfg.get('names', {})})")

    rng = random.Random(args.seed)
    sample = rng.sample(images, min(args.n, len(images)))

    out_dir = Path(args.out) / f"unified_{yaml_path.parent.name}_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_kp_drawn = n_bbox_only_drawn = 0
    for img_path in sample:
        instances = parse_pose_label(labels_dir / f"{img_path.stem}.txt")
        if not instances:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        overlay, kp_n, bb_n = render(img, instances, cfg.get("names", {}))
        n_kp_drawn += kp_n
        n_bbox_only_drawn += bb_n
        cv2.imwrite(str(out_dir / img_path.name), overlay)

    print(f"\nОтрисовано {len(sample)} фото в {out_dir}")
    print(f"  с keypoints: {n_kp_drawn} bbox, без: {n_bbox_only_drawn}")
    print(f"\nОткрой Finder на {out_dir} в режиме иконок и пробегись глазами:")
    print(f"  - magenta bbox + цветные точки = размеченные углы")
    print(f"  - серый bbox + серый крест = bbox-only (vis=0), это ожидаемо")
    print(f"  - точки должны лежать НА углах плашки, не на фоне")


if __name__ == "__main__":
    main()
