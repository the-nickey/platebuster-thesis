"""
Конвертер unified YOLO-keypoints в Roboflow-style COCO для обучения RF-DETR.

Выход:
    data/processed/coco/
    ├── finetune/
    │   ├── train/
    │   │   ├── _annotations.coco.json
    │   │   └── *.jpg (симлинки на unified/finetune/images/train/)
    │   └── valid/
    │       ├── _annotations.coco.json
    │       └── *.jpg
    └── test_per_region/
        ├── ccpd/    (_annotations.coco.json + *.jpg)
        ├── russian/
        ├── european/
        ├── openalpr/
        └── generic/

RF-DETR ожидает именно Roboflow-формат: имена папок `train` / `valid` / `test`,
аннотации лежат рядом с изображениями в `_annotations.coco.json`.
Keypoints в COCO не сохраняем — RF-DETR keypoints не поддерживает (правка 5
chapter1_diff.md), задача только bbox.

Запуск:
    python scripts/training/convert_yolo_to_coco.py
    python scripts/training/convert_yolo_to_coco.py --copy   # копировать вместо симлинков
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from common import REGIONS, TEST_REGIONS_DIR, REPO_ROOT


UNIFIED = REPO_ROOT / "data" / "processed" / "unified"
COCO_ROOT = REPO_ROOT / "data" / "processed" / "coco"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--copy", action="store_true",
                   help="копировать изображения вместо симлинков (медленнее, но переносимо в облако)")
    p.add_argument("--out", default=str(COCO_ROOT),
                   help=f"корень выходной структуры (default: {COCO_ROOT})")
    return p.parse_args()


def yolo_to_coco_box(parts: list[str], W: int, H: int) -> tuple[int, int, int, int] | None:
    if len(parts) < 5:
        return None
    cx, cy, w, h = map(float, parts[1:5])
    bw = w * W
    bh = h * H
    bx = cx * W - bw / 2
    by = cy * H - bh / 2
    if bw <= 0 or bh <= 0:
        return None
    return [bx, by, bw, bh]


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src.resolve(), dst)
    else:
        dst.symlink_to(src.resolve())


def convert_split(images_dir: Path, labels_dir: Path, out_dir: Path, copy: bool) -> dict:
    """Конвертирует один split в Roboflow-COCO структуру."""
    out_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {"description": "license plate detection", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "license_plate", "supercategory": "object"}],
        "images": [],
        "annotations": [],
    }

    img_paths = sorted([p for p in images_dir.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    n_skipped_empty = 0
    n_with_boxes = 0
    ann_id = 1

    for image_id, src in enumerate(img_paths, start=1):
        try:
            with Image.open(src) as im:
                W, H = im.size
        except Exception as e:
            print(f"  не открыть {src.name}: {e}")
            continue

        label_path = labels_dir / (src.stem + ".txt")
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.strip().split()
                box = yolo_to_coco_box(parts, W, H)
                if box is None:
                    continue
                boxes.append(box)

        if not boxes:
            n_skipped_empty += 1
            continue

        n_with_boxes += 1
        dst = out_dir / src.name
        link_or_copy(src, dst, copy)

        coco["images"].append({
            "id": image_id, "file_name": src.name, "width": W, "height": H,
        })
        for bx, by, bw, bh in boxes:
            coco["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [bx, by, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
                "segmentation": [],
            })
            ann_id += 1

    out_json = out_dir / "_annotations.coco.json"
    out_json.write_text(json.dumps(coco), encoding="utf-8")

    return {
        "n_with_boxes": n_with_boxes,
        "n_skipped_empty": n_skipped_empty,
        "n_annotations": len(coco["annotations"]),
    }


def main():
    args = parse_args()
    out_root = Path(args.out)
    print(f"=== convert_yolo_to_coco → {out_root} ===")
    print(f"режим: {'copy' if args.copy else 'symlink'}")

    summary = {}

    # finetune: train + val (RF-DETR называет val-сплит как 'valid')
    finetune = UNIFIED / "finetune"
    for src_split, dst_split in [("train", "train"), ("val", "valid")]:
        images_dir = finetune / "images" / src_split
        labels_dir = finetune / "labels" / src_split
        if not images_dir.exists():
            print(f"  пропуск {src_split}: нет {images_dir}")
            continue
        out = out_root / "finetune" / dst_split
        print(f"\n[finetune/{dst_split}]")
        s = convert_split(images_dir, labels_dir, out, args.copy)
        summary[f"finetune/{dst_split}"] = s
        print(f"  images: {s['n_with_boxes']}, annotations: {s['n_annotations']}, skipped_empty: {s['n_skipped_empty']}")

    # test_per_region — каждый регион отдельной "valid"-папкой,
    # чтобы можно было использовать тот же rfdetr eval API
    for region in REGIONS:
        images_dir = TEST_REGIONS_DIR / region / "images"
        labels_dir = TEST_REGIONS_DIR / region / "labels"
        if not images_dir.exists():
            continue
        out = out_root / "test_per_region" / region
        print(f"\n[test_per_region/{region}]")
        s = convert_split(images_dir, labels_dir, out, args.copy)
        summary[f"test_per_region/{region}"] = s
        print(f"  images: {s['n_with_boxes']}, annotations: {s['n_annotations']}, skipped_empty: {s['n_skipped_empty']}")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nсводка: {out_root / 'build_summary.json'}")


if __name__ == "__main__":
    main()
