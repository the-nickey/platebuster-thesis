"""
Нарезает crop'ы плашек из unified-датасета для обучения keypoint-head'а
(2-stage pipeline: bbox-detector → crop → keypoint regression).

Логика:
  - Берёт все labels с visibility=2 (углы размечены руками или в CCPD-имени файла)
  - Crop по bbox + padding (квадратный, чтобы поддерживать любой аспект плашки)
  - Resize в фиксированный размер
  - Преобразует углы из global-normalized координат в local-crop координаты
  - Сохраняет crop.jpg + label.txt с 8 нормализованными числами (x1 y1 ... x4 y4)

Источники (все берутся в train, потому что pretrain ≡ Stage A в новой схеме = просто данные):
  data/processed/unified/pretrain/{train,val}/        ← вся CCPD-разметка
  data/processed/unified/finetune/{train,val,test}/   ← все домены
  data/processed/unified/test_per_region/<region>/    ← per-region eval

Выход:
  data/processed/keypoint_crops/
    ├── train/{images,labels}/
    ├── val/{images,labels}/
    └── test_per_region/{ccpd,russian,european,openalpr,generic}/{images,labels}/

Запуск:
  python scripts/build_keypoint_crops.py
  python scripts/build_keypoint_crops.py --crop-size 224 --padding 0.3
  python scripts/build_keypoint_crops.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
UNIFIED = REPO_ROOT / "data" / "processed" / "unified"
PRETRAIN = UNIFIED / "pretrain"
FINETUNE = UNIFIED / "finetune"
TEST_REGIONS = UNIFIED / "test_per_region"

OUT_ROOT = REPO_ROOT / "data" / "processed" / "keypoint_crops"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crop-size", type=int, default=192,
                   help="размер квадратного crop'а в пикселях")
    p.add_argument("--padding", type=float, default=0.25,
                   help="доля padding'а вокруг bbox при кропе")
    p.add_argument("--jpeg-quality", type=int, default=92)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_pose_label(label_path: Path) -> list[dict]:
    """YOLO-keypoints формат → list of instances."""
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
            kpts = [
                (float(parts[5 + i*3]), float(parts[5 + i*3 + 1]), int(float(parts[5 + i*3 + 2])))
                for i in range(4)
            ]
            out.append({"class": cls, "cx": cx, "cy": cy, "w": w, "h": h, "kpts": kpts})
        except (ValueError, IndexError):
            continue
    return out


def has_kpts(inst: dict) -> bool:
    return all(v == 2 for _, _, v in inst["kpts"])


def crop_with_corners(img: np.ndarray, inst: dict, padding: float, target_size: int):
    """Возвращает (crop_resized, kpts_local_normalized) или None если crop пустой."""
    h, w = img.shape[:2]
    cx, cy, bw, bh = inst["cx"] * w, inst["cy"] * h, inst["w"] * w, inst["h"] * h

    # квадратный crop по max-стороне + padding (чтобы supporting любой аспект плашки)
    side = max(bw, bh) * (1 + 2 * padding)
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))

    # clamp в границы изображения
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)
    if x2c - x1c < 8 or y2c - y1c < 8:
        return None

    crop = img[y1c:y2c, x1c:x2c]

    # если упирались в края — паддим серым до квадрата
    cw_actual = x2c - x1c
    ch_actual = y2c - y1c
    if cw_actual != ch_actual:
        max_side = max(cw_actual, ch_actual)
        canvas = np.full((max_side, max_side, 3), 114, dtype=np.uint8)
        offset_x = (max_side - cw_actual) // 2
        offset_y = (max_side - ch_actual) // 2
        canvas[offset_y:offset_y + ch_actual, offset_x:offset_x + cw_actual] = crop
        crop = canvas
        crop_origin_x = x1c - offset_x
        crop_origin_y = y1c - offset_y
        crop_side = max_side
    else:
        crop_origin_x = x1c
        crop_origin_y = y1c
        crop_side = cw_actual

    crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # углы → локальные нормализованные [0..1]
    kpts_local = []
    for kx, ky, _ in inst["kpts"]:
        ax = kx * w  # абс
        ay = ky * h
        lx = (ax - crop_origin_x) / crop_side  # [0..1] в квадратном crop
        ly = (ay - crop_origin_y) / crop_side
        kpts_local.append((lx, ly))

    # sanity: точка должна быть в [0..1]
    for lx, ly in kpts_local:
        if not (-0.05 <= lx <= 1.05 and -0.05 <= ly <= 1.05):
            return None  # угол сильно вышел — кропа недостаточно, отбрасываем

    return crop_resized, kpts_local


def write_crop(crop: np.ndarray, kpts_local: list[tuple[float, float]],
               out_imgs: Path, out_lbls: Path, base_stem: str, idx: int,
               jpeg_quality: int):
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)
    name = f"{base_stem}__{idx}"
    cv2.imwrite(str(out_imgs / f"{name}.jpg"), crop,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    line = " ".join(f"{v:.6f}" for pt in kpts_local for v in pt)
    (out_lbls / f"{name}.txt").write_text(line, encoding="utf-8")


def iter_pose_labels(images_dir: Path, labels_dir: Path) -> Iterable[tuple[Path, list[dict]]]:
    if not images_dir.exists() or not labels_dir.exists():
        return
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        instances = parse_pose_label(labels_dir / f"{img_path.stem}.txt")
        if instances:
            yield img_path, instances


def process_pool(name: str, images_dir: Path, labels_dir: Path,
                 out_imgs: Path, out_lbls: Path, args, dry_run: bool) -> dict:
    stats = {"images_seen": 0, "instances_seen": 0, "with_kpts": 0,
             "crops_written": 0, "skipped_bad_crop": 0}

    # сначала соберём список — он быстрый, чтобы tqdm знал total
    pairs = list(iter_pose_labels(images_dir, labels_dir))
    if not pairs:
        print(f"  [{name:<20}] (нет данных)")
        return stats
    try:
        from tqdm import tqdm
        iterator = tqdm(pairs, desc=f"  [{name}]", unit="img")
    except ImportError:
        iterator = pairs

    for img_path, instances in iterator:
        stats["images_seen"] += 1
        stats["instances_seen"] += len(instances)
        kpt_instances = [inst for inst in instances if has_kpts(inst)]
        stats["with_kpts"] += len(kpt_instances)

        if not kpt_instances or dry_run:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for idx, inst in enumerate(kpt_instances):
            result = crop_with_corners(img, inst, args.padding, args.crop_size)
            if result is None:
                stats["skipped_bad_crop"] += 1
                continue
            crop, kpts_local = result
            write_crop(crop, kpts_local, out_imgs, out_lbls,
                       img_path.stem, idx, args.jpeg_quality)
            stats["crops_written"] += 1

    print(f"  [{name:<20}] images={stats['images_seen']:>6} | with_kpts_inst={stats['with_kpts']:>6} | "
          f"crops={stats['crops_written']:>6} | bad_crops_skipped={stats['skipped_bad_crop']}")
    return stats


def main():
    args = parse_args()

    if not args.dry_run and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    train_imgs = OUT_ROOT / "train" / "images"
    train_lbls = OUT_ROOT / "train" / "labels"
    val_imgs = OUT_ROOT / "val" / "images"
    val_lbls = OUT_ROOT / "val" / "labels"

    print(f"crop_size={args.crop_size}, padding={args.padding}, dry_run={args.dry_run}")
    print(f"\n=== TRAIN crops (CCPD pretrain + russian/ccpd_anchor с углами) ===")

    # train pool: pretrain/train + finetune/train (vis=2 only автоматом по has_kpts)
    process_pool("pretrain/train", PRETRAIN / "images" / "train",
                 PRETRAIN / "labels" / "train", train_imgs, train_lbls, args, args.dry_run)
    process_pool("finetune/train", FINETUNE / "images" / "train",
                 FINETUNE / "labels" / "train", train_imgs, train_lbls, args, args.dry_run)

    print(f"\n=== VAL crops ===")
    process_pool("pretrain/val", PRETRAIN / "images" / "val",
                 PRETRAIN / "labels" / "val", val_imgs, val_lbls, args, args.dry_run)
    process_pool("finetune/val", FINETUNE / "images" / "val",
                 FINETUNE / "labels" / "val", val_imgs, val_lbls, args, args.dry_run)

    print(f"\n=== TEST per-region crops ===")
    for region in ("ccpd", "russian", "european", "openalpr", "generic"):
        region_dir = TEST_REGIONS / region
        out_r_imgs = OUT_ROOT / "test_per_region" / region / "images"
        out_r_lbls = OUT_ROOT / "test_per_region" / region / "labels"
        process_pool(f"test_per_region/{region}", region_dir / "images",
                     region_dir / "labels", out_r_imgs, out_r_lbls, args, args.dry_run)

    if args.dry_run:
        print("\n>>> DRY RUN — ничего не записано")
    else:
        # суммарная статистика
        n_train = len(list(train_imgs.glob("*.jpg"))) if train_imgs.exists() else 0
        n_val = len(list(val_imgs.glob("*.jpg"))) if val_imgs.exists() else 0
        print(f"\n=== ИТОГО ===")
        print(f"train: {n_train}")
        print(f"val:   {n_val}")
        for region in ("ccpd", "russian", "european", "openalpr", "generic"):
            n = len(list((OUT_ROOT / "test_per_region" / region / "images").glob("*.jpg")))
            print(f"test_per_region/{region}: {n}")


if __name__ == "__main__":
    main()
