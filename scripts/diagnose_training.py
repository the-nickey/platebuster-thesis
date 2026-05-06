"""
Диагностика обучения: пробуем YOLO11n на чистом roboflow/russian с 1 классом,
без multi-domain mix, без AMP, с явным lr0.

Цель — изолировать причину mAP=0:
  - если mAP > 0.5 после 10 эпох → multi-domain виноват
  - если mAP всё ещё 0 → проблема в AMP/MPS/lr/архитектуре

Запуск:
    python scripts/diagnose_training.py
    python scripts/diagnose_training.py --no-amp --lr 0.001
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--no-amp", action="store_true",
                   help="отключить mixed precision (AMP). На MPS иногда ломает gradients.")
    p.add_argument("--lr", type=float, default=None,
                   help="explicit lr0 (default: auto)")
    p.add_argument("--cache", default="ram")
    return p.parse_args()


def make_russian_only_yaml() -> Path:
    """Russian dataset как ОДИН КЛАСС (унифицируем 0=n_p, 1=p_p → 0)."""
    russian_dir = REPO_ROOT / "data" / "roboflow" / "russian"
    if not russian_dir.exists():
        raise SystemExit(f"нет {russian_dir}")

    # ВАЖНО: исходный data.yaml имеет 2 класса (n_p, p_p).
    # Делаем yaml с 1 классом — YOLO просто маппит class id 0 как есть,
    # но class id 1 НЕ ПОЯВИТСЯ в metric'ах (но bboxes с class=1 будут
    # тренироваться как class 0... нет, на самом деле Ultralytics упадёт на class > nc).
    # Поэтому делаем temporary labels с class=0 для всех.
    return None  # см. ниже — нужен фикс labels


def remap_classes_to_zero(src_dir: Path, dst_dir: Path):
    """Копирует labels из src_dir в dst_dir, заменяя class_id на 0."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_lbl in src_dir.glob("*.txt"):
        text = src_lbl.read_text()
        new_lines = []
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            parts[0] = "0"
            new_lines.append(" ".join(parts))
        (dst_dir / src_lbl.name).write_text("\n".join(new_lines) + "\n")


def prepare_russian_one_class() -> Path:
    """Подготовить временный russian-датасет с 1 классом."""
    src_root = REPO_ROOT / "data" / "roboflow" / "russian"
    dst_root = REPO_ROOT / "data" / "processed" / "_russian_oneclass"

    # симлинк images, копия labels с class=0
    for split in ("train", "valid", "test"):
        src_img = src_root / split / "images"
        src_lbl = src_root / split / "labels"
        if not src_img.exists():
            continue
        dst_img = dst_root / split / "images"
        dst_lbl = dst_root / split / "labels"
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        if not dst_img.exists():
            dst_img.symlink_to(src_img.resolve())
        remap_classes_to_zero(src_lbl, dst_lbl)

    yaml_path = dst_root / "data.yaml"
    yaml_path.write_text(f"""path: {dst_root}
train: train/images
val: valid/images
test: test/images

names:
  0: license_plate
""", encoding="utf-8")
    return yaml_path


def main():
    args = parse_args()

    print("Подготовка russian-only датасета (1 класс)...")
    yaml_path = prepare_russian_one_class()
    print(f"  yaml: {yaml_path}")

    from ultralytics import YOLO
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=0,  # на MPS hardcoded anyway
        amp=not args.no_amp,
        cache=args.cache,
        project="runs",
        name=f"diag_russian_only_amp{not args.no_amp}_lr{args.lr or 'auto'}",
        exist_ok=True,
        plots=True,
    )
    if args.lr is not None:
        train_kwargs["lr0"] = args.lr
        train_kwargs["optimizer"] = "AdamW"

    print(f"\nTraining: amp={not args.no_amp}, lr={args.lr or 'auto'}, "
          f"epochs={args.epochs}, batch={args.batch}\n")

    model = YOLO("yolo11n.pt")
    results = model.train(**train_kwargs)

    print(f"\n=== ИТОГИ ===")
    if hasattr(results, "results_dict"):
        for k, v in results.results_dict.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
