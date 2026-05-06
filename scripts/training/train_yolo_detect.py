"""
Bbox-обучалка YOLO11n / YOLO12n на unified/finetune датасете.

Используется для всех CNN-bbox моделей в сравнении (см. chapter_2_backbone.md §1.2).
Конфигурации параметров — пресет под устройство:
- preset=cuda: AdamW lr=0.001, freeze=None, amp=True, mosaic=1.0, batch=64, workers=8
- preset=mps:  AdamW lr=0.001, freeze=10,   amp=False, mosaic=0.3, batch=32, workers=0
- preset=cpu:  то же что mps, workers=0

Запуск:
    # CUDA (Yandex Cloud A100):
    python scripts/training/train_yolo_detect.py --model yolo11n.pt --epochs 100
    python scripts/training/train_yolo_detect.py --model yolo12n.pt --epochs 100

    # MPS (M3 Pro):
    python scripts/training/train_yolo_detect.py --model yolo11n.pt --preset mps --epochs 100

    # Любой override:
    python scripts/training/train_yolo_detect.py --model yolo11n.pt \\
        --preset cuda --batch 128 --lr0 0.002 --no-amp
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from common import (
    REPO_ROOT, FINETUNE_YAML, REGIONS, TEST_REGIONS_DIR,
    pick_device, timestamp_dir, assert_unified_exists,
)


PRESETS = {
    "cuda": dict(
        epochs=100, batch=64, imgsz=640, workers=8,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        freeze=None, patience=15, amp=True, mosaic=1.0,
        cache="ram",
    ),
    "mps": dict(
        epochs=100, batch=32, imgsz=640, workers=0,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        freeze=10, patience=15, amp=False, mosaic=0.3,
        cache="ram",
    ),
    "cpu": dict(
        epochs=50, batch=16, imgsz=640, workers=0,
        optimizer="AdamW", lr0=0.001, lrf=0.01,
        freeze=10, patience=10, amp=False, mosaic=0.3,
        cache="ram",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt", help="bbox-only pretrained (yolo11n.pt | yolo12n.pt | yolo11s.pt | ...)")
    p.add_argument("--preset", choices=["cuda", "mps", "cpu", "auto"], default="auto",
                   help="набор гиперпараметров под устройство (auto = подобрать по pick_device())")
    p.add_argument("--name", default=None, help="имя run-а в runs/; default — yolo_detect_<timestamp>")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (override pick_device)")

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--cache", default=None, choices=["ram", "disk", "false"])

    p.add_argument("--optimizer", default=None, choices=["AdamW", "SGD", "auto"])
    p.add_argument("--lr0", type=float, default=None)
    p.add_argument("--lrf", type=float, default=None)
    p.add_argument("--freeze", type=int, default=None, help="число backbone-слоёв заморозить (по умолчанию из preset)")
    p.add_argument("--no-freeze", action="store_true", help="принудительно отключить freeze")
    p.add_argument("--mosaic", type=float, default=None)

    amp_group = p.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp_explicit", action="store_const", const=True)
    amp_group.add_argument("--no-amp", dest="amp_explicit", action="store_const", const=False)
    p.set_defaults(amp_explicit=None)

    p.add_argument("--skip-eval", action="store_true", help="не делать per-region eval после обучения")
    return p.parse_args()


def resolve_preset(args) -> tuple[str, dict]:
    device = args.device or pick_device()
    preset_name = args.preset
    if preset_name == "auto":
        preset_name = device if device in PRESETS else "cpu"
    preset = dict(PRESETS[preset_name])

    overrides = {
        "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
        "workers": args.workers, "patience": args.patience, "cache": args.cache,
        "optimizer": args.optimizer, "lr0": args.lr0, "lrf": args.lrf,
        "mosaic": args.mosaic,
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


def _materialize_bbox_split(src_images_dir: Path, src_labels_dir: Path,
                            dst_root: Path, split: str) -> int:
    """Материализует bbox-only детект-сплит в dst_root/{images,labels}/<split>/.

    POST-MORTEM 2026-05-05 (третья итерация):
    - Fix-1 (kpt_shape в yaml): не помог — флаг keypoint в verify_image_label
      определяется типом модели, не yaml-ом. YOLO('yolo11n.pt') = detect →
      keypoint=False → 17-числовые labels проваливаются в segment-fallback,
      vis=2 интерпретируется как координата → corrupt.
    - Fix-2 (симлинки bbox_split/images→../images, labels→../labels_bbox):
      не помог — Ultralytics в data/utils.py:472 делает (path / data[k]).resolve(),
      что разрешает симлинк-папку до реальной finetune/images/, и дальше
      img2label_paths приходит к старым 17-числовым labels.
    - Fix-3 (этот): создаём ФИЗИЧЕСКУЮ папку bbox_split/ с реальными
      директориями images/ и labels/ (не симлинками). Картинки — hardlinks
      на оригинальные JPG (бесплатно по диску, тот же inode). Labels —
      реальные txt-файлы с 5 первыми числами (cls cx cy w h). Path.resolve()
      теперь сохраняет путь bbox_split/images/<split>/foo.jpg, и подстановка
      images→labels приводит к bbox_split/labels/<split>/foo.txt — нашим
      5-числовым labels.

    Источник 17-числовых labels (finetune/labels/) не трогается — pose v2
    и любые pose-обучения продолжают работать на полном keypoint-датасете.

    Идемпотентно: если в dst-папках столько же файлов сколько в src — пропуск.
    Удали bbox_split/ для force-rebuild."""
    dst_img = dst_root / "images" / split
    dst_lbl = dst_root / "labels" / split

    n_src_imgs = sum(1 for f in src_images_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    n_src_lbls = sum(1 for _ in src_labels_dir.glob("*.txt")) if src_labels_dir.exists() else 0
    if dst_img.exists() and dst_lbl.exists():
        n_dst_imgs = sum(1 for _ in dst_img.iterdir())
        n_dst_lbls = sum(1 for _ in dst_lbl.iterdir())
        if n_dst_imgs == n_src_imgs and n_dst_lbls == n_src_lbls and n_dst_imgs > 0:
            return n_dst_imgs

    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lbl.mkdir(parents=True, exist_ok=True)

    # 1. hardlinks на JPG (Path.resolve() сохранит путь bbox_split/...)
    for src in src_images_dir.iterdir():
        if src.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        target = dst_img / src.name
        if target.exists():
            continue
        try:
            os.link(src, target)
        except OSError:
            # fallback: если os.link не работает (cross-filesystem) — копируем
            import shutil
            shutil.copy2(src, target)

    # 2. реальные labels с урезкой до 5 первых чисел
    if src_labels_dir.exists():
        for src in src_labels_dir.glob("*.txt"):
            target = dst_lbl / src.name
            if target.exists():
                continue
            out_lines = []
            for line in src.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    out_lines.append(" ".join(parts[:5]))
            target.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return sum(1 for _ in dst_img.iterdir())


def make_bbox_yaml_for_finetune() -> Path:
    """finetune-датасет с физической папкой bbox_split/ (hardlinks + 5-числовые labels).

    См. _materialize_bbox_split docstring — пройдено три итерации post-mortem'а
    для разрезолюции bug-а Ultralytics с 17-числовыми pose-labels в detect-режиме.

    Pose-обучение использует оригинальные finetune/labels/ — не трогаем."""
    finetune_root = FINETUNE_YAML.parent
    bbox_root = finetune_root / "bbox_split"
    for split in ("train", "val", "test"):
        src_imgs = finetune_root / "images" / split
        src_lbls = finetune_root / "labels" / split
        if not src_imgs.exists():
            continue
        n = _materialize_bbox_split(src_imgs, src_lbls, bbox_root, split)
        print(f"  bbox_split/{split}: {n} картинок (hardlinks) + 5-числовых labels")

    text = FINETUNE_YAML.read_text(encoding="utf-8")
    out_lines = []
    for line in text.splitlines():
        if line.startswith(("kpt_shape", "flip_idx")):
            continue  # detect-режим без keypoints
        if line.startswith("path:"):
            out_lines.append(f"path: {bbox_root}")
            continue
        out_lines.append(line)
    f = tempfile.NamedTemporaryFile(mode="w", suffix="_detect.yaml", delete=False, encoding="utf-8")
    f.write("\n".join(out_lines) + "\n")
    f.close()
    return Path(f.name)


def make_test_yaml_bbox(region: str) -> Path | None:
    """Per-region eval yaml для detect-моделей. Аналогично make_bbox_yaml_for_finetune
    создаёт test_per_region/<region>/bbox_split/ с hardlinks + 5-числовыми labels."""
    region_dir = TEST_REGIONS_DIR / region
    if not (region_dir / "images").exists():
        return None

    bbox_root = region_dir / "bbox_split"
    n = _materialize_bbox_split(region_dir / "images", region_dir / "labels", bbox_root, "all")
    print(f"  test/{region}/bbox_split: {n} картинок")

    lines = [
        f"path: {bbox_root}",
        "train: images/all",
        "val: images/all",
        "test: images/all",
        "",
        "names:",
        "  0: license_plate",
    ]
    f = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{region}_detect.yaml",
                                    delete=False, encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.close()
    return Path(f.name)


def main():
    args = parse_args()
    assert_unified_exists()

    from ultralytics import YOLO

    device, preset_name, params = resolve_preset(args)

    run_name = args.name or f"yolo_detect_{Path(args.model).stem}_{preset_name}"
    out_dir = timestamp_dir(run_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== train_yolo_detect ===")
    print(f"  model:    {args.model}")
    print(f"  device:   {device}")
    print(f"  preset:   {preset_name}")
    print(f"  out_dir:  {out_dir}")
    print(f"  params:")
    for k, v in sorted(params.items()):
        print(f"    {k}: {v}")
    print()

    yaml_path = make_bbox_yaml_for_finetune()
    print(f"detect-yaml: {yaml_path}")

    train_kwargs = dict(
        data=str(yaml_path),
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

    yaml_path.unlink(missing_ok=True)

    if args.skip_eval:
        print("--skip-eval, выходим")
        return

    print("=== per-region eval ===")
    results = {}
    eval_model = YOLO(str(best))
    for region in REGIONS:
        ry = make_test_yaml_bbox(region)
        if ry is None:
            print(f"  {region}: пропуск (нет {TEST_REGIONS_DIR / region / 'images'})")
            continue
        try:
            metrics = eval_model.val(
                data=str(ry), split="val",
                imgsz=params["imgsz"], batch=params["batch"],
                workers=params["workers"], device=device,
                plots=False, verbose=False,
            )
            results[region] = {
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
            }
            print(f"  {region}: mAP50={results[region]['mAP50']:.3f} "
                  f"mAP50-95={results[region]['mAP50_95']:.3f} "
                  f"P={results[region]['precision']:.3f} R={results[region]['recall']:.3f}")
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
            print(f"  {r:<10} mAP50={m['mAP50']:.3f}")


if __name__ == "__main__":
    main()
