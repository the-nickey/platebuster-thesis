"""Конвертация OpenALPR Benchmark в YOLO bbox формат.

OpenALPR Benchmark содержит ~900 фото из 4 регионов (Brazil, Europe, US, US extra).
Каждый регион имеет .jpg + .txt в одном каталоге; .txt — табулированный
формат «filename, x, y, w, h, plate_text».

Скрипт создаёт:

    data/processed/openalpr/
    ├── br/{images,labels}/
    ├── eu/{images,labels}/
    ├── us/{images,labels}/   (включает usimages)
    └── data.yaml

Symlink-режим (default) экономит диск; --copy для полной копии.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402
from plates.openalpr import parse_annotation, to_yolo_bbox  # noqa: E402

OPENALPR_RAW = REPO_ROOT / "data" / "openalpr_raw" / "endtoend"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "openalpr"

# US extra — объединяем в "us" для чистой стратификации по регионам.
REGIONS = {
    "br": ["br"],
    "eu": ["eu"],
    "us": ["us", "usimages"],
}


def _process_region(region: str, raw_dirs: list[str], copy_mode: bool) -> tuple[int, int]:
    img_out = OUT_ROOT / region / "images"
    lbl_out = OUT_ROOT / region / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    ok, fail = 0, 0
    for raw_name in raw_dirs:
        raw_dir = OPENALPR_RAW / raw_name
        if not raw_dir.exists():
            print(f"⚠  {raw_dir} не найден, пропускаю")
            continue

        for txt_path in sorted(raw_dir.glob("*.txt")):
            try:
                sample = parse_annotation(txt_path)
                src_img = raw_dir / sample.filename
                if not src_img.exists():
                    fail += 1
                    if fail <= 3:
                        print(f"  ✗  {txt_path.name}: нет {sample.filename}")
                    continue

                with Image.open(src_img) as im:
                    w, h = im.size
                line = to_yolo_bbox(sample, w, h)

                # уникализуем имя на случай совпадений между us/usimages
                stem = f"{raw_name}_{src_img.stem}"
                lbl_dst = lbl_out / f"{stem}.txt"
                img_dst = img_out / f"{stem}{src_img.suffix}"

                lbl_dst.write_text(line + "\n", encoding="utf-8")
                if copy_mode:
                    import shutil
                    shutil.copy2(src_img, img_dst)
                else:
                    if img_dst.exists() or img_dst.is_symlink():
                        img_dst.unlink()
                    img_dst.symlink_to(src_img.resolve())

                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if fail <= 3:
                    print(f"  ✗  {txt_path.name}: {exc}")

    print(f"✅  [{region}] ok={ok}, fail={fail}")
    return ok, fail


def _write_data_yaml() -> None:
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_path.write_text(
        f"""# YOLO bbox конфиг для OpenALPR Benchmark.
# Углов нет — это bbox-only датасет, доразметка делается в CVAT отдельно.
path: {OUT_ROOT.resolve()}

# Регионы организованы как поддиректории; финальные splits собираются
# отдельно в build_unified_dataset.py с учётом балансировки.
regions:
  - br/images
  - eu/images
  - us/images

names:
  0: license_plate
""",
        encoding="utf-8",
    )
    print(f"📝  data.yaml: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy", action="store_true")
    args = parser.parse_args()

    if not OPENALPR_RAW.exists():
        sys.exit(f"❌  Не найден {OPENALPR_RAW}. Сначала: git clone https://github.com/openalpr/benchmarks data/openalpr_raw")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total_ok, total_fail = 0, 0
    for region, raw_dirs in REGIONS.items():
        ok, fail = _process_region(region, raw_dirs, args.copy)
        total_ok += ok
        total_fail += fail

    _write_data_yaml()
    print(f"\n🎯  Итого: ok={total_ok}, fail={total_fail}")
    print(f"📁  {OUT_ROOT}")


if __name__ == "__main__":
    main()
