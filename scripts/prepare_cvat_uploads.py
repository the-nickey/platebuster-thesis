"""Упаковка фото из manual_queue/ + bbox-разметки в zip-архивы для импорта в CVAT.

После запуска получаем 4 архива в data/processed/cvat_uploads/:

    russian.zip    европейские, openalpr, generic, manual (если есть фото)

Каждый архив имеет YOLOv8-Detection структуру:

    russian.zip
    ├── data.yaml
    ├── train/
    │   ├── images/*.jpg
    │   └── labels/*.txt
    ├── valid/   (mapping val→valid — стандарт CVAT)
    └── test/

Юзер создаёт по одному CVAT-task на регион → загружает zip → CVAT
импортирует bbox-разметку как `license_plate` (rectangle). Поверх каждого
rectangle юзер расставляет skeleton с 4 точками (TL, TR, BR, BL).

См. infra/cvat/README.md — как поднять CVAT локально.
"""

from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402

from plates.openalpr import parse_annotation as parse_openalpr  # noqa: E402
from plates.openalpr import to_yolo_bbox as openalpr_to_yolo  # noqa: E402

DATA = REPO_ROOT / "data"
QUEUE = DATA / "processed" / "manual_queue"
OUT = DATA / "processed" / "cvat_uploads"

# Mapping наших splits → CVAT split-имён (CVAT ожидает train/valid/test).
SPLIT_MAP = {"train": "train", "val": "valid", "test": "test"}


def _resolve_original(symlink: Path) -> Path:
    """Получить путь к оригиналу за симлинком."""
    return symlink.resolve()


def _find_yolo_label(orig_image: Path) -> Path | None:
    """Найти соответствующий .txt label для Roboflow-фото.

    Roboflow структура: {dataset}/{train,valid,test}/{images,labels}/file.txt
    """
    parent_split = orig_image.parent.parent  # train/ valid/ test/
    label_path = parent_split / "labels" / (orig_image.stem + ".txt")
    return label_path if label_path.exists() else None


def _make_label_for_image(orig_image: Path) -> tuple[str, str]:
    """Вернуть (yolo_label_text, source_kind) для оригинального изображения.

    Выходной формат — нормализованный YOLO bbox:  '0 cx cy w h\\n0 cx2 cy2 w2 h2\\n...'
    Все классы перенумерованы в 0 = license_plate (объединяем n_p/p_p Russian
    и License_Plate Generic в один класс — для нашей задачи различие ненужно).

    Возвращает ('', kind) если разметки нет (manual фото).
    """
    # OpenALPR — .txt в той же папке что и .jpg
    if orig_image.parent.parent.name == "endtoend":
        txt = orig_image.with_suffix(".txt")
        if txt.exists():
            sample = parse_openalpr(txt)
            with Image.open(orig_image) as im:
                w, h = im.size
            return openalpr_to_yolo(sample, w, h) + "\n", "openalpr"
        return "", "openalpr_missing"

    # Roboflow YOLO — отдельная папка labels/
    lbl = _find_yolo_label(orig_image)
    if lbl is not None:
        text = lbl.read_text(encoding="utf-8")
        # перенумеровать все классы в 0
        normalised = []
        for line in text.splitlines():
            tokens = line.strip().split()
            if len(tokens) < 5:
                continue
            tokens[0] = "0"
            normalised.append(" ".join(tokens))
        return "\n".join(normalised) + ("\n" if normalised else ""), "roboflow"

    # Manual — без разметки, юзер сразу размечает 4 угла на пустом слайде
    return "", "manual"


def _data_yaml_text(region: str) -> str:
    return f"""# CVAT YOLOv8 Detection import config
# Region: {region}
path: .
train: train/images
val: valid/images
test: test/images
nc: 1
names:
  0: license_plate
"""


def _pack_region(region_dir: Path) -> Path | None:
    region = region_dir.name
    splits_present: dict[str, list[Path]] = {}
    for our_split, cvat_split in SPLIT_MAP.items():
        d = region_dir / our_split
        if not d.exists():
            continue
        files = sorted(p for p in d.iterdir() if p.is_symlink() or p.is_file())
        if files:
            splits_present[cvat_split] = files

    if not splits_present:
        print(f"  ⚠  {region}: нет split-ов, пропускаю")
        return None

    out_zip = OUT / f"{region}.zip"
    OUT.mkdir(parents=True, exist_ok=True)
    if out_zip.exists():
        out_zip.unlink()

    n_imgs = 0
    n_with_labels = 0
    n_empty = 0
    n_missing = 0

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", _data_yaml_text(region))

        for cvat_split, files in splits_present.items():
            for f in files:
                orig = _resolve_original(f)
                arc_img = f"{cvat_split}/images/{f.name}"
                zf.write(orig, arcname=arc_img)

                label_text, kind = _make_label_for_image(orig)
                arc_lbl = f"{cvat_split}/labels/{f.stem}.txt"
                zf.writestr(arc_lbl, label_text)

                n_imgs += 1
                if label_text:
                    n_with_labels += 1
                elif kind == "openalpr_missing":
                    n_missing += 1
                else:
                    n_empty += 1

    size_mb = out_zip.stat().st_size / 1024 / 1024
    print(f"  ✅  {region}: {out_zip.name}  {size_mb:.1f} MB  "
          f"(images={n_imgs}, with_labels={n_with_labels}, empty={n_empty}, missing={n_missing})")
    return out_zip


def main() -> None:
    if not QUEUE.exists():
        sys.exit(f"❌  Не найдена очередь {QUEUE}. Запусти scripts/sample_for_annotation.py --stage annotation")

    if OUT.exists():
        shutil.rmtree(OUT)

    print(f"📦  Упаковка из {QUEUE} → {OUT}\n")
    region_dirs = sorted(p for p in QUEUE.iterdir() if p.is_dir())

    packed: list[Path] = []
    for r in region_dirs:
        z = _pack_region(r)
        if z:
            packed.append(z)

    print()
    print(f"📁  Готовые архивы в  {OUT}/")
    for z in packed:
        print(f"     {z.name}  ({z.stat().st_size / 1024 / 1024:.1f} MB)")
    print()
    print("➡  Дальше: см. infra/cvat/README.md — поднять CVAT и импортировать архивы как tasks")


if __name__ == "__main__":
    main()
