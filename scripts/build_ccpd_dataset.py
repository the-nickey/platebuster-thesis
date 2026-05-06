"""Конвертация CCPD2019 в YOLO-keypoints формат.

CCPD2019 после распаковки имеет следующую структуру:

    data/ccpd/CCPD2019/
    ├── ccpd_base/        # ~200K — train/val
    ├── ccpd_blur/        # test
    ├── ccpd_challenge/   # test
    ├── ccpd_db/          # test
    ├── ccpd_fn/          # test
    ├── ccpd_np/          # фото без номеров (negative samples)
    ├── ccpd_rotate/      # test
    ├── ccpd_tilt/        # test
    ├── ccpd_weather/     # test (необязательный)
    └── splits/           # txt-файлы со split-ами
        ├── train.txt
        ├── val.txt
        └── test.txt

Скрипт создаёт:

    data/processed/ccpd/
    ├── images/{train,val,test}/*.jpg   # симлинки на оригиналы (экономим диск)
    ├── labels/{train,val,test}/*.txt   # YOLO-keypoints разметка (1 файл на изображение)
    └── data.yaml                        # конфиг для Ultralytics

Использование:
    python scripts/build_ccpd_dataset.py
    python scripts/build_ccpd_dataset.py --copy        # копировать вместо симлинков
    python scripts/build_ccpd_dataset.py --limit 1000  # отладка на маленькой выборке
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

# Добавляем src/ в путь, чтобы импортировать наш парсер.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image  # noqa: E402  (after sys.path mutation)

from plates.ccpd import parse_filename, to_yolo_keypoints  # noqa: E402

# ---- Конфигурация split-ов ---------------------------------------------------

CCPD_ROOT = REPO_ROOT / "data" / "ccpd" / "CCPD2019"
OUT_ROOT = REPO_ROOT / "data" / "processed" / "ccpd"

# Согласно README CCPD: ccpd_base разделён на train/val,
# остальные subset-ы используются как test.
TRAIN_SUBSETS = ("ccpd_base",)
TEST_SUBSETS = (
    "ccpd_blur",
    "ccpd_challenge",
    "ccpd_db",
    "ccpd_fn",
    "ccpd_rotate",
    "ccpd_tilt",
    "ccpd_weather",
)

# Из ccpd_base 10% уходит в val, остальное — train.
VAL_FRACTION = 0.10


@dataclass(frozen=True, slots=True)
class _ConvertResult:
    ok: bool
    src: Path
    error: str | None = None


def _process_one(args: tuple[Path, Path, Path, bool]) -> _ConvertResult:
    """Обработать одно изображение: создать label-файл + симлинк/копию.

    Запускается в worker-процессе. Подписан на pickle-able типы.
    """
    src, img_dst, lbl_dst, copy_mode = args
    try:
        # Считываем размер изображения (CCPD стандартно 720x1160, но не все).
        with Image.open(src) as im:
            w, h = im.size

        sample = parse_filename(src.name)
        line = to_yolo_keypoints(sample, img_width=w, img_height=h)

        lbl_dst.write_text(line + "\n", encoding="utf-8")

        if copy_mode:
            import shutil
            shutil.copy2(src, img_dst)
        else:
            if img_dst.exists() or img_dst.is_symlink():
                img_dst.unlink()
            img_dst.symlink_to(src.resolve())

        return _ConvertResult(ok=True, src=src)
    except Exception as exc:  # noqa: BLE001
        return _ConvertResult(ok=False, src=src, error=str(exc))


def _ensure_dirs() -> None:
    for split in ("train", "val", "test"):
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)


def _collect_images(subsets: tuple[str, ...], limit: int | None = None) -> list[Path]:
    files: list[Path] = []
    for sub in subsets:
        sub_dir = CCPD_ROOT / sub
        if not sub_dir.exists():
            print(f"⚠  Subset не найден: {sub_dir} — пропускаю")
            continue
        files.extend(sorted(sub_dir.glob("*.jpg")))
    if limit is not None:
        files = files[:limit]
    return files


def _split_train_val(files: list[Path], val_fraction: float) -> tuple[list[Path], list[Path]]:
    """Детерминированный hash-split: воспроизводимый, не зависит от порядка чтения файлов."""
    import hashlib

    def to_val(p: Path) -> bool:
        h = int(hashlib.md5(p.name.encode()).hexdigest(), 16)  # noqa: S324
        return (h % 10_000) / 10_000.0 < val_fraction

    train, val = [], []
    for f in files:
        (val if to_val(f) else train).append(f)
    return train, val


def _convert_split(
    files: list[Path],
    split_name: str,
    copy_mode: bool,
    workers: int,
) -> tuple[int, int]:
    img_split_dir = OUT_ROOT / "images" / split_name
    lbl_split_dir = OUT_ROOT / "labels" / split_name

    tasks: list[tuple[Path, Path, Path, bool]] = []
    for src in files:
        img_dst = img_split_dir / src.name
        lbl_dst = lbl_split_dir / (src.stem + ".txt")
        tasks.append((src, img_dst, lbl_dst, copy_mode))

    ok_count = 0
    fail_count = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_process_one, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            res = fut.result()
            if res.ok:
                ok_count += 1
            else:
                fail_count += 1
                if fail_count <= 5:
                    print(f"  ✗  {res.src.name}: {res.error}")
            if i % 5_000 == 0:
                print(f"  [{split_name}] {i:>6}/{len(tasks)}  ok={ok_count}  fail={fail_count}")

    print(f"✅  [{split_name}] готово: ok={ok_count}, fail={fail_count}")
    return ok_count, fail_count


def _write_data_yaml() -> None:
    yaml_path = OUT_ROOT / "data.yaml"
    yaml_path.write_text(
        f"""# YOLO-keypoints конфиг датасета CCPD2019
path: {OUT_ROOT.resolve()}
train: images/train
val: images/val
test: images/test

# 1 класс — license_plate
names:
  0: license_plate

# 4 keypoint-а (углы пластины), все видимые (3 значения на точку: x, y, visibility)
kpt_shape: [4, 3]

# Симметричный flip: при горизонтальном отражении TL↔TR и BL↔BR.
# Порядок углов в .txt: TL, TR, BR, BL → индексы 0,1,2,3.
flip_idx: [1, 0, 3, 2]
""",
        encoding="utf-8",
    )
    print(f"📝  data.yaml записан: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Копировать изображения вместо создания симлинков (медленнее, +12 ГБ)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить число изображений в каждом subset для отладки",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Число параллельных процессов (default: 8)",
    )
    args = parser.parse_args()

    if not CCPD_ROOT.exists():
        sys.exit(f"❌  Не найдена директория CCPD2019: {CCPD_ROOT}\nРаспакуй архив и повтори.")

    _ensure_dirs()

    print("📂  Собираю списки изображений…")
    train_pool = _collect_images(TRAIN_SUBSETS, args.limit)
    test_files = _collect_images(TEST_SUBSETS, args.limit)
    train_files, val_files = _split_train_val(train_pool, VAL_FRACTION)

    print(f"   train: {len(train_files):>7}")
    print(f"   val:   {len(val_files):>7}")
    print(f"   test:  {len(test_files):>7}")

    total_ok = 0
    total_fail = 0
    for split_name, files in (
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ):
        ok, fail = _convert_split(files, split_name, args.copy, args.workers)
        total_ok += ok
        total_fail += fail

    _write_data_yaml()

    print()
    print(f"🎯  Итого: ok={total_ok}, fail={total_fail}")
    print(f"📁  Готовый датасет: {OUT_ROOT}")


if __name__ == "__main__":
    main()
