"""Скачивание CCPD2019 с Google Drive.

Использование:
    python scripts/download_ccpd.py [--green]

Без флагов: скачивает CCPD2019 (~8 ГБ архив, ~12 ГБ распакованных, ~250K фото).
С --green: скачивает дополнительно CCPD-Green (электромобили, ~11K фото).

Источник: https://github.com/detectRecog/CCPD
Лицензия датасета: MIT.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path

try:
    import gdown
except ImportError:
    sys.exit(
        "Error: модуль 'gdown' не найден.\n"
        "Активируй venv и поставь:  pip install gdown"
    )

# Google Drive ID-шники из README CCPD-репозитория
CCPD2019_GDRIVE_ID = "1rdEsCUcIUaYOVRkx5IMTRNA7PcGMmSgc"
CCPD_GREEN_GDRIVE_ID = "1m8w1kFxnCEiqz_-t2vTcgrgqNIv986PR"

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "ccpd"


def download(gdrive_id: str, output: Path) -> None:
    """Скачать файл с Google Drive по ID через gdown (Python API).

    Использует resume=True — восстановление прерванных загрузок по диапазонам,
    важно для большого CCPD2019.tar.xz (~8 ГБ) и нестабильных соединений.
    """
    print(f"⬇  Скачиваю {gdrive_id} -> {output}")
    gdown.download(
        id=gdrive_id,
        output=str(output),
        quiet=False,
        resume=True,
    )


def extract(archive: Path, target_dir: Path) -> None:
    """Распаковать tar.xz архив в целевую директорию."""
    print(f"📦  Распаковываю {archive.name} -> {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:xz") as tf:
        tf.extractall(target_dir, filter="data")
    print(f"✅  Готово: {target_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать CCPD2019")
    parser.add_argument(
        "--green",
        action="store_true",
        help="Также скачать CCPD-Green (электромобили, ~11K фото)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Не распаковывать архив автоматически",
    )
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # CCPD2019
    archive = DATA_DIR / "CCPD2019.tar.xz"
    if archive.exists():
        print(f"⚠  {archive.name} уже существует, пропускаю скачивание")
    else:
        download(CCPD2019_GDRIVE_ID, archive)

    if not args.no_extract:
        if (DATA_DIR / "CCPD2019").exists():
            print("⚠  CCPD2019/ уже распакован, пропускаю")
        else:
            extract(archive, DATA_DIR)

    # CCPD-Green (опционально)
    if args.green:
        green_archive = DATA_DIR / "ccpd_green.tar.xz"
        if green_archive.exists():
            print(f"⚠  {green_archive.name} уже существует, пропускаю")
        else:
            download(CCPD_GREEN_GDRIVE_ID, green_archive)
        if not args.no_extract and not (DATA_DIR / "ccpd_green").exists():
            extract(green_archive, DATA_DIR)


if __name__ == "__main__":
    main()
