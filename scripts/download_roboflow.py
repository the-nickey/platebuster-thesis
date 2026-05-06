"""Скачать публичные датасеты с Roboflow Universe.

Использование:
    export ROBOFLOW_API_KEY=...    # или положи в .env (см. .env.example)
    python scripts/download_roboflow.py

Скачивает 3 публичных датасета номеров в data/roboflow/:
    - russian/   — Russian License Plates Classification by Type   (~900)
    - european/  — European License Plates                          (~500)
    - generic/   — License Plate Recognition                        (~2000)

Все три приходят в YOLOv8 bbox-формате; доразметка углов делается отдельно
через CVAT (см. infra/cvat/README.md) или собственные annotator-ы из
scripts/annotation/.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data" / "roboflow"


@dataclass(frozen=True, slots=True)
class RoboflowDataset:
    """Координаты публичного Roboflow-проекта для скачивания через SDK."""

    local_name: str            # имя локальной директории под data/roboflow/
    workspace: str             # workspace ID в URL Roboflow Universe
    project: str               # project slug в URL Roboflow Universe
    version: int               # номер версии датасета (см. страницу проекта)
    fmt: str = "yolov8"        # формат экспорта


# 3 датасета из плана главы 1. Версии — последние стабильные на 2026-04.
# Если автор обновит проект, инкрементируйте `version`.
DATASETS: tuple[RoboflowDataset, ...] = (
    # Берём raw (без Roboflow-augmentations) версии — Ultralytics сам
    # делает rotate/blur/brightness/mosaic на трейне, дублировать незачем.
    # Соответствие проверено через `proj.versions()` + .images count.
    RoboflowDataset(
        local_name="russian",
        workspace="testcarplate",
        project="russian-license-plates-classification-by-this-type",
        version=1,        # 2408 фото raw vs v11 7879 c augm
    ),
    RoboflowDataset(
        local_name="european",
        workspace="e-hh49k",
        project="european-license-plates-tjviy",
        version=1,        # 1455 фото (единственная версия)
    ),
    RoboflowDataset(
        local_name="generic",
        workspace="roboflow-universe-projects",
        project="license-plate-recognition-rxg4e",
        version=6,        # 10125 фото raw vs v13 101866 c augm (x10)
    ),
)


def _load_dotenv(env_path: Path) -> None:
    """Минимальный .env loader без зависимости от python-dotenv."""
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_api_key() -> str:
    _load_dotenv(REPO_ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit(
            "❌  ROBOFLOW_API_KEY не задан.\n"
            "   1) Возьми ключ на https://app.roboflow.com/settings/api (Private API Key)\n"
            "   2) Положи в .env  →  ROBOFLOW_API_KEY=xxxx\n"
            "      или экспортни  →  export ROBOFLOW_API_KEY=xxxx\n"
        )
    return key


def download_one(rf, ds: RoboflowDataset) -> Path:
    """Скачать один датасет через Roboflow SDK.

    Roboflow SDK странно реагирует на существующую (даже пустую) target-директорию
    с overwrite=False: иногда "якобы скачивает", но папка остаётся пустой.
    Поэтому удаляем целевую директорию перед скачиванием, кроме случая когда
    в ней уже есть data.yaml (значит — реально скачано).
    """
    target = DATA_ROOT / ds.local_name
    if (target / "data.yaml").exists():
        print(f"⚠  {ds.local_name}: уже скачан в {target}, пропускаю")
        return target

    if target.exists():
        import shutil
        shutil.rmtree(target)

    print(f"⬇  {ds.local_name}: {ds.workspace}/{ds.project}@v{ds.version} → {target}")

    project = rf.workspace(ds.workspace).project(ds.project)
    project.version(ds.version).download(ds.fmt, location=str(target), overwrite=True)

    if not (target / "data.yaml").exists():
        raise RuntimeError(
            f"Roboflow завершил .download(), но {target}/data.yaml не появился. "
            f"Скорее всего проблема с версией v{ds.version} или форматом '{ds.fmt}'."
        )

    print(f"✅  {ds.local_name}: готово")
    return target


def main() -> None:
    try:
        from roboflow import Roboflow
    except ImportError:
        sys.exit(
            "❌  Модуль 'roboflow' не установлен.\n"
            "   .venv/bin/pip install roboflow"
        )

    api_key = _get_api_key()
    DATA_ROOT.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    for ds in DATASETS:
        try:
            download_one(rf, ds)
        except Exception as exc:  # noqa: BLE001
            print(f"❌  {ds.local_name}: {exc}")

    print()
    print(f"📁  Все датасеты: {DATA_ROOT}")


if __name__ == "__main__":
    main()
