"""Общие утилиты для обучалок: устройство, пути, per-region eval."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import tempfile

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
UNIFIED = REPO_ROOT / "data" / "processed" / "unified"
PRETRAIN_YAML = UNIFIED / "pretrain" / "data.yaml"
FINETUNE_YAML = UNIFIED / "finetune" / "data.yaml"
TEST_REGIONS_DIR = UNIFIED / "test_per_region"

RUNS_ROOT = REPO_ROOT / "runs"

REGIONS = ["ccpd", "russian", "european", "openalpr", "generic"]


def pick_device(prefer_mps: bool = True) -> str:
    """Выбор устройства: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def timestamp_dir(prefix: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_ROOT / f"{prefix}_{ts}"


def make_test_yaml_for_region(region: str, kpt_shape: list[int] | None = None,
                              flip_idx: list[int] | None = None) -> Path:
    """Создаёт временный data.yaml для test_per_region/<region>/.
    Возвращает путь к временному файлу.

    Ultralytics требует data.yaml с train/val/test ссылками. Для eval-only
    указываем images-папку как val (Ultralytics использует val для metrics)."""
    region_dir = TEST_REGIONS_DIR / region
    if not (region_dir / "images").exists():
        return None

    lines = [
        f"path: {region_dir}",
        f"train: images",
        f"val: images",
        f"test: images",
        "",
        "names:",
        "  0: license_plate",
    ]
    if kpt_shape and flip_idx:
        lines += [
            "",
            f"kpt_shape: {kpt_shape}",
            f"flip_idx: {flip_idx}",
        ]

    f = tempfile.NamedTemporaryFile(mode="w", suffix=f"_{region}.yaml",
                                    delete=False, encoding="utf-8")
    f.write("\n".join(lines) + "\n")
    f.close()
    return Path(f.name)


def assert_unified_exists():
    # pretrain (Stage A) опционален в 2-stage Avito-style: его исключают из
    # bundle через --exclude=unified/pretrain ради лимита 5 GB Object Storage,
    # а сами модели обучаются с нуля или с COCO-весов на finetune.
    if not FINETUNE_YAML.exists():
        raise SystemExit(
            f"не найден {FINETUNE_YAML}\n"
            f"запусти `python scripts/build_unified_dataset.py` сначала"
        )
