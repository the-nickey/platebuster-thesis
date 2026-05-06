"""Сэмплирование данных для двух стадий обучения и ручной разметки.

  --stage pretrain     Stratified subsample CCPD до ~56K с приоритетом edge cases.
  --stage annotation   CLIP-diverse-sampling 1500 фото из non-CCPD под CVAT.

Результаты — symlinks (без копирования) в data/processed/.

Использование:
    python scripts/sample_for_annotation.py --stage pretrain
    python scripts/sample_for_annotation.py --stage annotation
    python scripts/sample_for_annotation.py --stage annotation \\
        --regions russian european --device mps

CLIP-эмбеддинги (ViT-B-32 OpenAI weights) считаются один раз на регион
и кэшируются в data/processed/_cache/embeddings_{region}.npy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"


# ─── Конфигурация ────────────────────────────────────────────────────────────

# Часть 1: pretrain-mix из CCPD (бюджет по subset-ам).
PRETRAIN_BUDGETS: dict[str, int] = {
    "ccpd_base": 25_000,
    "ccpd_blur": 5_000,
    "ccpd_challenge": 8_000,
    "ccpd_db": 3_000,
    "ccpd_fn": 4_000,
    "ccpd_rotate": 3_000,
    "ccpd_tilt": 6_000,
    "ccpd_weather": 2_000,
}
PRETRAIN_OUT = DATA / "processed" / "pretrain_mix"


# Часть 3: ручная разметка через CLIP-diverse.
@dataclass(frozen=True)
class RegionConfig:
    name: str
    sources: tuple[Path, ...]   # директории с .jpg/.png
    budget: int                 # сколько фото в очередь на разметку
    splits: tuple[float, float, float]  # train, val, test доли


ANNOTATION_REGIONS: tuple[RegionConfig, ...] = (
    RegionConfig(
        "russian",
        (DATA / "roboflow/russian/train/images",
         DATA / "roboflow/russian/valid/images",
         DATA / "roboflow/russian/test/images"),
        budget=600,
        splits=(0.50, 0.17, 0.33),  # 300 / 100 / 200
    ),
    RegionConfig(
        "european",
        (DATA / "roboflow/european/train/images",
         DATA / "roboflow/european/valid/images",
         DATA / "roboflow/european/test/images"),
        budget=400,
        splits=(0.50, 0.20, 0.30),  # 200 / 80 / 120
    ),
    RegionConfig(
        "openalpr",
        (DATA / "openalpr_raw/endtoend/br",
         DATA / "openalpr_raw/endtoend/eu",
         DATA / "openalpr_raw/endtoend/us",
         DATA / "openalpr_raw/endtoend/usimages"),
        budget=300,
        splits=(0.50, 0.20, 0.30),  # 150 / 60 / 90
    ),
    RegionConfig(
        "generic",
        (DATA / "roboflow/generic/train/images",
         DATA / "roboflow/generic/valid/images",
         DATA / "roboflow/generic/test/images"),
        budget=200,
        splits=(0.50, 0.25, 0.25),  # 100 / 50 / 50
    ),
    RegionConfig(
        "manual",
        (DATA / "manual",),
        budget=10_000,                # «все, что есть»
        splits=(0.70, 0.15, 0.15),
    ),
)
ANNOTATION_OUT = DATA / "processed" / "manual_queue"
EMB_CACHE_DIR = DATA / "processed" / "_cache"


# ─── Stage 1: pretrain subsample ───────────────────────────────────────────


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve())


def stage_pretrain(seed: int) -> None:
    rng = random.Random(seed)
    if PRETRAIN_OUT.exists():
        # быстрая очистка от предыдущих симлинков
        import shutil
        shutil.rmtree(PRETRAIN_OUT)
    PRETRAIN_OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for subset, budget in PRETRAIN_BUDGETS.items():
        src_dir = DATA / "ccpd" / "CCPD2019" / subset
        if not src_dir.exists():
            print(f"⚠  {src_dir} не существует, пропускаю")
            continue
        files = sorted(src_dir.glob("*.jpg"))
        n_take = min(budget, len(files))
        chosen = rng.sample(files, n_take)
        for f in chosen:
            _link(f, PRETRAIN_OUT / subset / f.name)
        summary.append((subset, len(files), n_take))
        print(f"  {subset:>16}  {len(files):>7}  →  {n_take:>6}")

    total_have = sum(have for _, have, _ in summary)
    total_take = sum(take for _, _, take in summary)
    print(f"\n  ИТОГО          {total_have:>7}  →  {total_take:>6}  ({100*total_take/total_have:.1f}%)")
    print(f"📁  {PRETRAIN_OUT}")


# ─── Stage 3: CLIP-diverse-sampling ────────────────────────────────────────


def _list_images(dirs: tuple[Path, ...]) -> list[Path]:
    out: list[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        out.extend(sorted(d.glob("*.jpg")))
        out.extend(sorted(d.glob("*.jpeg")))
        out.extend(sorted(d.glob("*.png")))
        out.extend(sorted(d.glob("*.heic")))
    return out


def _cache_key(files: list[Path]) -> str:
    """Хэшируем (имя+mtime) первых/последних 5 файлов и общий count — быстро и устойчиво."""
    h = hashlib.sha1()
    h.update(str(len(files)).encode())
    sample = files[:5] + files[-5:] if len(files) > 10 else files
    for f in sample:
        try:
            mt = int(f.stat().st_mtime)
        except OSError:
            mt = 0
        h.update(f"{f.name}:{mt}".encode())
    return h.hexdigest()[:12]


def _compute_embeddings(files: list[Path], device: str) -> "np.ndarray":  # noqa: F821
    """Прогнать список изображений через CLIP ViT-B-32 OpenAI и вернуть (N, 512) ndarray."""
    import numpy as np
    import torch
    import open_clip
    from PIL import Image
    from tqdm import tqdm

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.eval().to(device)

    BATCH = 32
    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(files), BATCH), desc="CLIP"):
            batch_paths = files[i:i + BATCH]
            tensors = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(preprocess(img))
                except Exception as e:  # noqa: BLE001
                    print(f"  ⚠  {p.name}: {e}")
                    tensors.append(torch.zeros(3, 224, 224))
            batch = torch.stack(tensors).to(device)
            feat = model.encode_image(batch)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            embs.append(feat.cpu().numpy())
    return np.concatenate(embs, axis=0)


def _diverse_indices(embeddings: "np.ndarray", k: int, seed: int) -> list[int]:  # noqa: F821
    """K-means по эмбеддингам → ближайший к центроиду из каждого кластера.

    Это approx-к-medoids; для 14K×512 — секунды.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    n = len(embeddings)
    if k >= n:
        return list(range(n))

    km = KMeans(n_clusters=k, n_init=4, random_state=seed)
    labels = km.fit_predict(embeddings)
    centers = km.cluster_centers_

    chosen: list[int] = []
    for cl in range(k):
        idxs = np.where(labels == cl)[0]
        if len(idxs) == 0:
            continue
        d = np.linalg.norm(embeddings[idxs] - centers[cl], axis=1)
        chosen.append(int(idxs[d.argmin()]))
    return chosen


def _split_indices(items: list, splits: tuple[float, float, float], seed: int) -> dict[str, list]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * splits[0]))
    n_val = int(round(n * splits[1]))
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def stage_annotation(seed: int, regions_filter: list[str] | None, device: str) -> None:
    EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for cfg in ANNOTATION_REGIONS:
        if regions_filter and cfg.name not in regions_filter:
            continue

        print(f"\n=== {cfg.name.upper()} ===")
        files = _list_images(cfg.sources)
        if not files:
            print(f"  ⚠  нет файлов в {cfg.sources}")
            continue
        print(f"  найдено {len(files)} фото, бюджет {cfg.budget}")

        if cfg.budget >= len(files):
            chosen_idx = list(range(len(files)))
            print("  бюджет ≥ количество — берём всё, без CLIP")
        else:
            cache_path = EMB_CACHE_DIR / f"emb_{cfg.name}_{_cache_key(files)}.npy"
            if cache_path.exists():
                import numpy as np
                embeddings = np.load(cache_path)
                print(f"  ↻ кэш эмбеддингов: {cache_path.name}")
            else:
                embeddings = _compute_embeddings(files, device=device)
                import numpy as np
                np.save(cache_path, embeddings)
                print(f"  💾 эмбеддинги сохранены: {cache_path.name}")
            chosen_idx = _diverse_indices(embeddings, cfg.budget, seed)
            print(f"  выбрано {len(chosen_idx)} разнообразных")

        chosen_files = [files[i] for i in chosen_idx]
        splits = _split_indices(chosen_files, cfg.splits, seed)

        out_root = ANNOTATION_OUT / cfg.name
        if out_root.exists():
            import shutil
            shutil.rmtree(out_root)
        for split, paths in splits.items():
            for p in paths:
                # уникализуем имя по родителю — чтобы не было коллизий между br/eu/us etc.
                stem = f"{p.parent.name}__{p.name}"
                _link(p, out_root / split / stem)
            print(f"  {split:>5}: {len(paths):>4}")

        # Метаданные о выборке (для воспроизводимости)
        manifest = {
            "region": cfg.name,
            "seed": seed,
            "budget": cfg.budget,
            "n_total": len(files),
            "n_chosen": len(chosen_files),
            "method": "clip_vit_b32_openai_kmeans" if cfg.budget < len(files) else "all",
            "splits": {k: len(v) for k, v in splits.items()},
        }
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        summary_rows.append(manifest)

    print("\n📊  Итог по регионам:")
    for m in summary_rows:
        s = m["splits"]
        print(f"  {m['region']:>10}: train={s['train']} val={s['val']} test={s['test']}  (из {m['n_total']})")
    total = sum(m["n_chosen"] for m in summary_rows)
    print(f"\n📁  {ANNOTATION_OUT}\n🎯  Всего на разметку: {total} фото")


# ─── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pretrain", "annotation"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regions", nargs="*", default=None,
                        help="фильтр по регионам для --stage annotation")
    parser.add_argument("--device", default="mps",
                        help="устройство для CLIP: mps / cpu / cuda")
    args = parser.parse_args()

    if args.stage == "pretrain":
        stage_pretrain(seed=args.seed)
    else:
        stage_annotation(seed=args.seed, regions_filter=args.regions, device=args.device)


if __name__ == "__main__":
    main()
