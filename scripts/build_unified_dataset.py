"""
Сборщик финального unified-датасета для two-stage обучения.

Создаёт две раздельные сборки:

  data/processed/unified/pretrain/   — Stage A (CCPD-only keypoints)
    images/{train,val}/   симлинки на pretrain_mix (~56K) + ccpd val sample
    labels/{train,val}/   копии YOLO-keypoints labels из processed/ccpd/

  data/processed/unified/finetune/   — Stage B (multi-domain)
    images/{train,val,test}/  симлинки на russian/european/generic/openalpr/ccpd
    labels/{train,val,test}/  YOLO-keypoints с visibility=2 если углы есть, иначе vis=0

  data/processed/unified/test_per_region/   — изолированные test-сплиты по доменам
    {ccpd,russian,european,openalpr,generic}/{images,labels}/

Логика:
  - класс везде унифицируется в `0: license_plate` (russian имеет 0=n_p и 1=p_p, оба → 0)
  - русским углам делается IoU-матчинг с bbox; матч → vis=2, без матча → vis=0
  - CCPD-pretrain исключается из CCPD-test, чтобы не было утечки
  - всё работает через симлинки (без копирования)
  - идемпотентен: пере-запуск переписывает unified/ с нуля

Запуск:
    python scripts/build_unified_dataset.py
    python scripts/build_unified_dataset.py --dry-run
    python scripts/build_unified_dataset.py --ccpd-anchor-train 8000 --generic-train 5000
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data"
PROCESSED = DATA / "processed"

# источники
CCPD_DIR = PROCESSED / "ccpd"
PRETRAIN_MIX_DIR = PROCESSED / "pretrain_mix"
ROBOFLOW_DIR = DATA / "roboflow"
OPENALPR_DIR = PROCESSED / "openalpr"

# выход
UNIFIED = PROCESSED / "unified"
PRETRAIN_OUT = UNIFIED / "pretrain"
FINETUNE_OUT = UNIFIED / "finetune"
TEST_REGIONS_OUT = UNIFIED / "test_per_region"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# сколько keypoints'ов в YOLO-формате: TL, TR, BR, BL
KPT_SHAPE = (4, 3)
FLIP_IDX = [1, 0, 3, 2]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ccpd-anchor-train", type=int, default=5000,
                   help="сколько CCPD добавить в Stage B train (anchor против забывания)")
    p.add_argument("--ccpd-anchor-val", type=int, default=1000)
    p.add_argument("--ccpd-anchor-test", type=int, default=2000,
                   help="сколько CCPD взять в test_per_region/ccpd")
    p.add_argument("--pretrain-val", type=int, default=2000,
                   help="сколько CCPD val в Stage A val")
    p.add_argument("--generic-train", type=int, default=3000)
    p.add_argument("--generic-val", type=int, default=500)
    p.add_argument("--generic-test", type=int, default=500)
    p.add_argument("--iou-threshold", type=float, default=0.3,
                   help="порог IoU для матчинга bbox↔corners в russian")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ---------- утилиты ----------

def list_images(d: Path) -> list[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def stem_index(paths: Iterable[Path]) -> dict[str, Path]:
    return {p.stem: p for p in paths}


def make_symlink(src: Path, dst: Path):
    """Создаёт симлинк src→dst, перезаписывая если есть."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def write_label(path: Path, lines: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def parse_yolo_bboxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    """Читает YOLO bbox-строки → [(class, cx, cy, w, h), ...]"""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
        out.append((cls, cx, cy, w, h))
    return out


def parse_corners(corners_path: Path) -> list[tuple[float, float, float, float, float, float, float, float]]:
    """Читает файл углов (одна или несколько строк по 8 чисел: TLx TLy TRx TRy BRx BRy BLx BLy)."""
    if not corners_path.exists() or corners_path.name.startswith("_"):
        return []
    out = []
    for line in corners_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        try:
            out.append(tuple(map(float, parts)))
        except ValueError:
            continue
    return out


def bbox_xyxy(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def corners_to_bbox(c: tuple) -> tuple[float, float, float, float]:
    xs = (c[0], c[2], c[4], c[6])
    ys = (c[1], c[3], c[5], c[7])
    return (min(xs), min(ys), max(xs), max(ys))


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    union = ua + ub - inter
    return inter / union if union > 0 else 0.0


def match_corners_to_bboxes(
    bboxes: list[tuple[int, float, float, float, float]],
    corners_list: list[tuple],
    iou_thr: float,
) -> dict[int, tuple]:
    """Greedy IoU-матчинг: возвращает {bbox_idx: corners_8float}."""
    if not bboxes or not corners_list:
        return {}

    pairs = []
    for ci, c in enumerate(corners_list):
        c_box = corners_to_bbox(c)
        for bi, (_, cx, cy, w, h) in enumerate(bboxes):
            score = iou(c_box, bbox_xyxy(cx, cy, w, h))
            if score >= iou_thr:
                pairs.append((score, ci, bi))
    pairs.sort(reverse=True)

    used_b, used_c, result = set(), set(), {}
    for _, ci, bi in pairs:
        if ci in used_c or bi in used_b:
            continue
        result[bi] = corners_list[ci]
        used_c.add(ci)
        used_b.add(bi)
    return result


def make_pose_label_lines(
    bboxes: list[tuple[int, float, float, float, float]],
    bbox_to_corners: dict[int, tuple],
    target_class: int = 0,
) -> list[str]:
    """Собирает YOLO-keypoints строки (1 класс, 4 keypoints с visibility)."""
    lines = []
    for bi, (_, cx, cy, w, h) in enumerate(bboxes):
        c = bbox_to_corners.get(bi)
        if c is None:
            kpts = "0 0 0 0 0 0 0 0 0 0 0 0"  # 4 точки × (x, y, vis=0)
        else:
            kpts = " ".join([
                f"{c[0]:.6f} {c[1]:.6f} 2",
                f"{c[2]:.6f} {c[3]:.6f} 2",
                f"{c[4]:.6f} {c[5]:.6f} 2",
                f"{c[6]:.6f} {c[7]:.6f} 2",
            ])
        lines.append(f"{target_class} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {kpts}")
    return lines


# ---------- индексация CCPD ----------

def index_ccpd_labels() -> dict[str, Path]:
    """Stem → label_path. Ищем во всех split'ах (train/val/test) — pretrain_mix
    был seeded по всему пулу, лейблы могут лежать где угодно."""
    idx: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for p in (CCPD_DIR / "labels" / split).glob("*.txt"):
            idx[p.stem] = p
    return idx


def index_ccpd_images() -> dict[str, Path]:
    idx: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        for p in (CCPD_DIR / "images" / split).glob("*.jpg"):
            idx[p.stem] = p
    return idx


# ---------- сборка ----------

class Builder:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.rng = random.Random(args.seed)
        self.stats: dict[str, int] = {}

    def log(self, key: str, n: int):
        self.stats[key] = self.stats.get(key, 0) + n

    def reset_output(self):
        if self.args.dry_run:
            return
        if UNIFIED.exists():
            shutil.rmtree(UNIFIED)
        for d in (PRETRAIN_OUT, FINETUNE_OUT, TEST_REGIONS_OUT):
            d.mkdir(parents=True, exist_ok=True)

    # ---- Stage A pretrain ----

    def build_pretrain(self):
        print("\n[Stage A pretrain] CCPD-keypoints")
        ccpd_labels = index_ccpd_labels()

        # train: pretrain_mix/*/*.jpg
        pretrain_mix_imgs = [p for sub in PRETRAIN_MIX_DIR.iterdir() if sub.is_dir()
                             for p in sub.glob("*.jpg")]
        print(f"  pretrain_mix images найдено: {len(pretrain_mix_imgs)}")

        train_imgs_dir = PRETRAIN_OUT / "images" / "train"
        train_lbls_dir = PRETRAIN_OUT / "labels" / "train"

        attached, missing = 0, 0
        for img in pretrain_mix_imgs:
            label = ccpd_labels.get(img.stem)
            if label is None:
                missing += 1
                continue
            if not self.args.dry_run:
                make_symlink(img, train_imgs_dir / img.name)
                make_symlink(label, train_lbls_dir / label.name)
            attached += 1
        self.log("pretrain_train", attached)
        print(f"  train: {attached} (без labels: {missing})")

        # val: subsample из ccpd/val, ИСКЛЮЧАЯ всё что попало в pretrain train
        used_stems = {img.stem for img in pretrain_mix_imgs}
        ccpd_val_imgs = list_images(CCPD_DIR / "images" / "val")
        ccpd_val_clean = [p for p in ccpd_val_imgs if p.stem not in used_stems]

        n_val = min(self.args.pretrain_val, len(ccpd_val_clean))
        sample = self.rng.sample(ccpd_val_clean, n_val) if ccpd_val_clean else []

        val_imgs_dir = PRETRAIN_OUT / "images" / "val"
        val_lbls_dir = PRETRAIN_OUT / "labels" / "val"
        for img in sample:
            label = ccpd_labels.get(img.stem)
            if label is None:
                continue
            if not self.args.dry_run:
                make_symlink(img, val_imgs_dir / img.name)
                make_symlink(label, val_lbls_dir / label.name)
        self.log("pretrain_val", len(sample))
        print(f"  val: {len(sample)} (исключая {len(ccpd_val_imgs) - len(ccpd_val_clean)} из train)")

        # data.yaml
        if not self.args.dry_run:
            self._write_data_yaml(
                PRETRAIN_OUT / "data.yaml",
                PRETRAIN_OUT,
                {"train": "images/train", "val": "images/val"},
            )

    # ---- Stage B finetune ----

    def build_finetune(self):
        print("\n[Stage B finetune] multi-domain")
        # train
        self._build_finetune_split("train")
        # val
        self._build_finetune_split("val")
        # test (один общий, плюс per-region копии)
        self._build_finetune_split("test")

        if not self.args.dry_run:
            self._write_data_yaml(
                FINETUNE_OUT / "data.yaml",
                FINETUNE_OUT,
                {"train": "images/train", "val": "images/val", "test": "images/test"},
            )

    def _build_finetune_split(self, split: str):
        print(f"\n  -- {split} --")
        out_imgs = FINETUNE_OUT / "images" / split
        out_lbls = FINETUNE_OUT / "labels" / split

        # russian (corners на train/valid/test)
        rus_n = self._add_roboflow(
            ROBOFLOW_DIR / "russian", out_imgs, out_lbls, split,
            prefix="rus", name="russian",
        )
        print(f"    russian:  {rus_n}")

        # european (corners на valid/test, train без углов → автом. bbox-only)
        eur_n = self._add_roboflow(
            ROBOFLOW_DIR / "european", out_imgs, out_lbls, split,
            prefix="eur", name="european",
        )
        print(f"    european: {eur_n}")

        # generic (subsample, corners нет → bbox-only)
        gen_subsample = {
            "train": self.args.generic_train,
            "val": self.args.generic_val,
            "test": self.args.generic_test,
        }[split]
        gen_n = self._add_roboflow(
            ROBOFLOW_DIR / "generic", out_imgs, out_lbls, split,
            prefix="gen", name="generic", subsample=gen_subsample,
        )
        print(f"    generic:  {gen_n}")

        # openalpr (один пул, делим 70/15/15 по seed)
        oalpr_n = self._add_openalpr(out_imgs, out_lbls, split, prefix="oalpr")
        print(f"    openalpr: {oalpr_n}")

        # ccpd anchor (с keypoints!)
        anchor_n = {
            "train": self.args.ccpd_anchor_train,
            "val": self.args.ccpd_anchor_val,
            "test": self.args.ccpd_anchor_test,
        }[split]
        ccpd_n = self._add_ccpd_anchor(out_imgs, out_lbls, split, anchor_n, prefix="ccpd")
        print(f"    ccpd:     {ccpd_n}")

    def _add_roboflow(
        self, dataset_dir: Path, out_imgs: Path, out_lbls: Path,
        split: str, prefix: str, name: str,
        subsample: int | None = None,
    ) -> int:
        """Универсальная сборка из roboflow-структуры (`<split>/{images,labels[,corners]}`).

        Если на диске лежит `<split>/corners/<stem>.txt` — углы подхватываются и
        матчатся с bbox по IoU (visibility=2). Где corners-файла нет (например,
        generic, european/train, openalpr/eu) — graceful fallback в bbox-only
        (visibility=0)."""
        rf_split = {"train": "train", "val": "valid", "test": "test"}[split]
        src = dataset_dir / rf_split
        if not src.exists():
            return 0

        imgs = list_images(src / "images")
        if subsample is not None and subsample < len(imgs):
            imgs = self.rng.sample(imgs, subsample)

        added = 0
        for img in imgs:
            bboxes = parse_yolo_bboxes(src / "labels" / f"{img.stem}.txt")
            if not bboxes:
                continue
            corners = parse_corners(src / "corners" / f"{img.stem}.txt")
            matched = match_corners_to_bboxes(bboxes, corners, self.args.iou_threshold)

            lines = make_pose_label_lines(bboxes, matched, target_class=0)
            new_name = f"{prefix}_{img.name}"
            if not self.args.dry_run:
                make_symlink(img, out_imgs / new_name)
                write_label(out_lbls / f"{prefix}_{img.stem}.txt", lines)
            added += 1

            if matched:
                self.log(f"{name}_{split}_with_corners", 1)
            else:
                self.log(f"{name}_{split}_bbox_only", 1)
        return added

    def _add_openalpr(self, out_imgs: Path, out_lbls: Path, split: str, prefix: str) -> int:
        # собираем все openalpr из br/eu/us, делим стабильно по seed
        # corners есть для br/us (eu пропустили — низкое разрешение)
        all_pairs = []
        for region in ("br", "eu", "us"):
            region_dir = OPENALPR_DIR / region
            for img in list_images(region_dir / "images"):
                lbl = region_dir / "labels" / f"{img.stem}.txt"
                if lbl.exists():
                    corners_path = region_dir / "corners" / f"{img.stem}.txt"
                    all_pairs.append((img, lbl, corners_path, region))

        if not all_pairs:
            return 0

        # стабильный shuffle (детерминированный по seed); добавление corners_path
        # в tuple не меняет перестановку — random.shuffle переставляет ссылки
        # на элементы по их позиции, identity элементов фиксирована.
        rng = random.Random(self.args.seed)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        slices = {
            "train": all_pairs[:train_end],
            "val": all_pairs[train_end:val_end],
            "test": all_pairs[val_end:],
        }

        added = 0
        for img, lbl, corners_path, region in slices[split]:
            bboxes = parse_yolo_bboxes(lbl)
            if not bboxes:
                continue
            corners = parse_corners(corners_path)
            matched = match_corners_to_bboxes(bboxes, corners, self.args.iou_threshold)
            lines = make_pose_label_lines(bboxes, matched, target_class=0)
            new_name = f"{prefix}_{region}_{img.name}"
            new_stem = f"{prefix}_{region}_{img.stem}"
            if not self.args.dry_run:
                make_symlink(img, out_imgs / new_name)
                write_label(out_lbls / f"{new_stem}.txt", lines)
            added += 1
            if matched:
                self.log(f"openalpr_{region}_{split}_with_corners", 1)
            else:
                self.log(f"openalpr_{region}_{split}_bbox_only", 1)
        return added

    def _add_ccpd_anchor(
        self, out_imgs: Path, out_lbls: Path, split: str, n_target: int, prefix: str
    ) -> int:
        ccpd_imgs = index_ccpd_images()
        ccpd_labels = index_ccpd_labels()

        # исключаем то, что попало в pretrain (если pretrain собран)
        pretrain_used: set[str] = set()
        if (PRETRAIN_OUT / "images" / "train").exists():
            pretrain_used = {p.stem for p in (PRETRAIN_OUT / "images" / "train").iterdir()}

        # для anchor берём из ccpd/<split> (val→val, test→test, train→train)
        ccpd_split = {"train": "train", "val": "val", "test": "test"}[split]
        candidates = [p for p in (CCPD_DIR / "images" / ccpd_split).glob("*.jpg")
                      if p.stem not in pretrain_used]

        # дополнительно гарантируем, что для train не пересекаемся с anchor val/test
        # (val/test в ccpd физически разные папки → пересечений и так нет)

        n_target = min(n_target, len(candidates))
        sample = self.rng.sample(candidates, n_target) if candidates else []

        added = 0
        for img in sample:
            lbl = ccpd_labels.get(img.stem)
            if lbl is None:
                continue
            new_name = f"{prefix}_{img.name}"
            new_stem = f"{prefix}_{img.stem}"
            if not self.args.dry_run:
                make_symlink(img, out_imgs / new_name)
                # лейбл уже в правильном YOLO-pose формате — копируем как симлинк
                make_symlink(lbl, out_lbls / f"{new_stem}.txt")
            added += 1
        return added

    # ---- per-region test ----

    def build_test_per_region(self):
        print("\n[test_per_region] изолированные test-сплиты по доменам")
        regions = [
            # (name, dataset_subdir, prefix, subsample)
            ("russian",  "russian",  "rus", None),
            ("european", "european", "eur", None),
            ("generic",  "generic",  "gen", self.args.generic_test),
        ]
        for name, dataset, prefix, subsample in regions:
            out_imgs = TEST_REGIONS_OUT / name / "images"
            out_lbls = TEST_REGIONS_OUT / name / "labels"
            added = self._add_roboflow(
                ROBOFLOW_DIR / dataset, out_imgs, out_lbls, "test",
                prefix=prefix, name=name, subsample=subsample,
            )
            print(f"  {name}: {added}")

        # openalpr test
        out_imgs = TEST_REGIONS_OUT / "openalpr" / "images"
        out_lbls = TEST_REGIONS_OUT / "openalpr" / "labels"
        added = self._add_openalpr(out_imgs, out_lbls, "test", prefix="oalpr")
        print(f"  openalpr: {added}")

        # ccpd test (anchor sample, без пересечения с pretrain)
        out_imgs = TEST_REGIONS_OUT / "ccpd" / "images"
        out_lbls = TEST_REGIONS_OUT / "ccpd" / "labels"
        added = self._add_ccpd_anchor(out_imgs, out_lbls, "test", self.args.ccpd_anchor_test, prefix="ccpd")
        print(f"  ccpd: {added}")

    # ---- yaml ----

    def _write_data_yaml(self, yaml_path: Path, path: Path, splits: dict[str, str]):
        lines = [
            f"# Auto-generated by build_unified_dataset.py",
            f"path: {path}",
        ]
        for k, v in splits.items():
            lines.append(f"{k}: {v}")
        lines += [
            "",
            "names:",
            "  0: license_plate",
            "",
            f"kpt_shape: [{KPT_SHAPE[0]}, {KPT_SHAPE[1]}]",
            f"flip_idx: {FLIP_IDX}",
        ]
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- run ----

    def run(self):
        if self.args.dry_run:
            print(">>> DRY RUN — ничего не пишем\n")
        self.reset_output()
        self.build_pretrain()
        self.build_finetune()
        self.build_test_per_region()

        print("\n=== ИТОГО ===")
        for k, v in sorted(self.stats.items()):
            print(f"  {k}: {v}")

        if not self.args.dry_run:
            (UNIFIED / "build_summary.json").write_text(
                json.dumps({"args": vars(self.args), "stats": self.stats},
                           ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


def main():
    args = parse_args()
    Builder(args).run()


if __name__ == "__main__":
    main()
