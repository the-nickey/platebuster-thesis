"""
Десктопный аннотатор 4 углов номера через cv2.

Запуск:
    python corner_annotator.py --dataset russian --split valid
    python corner_annotator.py --dataset russian --split test
    python corner_annotator.py --dataset openalpr --split eu

Клики: TL → TR → BR → BL.
Клавиши: u — отменить последний клик, s — пропустить, q — выйти.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent

ROBOFLOW_DATASETS = {
    "russian": REPO_ROOT / "data" / "roboflow" / "russian",
    "european": REPO_ROOT / "data" / "roboflow" / "european",
    "generic": REPO_ROOT / "data" / "roboflow" / "generic",
}

OPENALPR_REGIONS = {
    "br": REPO_ROOT / "data" / "processed" / "openalpr" / "br",
    "eu": REPO_ROOT / "data" / "processed" / "openalpr" / "eu",
    "us": REPO_ROOT / "data" / "processed" / "openalpr" / "us",
}

SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

POINT_NAMES = ["TL", "TR", "BR", "BL"]
POINT_COLORS = [
    (0, 0, 255),    # TL red
    (0, 200, 0),    # TR green
    (255, 0, 0),    # BR blue
    (0, 215, 255),  # BL yellow
]


def resolve_split_dir(dataset: str, split: str) -> Path:
    if dataset == "openalpr":
        if split not in OPENALPR_REGIONS:
            raise SystemExit(f"для openalpr split должен быть одним из {list(OPENALPR_REGIONS)}")
        return OPENALPR_REGIONS[split]
    if dataset not in ROBOFLOW_DATASETS:
        raise SystemExit(f"неизвестный dataset {dataset!r}")
    if split not in SPLIT_ALIASES:
        raise SystemExit(f"неизвестный split {split!r}")
    return ROBOFLOW_DATASETS[dataset] / SPLIT_ALIASES[split]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="russian",
                   choices=["russian", "european", "generic", "openalpr"])
    p.add_argument("--split", default="train",
                   help="train/valid/test для roboflow; br/eu/us для openalpr")
    p.add_argument("--min-bbox-px", type=int, default=30,
                   help="не показывать фото, где самый крупный bbox по короткой стороне меньше порога. "
                        "не прошедшие добавляются в _skipped.txt и попадают в датасет как bbox-only.")
    p.add_argument("--min-image-px", type=int, default=480)
    return p.parse_args()


def find_images(images_dir: Path) -> list[Path]:
    return sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def get_skipped_stems(skipped_file: Path) -> set[str]:
    if not skipped_file.exists():
        return set()
    return {line.strip() for line in skipped_file.read_text(encoding="utf-8").splitlines() if line.strip()}


def save_points(out_path: Path, points, image_shape):
    h, w = image_shape[:2]
    normalized = []
    for x, y in points:
        normalized.append(x / w)
        normalized.append(y / h)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(" ".join(f"{v:.6f}" for v in normalized), encoding="utf-8")
    print(f"Сохранено: {out_path.name}")


def append_skip(skipped_file: Path, stem: str):
    skipped_file.parent.mkdir(parents=True, exist_ok=True)
    with skipped_file.open("a", encoding="utf-8") as f:
        f.write(stem + "\n")


def draw_overlay(img, points):
    canvas = img.copy()
    for i, (x, y) in enumerate(points):
        color = POINT_COLORS[i]
        cv2.circle(canvas, (x, y), 6, color, -1)
        cv2.putText(canvas, POINT_NAMES[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(canvas, points[i], points[i + 1], (255, 255, 255), 2)
    if len(points) == 4:
        cv2.line(canvas, points[3], points[0], (255, 255, 255), 2)

    next_name = POINT_NAMES[len(points)] if len(points) < 4 else "SAVE"
    text = f"Click {len(points) + 1}/4: {next_name} | u=undo, s=skip, q=quit"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(canvas, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resize_for_screen(img, max_width=1200, max_height=850):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    if scale == 1.0:
        return img, 1.0
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale


def annotate_image(image_path: Path, out_dir: Path, skipped_file: Path):
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"Не удалось открыть: {image_path}")
        return "next"

    shown_img, scale = resize_for_screen(original)
    points_shown: list[tuple[int, int]] = []
    window_name = "corner annotator"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points_shown) < 4:
            points_shown.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        cv2.imshow(window_name, draw_overlay(shown_img, points_shown))
        key = cv2.waitKey(20) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            return "quit"
        if key == ord("s"):
            append_skip(skipped_file, image_path.stem)
            print(f"Пропущено: {image_path.name}")
            return "next"
        if key == ord("u") and points_shown:
            points_shown.pop()

        if len(points_shown) == 4:
            points_original = [(int(x / scale), int(y / scale)) for x, y in points_shown]
            save_points(out_dir / f"{image_path.stem}.txt", points_original, original.shape)
            return "next"


def auto_skip_too_small(images_dir: Path, labels_dir: Path, skipped_file: Path,
                        out_dir: Path, min_image_px: int, min_bbox_px: int) -> int:
    """Прогон фильтров до начала разметки. Не прошедшие — в _skipped.txt."""
    existing = {p.stem for p in out_dir.glob("*.txt") if not p.name.startswith("_")}
    skipped = get_skipped_stems(skipped_file)
    new_skips: list[str] = []

    for img_path in find_images(images_dir):
        if img_path.stem in existing or img_path.stem in skipped:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            new_skips.append(img_path.stem)
            continue
        h, w = img.shape[:2]
        if min(w, h) < min_image_px:
            new_skips.append(img_path.stem)
            continue

        lbl = labels_dir / f"{img_path.stem}.txt"
        if not lbl.exists():
            new_skips.append(img_path.stem)
            continue

        max_bbox_short = 0
        for line in lbl.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _, _, _, bw_n, bh_n = parts[:5]
                bw, bh = float(bw_n) * w, float(bh_n) * h
                max_bbox_short = max(max_bbox_short, min(bw, bh))
            except ValueError:
                continue

        if max_bbox_short < min_bbox_px:
            new_skips.append(img_path.stem)

    if new_skips:
        skipped_file.parent.mkdir(parents=True, exist_ok=True)
        with skipped_file.open("a", encoding="utf-8") as f:
            for s in new_skips:
                f.write(s + "\n")
    return len(new_skips)


def main():
    args = parse_args()
    split_dir = resolve_split_dir(args.dataset, args.split)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    out_dir = split_dir / "corners"
    skipped_file = out_dir / "_skipped.txt"

    if not images_dir.exists():
        raise SystemExit(f"не найдено {images_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(images_dir)
    if not images:
        print(f"Нет картинок в {images_dir}")
        return

    print(f"Dataset:  {args.dataset}/{args.split}")
    print(f"Папка:    {split_dir}")
    print(f"Фильтры:  min_image_px={args.min_image_px}, min_bbox_px={args.min_bbox_px}")

    if args.min_bbox_px > 0 or args.min_image_px > 0:
        added = auto_skip_too_small(images_dir, labels_dir, skipped_file, out_dir,
                                    args.min_image_px, args.min_bbox_px)
        print(f"Авто-фильтр: добавлено {added} в skipped")

    existing = {p.stem for p in out_dir.glob("*.txt") if not p.name.startswith("_")}
    skipped = get_skipped_stems(skipped_file)
    to_annotate = [img for img in images if img.stem not in existing and img.stem not in skipped]

    print(f"Всего:    {len(images)} | размечено: {len(existing)} | "
          f"пропущено: {len(skipped)} | осталось: {len(to_annotate)}\n")
    print("Клики: TL → TR → BR → BL")
    print("u — отменить, s — пропустить, q — выйти\n")

    for image_path in to_annotate:
        print(f"Открываю: {image_path.name}")
        if annotate_image(image_path, out_dir, skipped_file) == "quit":
            print("Выход.")
            break

    cv2.destroyAllWindows()
    print("Готово.")


if __name__ == "__main__":
    main()
