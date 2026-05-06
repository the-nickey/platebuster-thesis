"""Sanity-check парсера CCPD: рисует bbox + 4 угла поверх случайных фото.

Использование:
    python scripts/sanity_check_ccpd.py [-n 20]

Результат — N PNG-файлов в data/processed/sanity/, открой их в Finder
и глазами убедись, что bbox и углы расположены на номерных знаках.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from plates.ccpd import parse_filename  # noqa: E402

CCPD_ROOT = REPO_ROOT / "data" / "ccpd" / "CCPD2019"
OUT_DIR = REPO_ROOT / "data" / "processed" / "sanity"

CORNER_COLORS = ("red", "green", "blue", "yellow")  # TL TR BR BL
BBOX_COLOR = "magenta"
LINE_WIDTH = 4
POINT_RADIUS = 8


def draw_overlays(img_path: Path, out_path: Path) -> None:
    sample = parse_filename(img_path.name)
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)

        # bbox
        x1, y1, x2, y2 = sample.bbox_xyxy
        draw.rectangle([(x1, y1), (x2, y2)], outline=BBOX_COLOR, width=LINE_WIDTH)

        # 4 угла + полигон по часовой
        corners = sample.corners_clockwise  # TL, TR, BR, BL
        draw.polygon(list(corners), outline="cyan", width=LINE_WIDTH)
        for (x, y), color in zip(corners, CORNER_COLORS, strict=True):
            draw.ellipse(
                [(x - POINT_RADIUS, y - POINT_RADIUS), (x + POINT_RADIUS, y + POINT_RADIUS)],
                fill=color,
                outline="black",
            )

        # подпись с распознанным номером
        try:
            text = sample.plate_text
        except Exception:  # noqa: BLE001
            text = "?"
        draw.text((10, 10), f"plate: {text}", fill="white")

        im.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num", type=int, default=20, help="Сколько фото проверить")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CCPD_ROOT.exists():
        sys.exit(f"❌  Не найдена директория CCPD2019: {CCPD_ROOT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Берём из ccpd_rotate и ccpd_tilt чтобы видеть углы под наклоном —
    # там парсер должен особенно хорошо себя показать.
    pool: list[Path] = []
    for sub in ("ccpd_rotate", "ccpd_tilt", "ccpd_base"):
        d = CCPD_ROOT / sub
        if d.exists():
            pool.extend(list(d.glob("*.jpg"))[:5_000])  # ускоряем выборку

    if not pool:
        sys.exit("❌  Не найдено ни одного .jpg в CCPD2019/")

    rng = random.Random(args.seed)
    sample_files = rng.sample(pool, min(args.num, len(pool)))

    for src in sample_files:
        out = OUT_DIR / f"sanity_{src.stem[:40]}.jpg"
        try:
            draw_overlays(src, out)
            print(f"✅  {out.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"❌  {src.name}: {exc}")

    print(f"\n📁  Открой папку: open {OUT_DIR}")


if __name__ == "__main__":
    main()
