"""Парсер аннотаций OpenALPR Benchmark.

Формат разметки в OpenALPR/benchmarks/endtoend/:

    {filename}\\t{x}\\t{y}\\t{w}\\t{h}\\t{plate_text}

Один файл `*.txt` на одно изображение, по одной строке. (x, y) — top-left угол bbox,
(w, h) — ширина и высота. Координаты в пикселях, текст номера — в латинице.

Источник: https://github.com/openalpr/benchmarks
Лицензия: AGPL-3.0 (совместима с YOLO26 AGPL-3.0).
Регионы:
    - br (Бразилия): 229 фото
    - eu (Европа):   216 фото
    - us (США):      444 фото
    - usimages:       22 фото (US доп.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OpenALPRSample:
    """Распарсенная аннотация одной фотографии OpenALPR."""

    filename: str                       # например "eu1.jpg" или "AYO9034.jpg"
    bbox_xywh: tuple[int, int, int, int]   # (x, y, w, h) в пикселях, x/y = top-left
    plate_text: str

    @property
    def bbox_xyxy(self) -> tuple[int, int, int, int]:
        x, y, w, h = self.bbox_xywh
        return (x, y, x + w, y + h)


def parse_annotation(txt_path: Path) -> OpenALPRSample:
    """Распарсить .txt-файл OpenALPR в OpenALPRSample.

    Raises:
        ValueError: если формат не соответствует ожидаемому.
        FileNotFoundError: если txt_path не существует.
    """
    raw = txt_path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"Пустой файл аннотации: {txt_path}")

    # Берём только первую строку — в OpenALPR endtoend всегда 1 строка/файл.
    line = raw.splitlines()[0]
    parts = line.split("\t")
    if len(parts) < 6:
        raise ValueError(
            f"Ожидалось 6 полей через TAB, получено {len(parts)}: {line!r}"
        )

    filename = parts[0]
    try:
        x, y, w, h = (int(parts[i]) for i in range(1, 5))
    except ValueError as exc:
        raise ValueError(f"Некорректные координаты bbox в {txt_path}: {exc}") from exc

    plate_text = parts[5]

    return OpenALPRSample(
        filename=filename,
        bbox_xywh=(x, y, w, h),
        plate_text=plate_text,
    )


def to_yolo_bbox(
    sample: OpenALPRSample,
    img_width: int,
    img_height: int,
) -> str:
    """Конвертировать в строку YOLO bbox формата (без keypoints).

    YOLO bbox формат:
        class_id  cx  cy  w  h

    Где cx,cy,w,h — нормализованные [0, 1].

    Углы у OpenALPR не размечены — добавлять keypoints в этот формат нечего.
    Доразметка 4 углов делается отдельно через CVAT (см. infra/cvat/README.md)
    или собственные annotator-ы из scripts/annotation/.
    """
    x, y, w, h = sample.bbox_xywh
    cx = (x + w / 2) / img_width
    cy = (y + h / 2) / img_height
    nw = w / img_width
    nh = h / img_height
    return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
