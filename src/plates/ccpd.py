"""Парсер аннотаций CCPD (Chinese City Parking Dataset).

Аннотации в CCPD зашиты прямо в имя файла. Формат:

    area-tilt-bbox-four_vertices-plate_number-brightness-blurriness.jpg

Пример:
    025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg

Поля:
    1. area: отношение площади номера к площади кадра, ‰ (промилле).
    2. tilt: горизонтальный_вертикальный наклон в градусах.
    3. bbox: TL_BR в координатах (TL=top-left, BR=bottom-right) пикселей.
    4. four_vertices: 4 угла номера в порядке RB_LB_LT_RT (см. README CCPD).
    5. plate_number: индексы 7 символов (province, alpha, 5×alphanumeric).
    6. brightness, blurriness: служебные характеристики.

Источник формата: https://github.com/detectRecog/CCPD (Xu et al., ECCV 2018).
Лицензия датасета: MIT.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Алфавиты символов китайских номеров — для опциональной декодировки plate_number.
PROVINCES = [
    "皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙",
    "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵",
    "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O",
]
ALPHABETS = list("ABCDEFGHJKLMNPQRSTUVWXYZO")
ADS = list("ABCDEFGHJKLMNPQRSTUVWXYZ0123456789O")


@dataclass(frozen=True, slots=True)
class CCPDSample:
    """Распарсенная аннотация одной фотографии CCPD."""

    filename: str
    area_per_mille: int
    tilt_h: int
    tilt_v: int
    bbox_xyxy: tuple[int, int, int, int]      # (x1, y1, x2, y2)
    corners_rb_lb_lt_rt: tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ]
    plate_indices: tuple[int, ...]
    brightness: int
    blurriness: int

    # ----- удобные представления -----

    @property
    def corners_clockwise(self) -> tuple[
        tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]
    ]:
        """Углы в порядке TL → TR → BR → BL (по часовой, стандарт для warpPerspective).

        В CCPD углы хранятся как RB_LB_LT_RT, что эквивалентно BR_BL_TL_TR.
        Перестановка → TL_TR_BR_BL.
        """
        rb, lb, lt, rt = self.corners_rb_lb_lt_rt
        return (lt, rt, rb, lb)

    @property
    def plate_text(self) -> str:
        """Декодированный номер: провинция + 6 символов."""
        if len(self.plate_indices) != 7:
            return "?"
        idx = self.plate_indices
        return PROVINCES[idx[0]] + ALPHABETS[idx[1]] + "".join(ADS[i] for i in idx[2:])


def parse_filename(filename: str) -> CCPDSample:
    """Распарсить имя файла CCPD в структурированный CCPDSample.

    Args:
        filename: имя файла, с расширением или без, путь допустим (берётся basename).

    Returns:
        CCPDSample с распарсенными полями.

    Raises:
        ValueError: если формат имени файла не соответствует CCPD.
    """
    stem = Path(filename).stem  # отрезаем .jpg
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"Ожидалось 7 полей через '-', получено {len(parts)}: {filename!r}"
        )

    area_str, tilt_str, bbox_str, corners_str, plate_str, bright_str, blur_str = parts

    try:
        area = int(area_str)

        tilt_h, tilt_v = (int(x) for x in tilt_str.split("_"))

        # bbox: "x1&y1_x2&y2"
        tl_str, br_str = bbox_str.split("_")
        x1, y1 = (int(v) for v in tl_str.split("&"))
        x2, y2 = (int(v) for v in br_str.split("&"))

        # corners: "x&y_x&y_x&y_x&y" — 4 точки в порядке RB_LB_LT_RT
        corner_chunks = corners_str.split("_")
        if len(corner_chunks) != 4:
            raise ValueError(f"Ожидалось 4 угла, получено {len(corner_chunks)}")
        corners = tuple(
            tuple(int(v) for v in c.split("&"))  # type: ignore[misc]
            for c in corner_chunks
        )

        plate_indices = tuple(int(x) for x in plate_str.split("_"))
        brightness = int(bright_str)
        blurriness = int(blur_str)
    except (ValueError, IndexError) as exc:
        raise ValueError(f"Не удалось распарсить {filename!r}: {exc}") from exc

    return CCPDSample(
        filename=Path(filename).name,
        area_per_mille=area,
        tilt_h=tilt_h,
        tilt_v=tilt_v,
        bbox_xyxy=(x1, y1, x2, y2),
        corners_rb_lb_lt_rt=corners,  # type: ignore[arg-type]
        plate_indices=plate_indices,
        brightness=brightness,
        blurriness=blurriness,
    )


def iter_ccpd(root: Path, glob: str = "**/*.jpg") -> Iterable[CCPDSample]:
    """Итеративно пройти по всем .jpg в корне CCPD и вернуть распарсенные образцы.

    Файлы с битым именем — пропускаются с предупреждением (а не падением),
    т.к. при распаковке CCPD изредка попадаются служебные изображения.
    """
    import warnings

    for p in root.glob(glob):
        try:
            yield parse_filename(p.name)
        except ValueError as exc:
            warnings.warn(f"Skipping {p.name}: {exc}", stacklevel=2)


def to_yolo_keypoints(
    sample: CCPDSample,
    img_width: int,
    img_height: int,
) -> str:
    """Конвертировать CCPDSample в строку YOLO-keypoints формата.

    YOLO-keypoints формат:
        class_id  cx  cy  w  h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4

    где cx,cy,w,h — нормализованный bbox, (xi,yi) — нормализованные углы,
    vi — visibility (2 = видимый, всегда в датасете CCPD).

    Углы возвращаем в стандартном порядке TL → TR → BR → BL (по часовой).
    """
    x1, y1, x2, y2 = sample.bbox_xyxy
    cx = (x1 + x2) / 2 / img_width
    cy = (y1 + y2) / 2 / img_height
    w = (x2 - x1) / img_width
    h = (y2 - y1) / img_height

    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for px, py in sample.corners_clockwise:
        parts.append(f"{px / img_width:.6f} {py / img_height:.6f} 2")

    return " ".join(parts)
