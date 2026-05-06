"""Тесты парсера имён CCPD."""

from __future__ import annotations

import pytest

from plates.ccpd import CCPDSample, parse_filename, to_yolo_keypoints


# Реальный пример из README CCPD.
EXAMPLE_FILENAME = (
    "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402"
    "-0_0_22_27_27_33_16-37-15.jpg"
)


def test_parse_filename_returns_ccpd_sample() -> None:
    sample = parse_filename(EXAMPLE_FILENAME)
    assert isinstance(sample, CCPDSample)


def test_parse_filename_area() -> None:
    assert parse_filename(EXAMPLE_FILENAME).area_per_mille == 25


def test_parse_filename_tilt() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    assert s.tilt_h == 95
    assert s.tilt_v == 113


def test_parse_filename_bbox() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    assert s.bbox_xyxy == (154, 383, 386, 473)


def test_parse_filename_corners_raw_order_rb_lb_lt_rt() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    # Согласно README CCPD, порядок именно RB → LB → LT → RT.
    assert s.corners_rb_lb_lt_rt == (
        (386, 473),  # RB
        (177, 454),  # LB
        (154, 383),  # LT
        (363, 402),  # RT
    )


def test_corners_clockwise_starts_from_top_left() -> None:
    """Стандартный порядок для warpPerspective: TL → TR → BR → BL."""
    s = parse_filename(EXAMPLE_FILENAME)
    tl, tr, br, bl = s.corners_clockwise
    assert tl == (154, 383)  # верх-лево
    assert tr == (363, 402)  # верх-право
    assert br == (386, 473)  # низ-право
    assert bl == (177, 454)  # низ-лево


def test_parse_filename_plate_indices() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    assert s.plate_indices == (0, 0, 22, 27, 27, 33, 16)


def test_parse_filename_brightness_blurriness() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    assert s.brightness == 37
    assert s.blurriness == 15


def test_plate_text_decodes() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    # province[0]=皖, alpha[0]=A, ads[22]=N, ads[27]=4, ads[27]=4, ads[33]=R, ads[16]=R
    # Проверяем хотя бы первый символ (китайская провинция) и не падение целиком.
    assert s.plate_text.startswith("皖")
    assert len(s.plate_text) == 7


def test_to_yolo_keypoints_normalisation() -> None:
    s = parse_filename(EXAMPLE_FILENAME)
    line = to_yolo_keypoints(s, img_width=720, img_height=1160)
    tokens = line.split()
    # 1 класс + 4 координаты bbox + 4 точки * 3 значения = 17 токенов
    assert len(tokens) == 1 + 4 + 4 * 3
    assert tokens[0] == "0"
    # все нормализованные значения в диапазоне [0, 1]
    nums = [float(t) for t in tokens[1:]]
    visibilities = [int(t) for t in tokens[7::3]]  # x,y,v -> v на позициях 7,10,13,16
    assert all(0.0 <= n <= 1.0 for n in nums[:6])  # bbox cx,cy,w,h + первые 2 коорд
    assert all(v == 2 for v in visibilities)


def test_parse_invalid_filename_raises() -> None:
    with pytest.raises(ValueError):
        parse_filename("garbage_no_dashes.jpg")
