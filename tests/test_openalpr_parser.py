"""Тесты парсера OpenALPR Benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest

from plates.openalpr import OpenALPRSample, parse_annotation, to_yolo_bbox


@pytest.fixture
def eu_annotation(tmp_path: Path) -> Path:
    p = tmp_path / "eu1.txt"
    p.write_text("eu1.jpg\t396\t340\t203\t46\tM5XSX\n", encoding="utf-8")
    return p


@pytest.fixture
def br_annotation(tmp_path: Path) -> Path:
    p = tmp_path / "AYO9034.txt"
    p.write_text("AYO9034.jpg\t528\t412\t162\t52\tAYO9034", encoding="utf-8")
    return p


def test_parse_eu_annotation(eu_annotation: Path) -> None:
    sample = parse_annotation(eu_annotation)
    assert isinstance(sample, OpenALPRSample)
    assert sample.filename == "eu1.jpg"
    assert sample.bbox_xywh == (396, 340, 203, 46)
    assert sample.plate_text == "M5XSX"


def test_parse_br_annotation_no_trailing_newline(br_annotation: Path) -> None:
    sample = parse_annotation(br_annotation)
    assert sample.filename == "AYO9034.jpg"
    assert sample.plate_text == "AYO9034"


def test_bbox_xyxy_converts_correctly(eu_annotation: Path) -> None:
    sample = parse_annotation(eu_annotation)
    assert sample.bbox_xyxy == (396, 340, 396 + 203, 340 + 46)


def test_to_yolo_bbox_normalisation(eu_annotation: Path) -> None:
    sample = parse_annotation(eu_annotation)
    line = to_yolo_bbox(sample, img_width=720, img_height=480)
    tokens = line.split()
    assert len(tokens) == 5
    assert tokens[0] == "0"
    nums = [float(t) for t in tokens[1:]]
    assert all(0 <= n <= 1 for n in nums), f"out of range: {nums}"
    # центр: cx = (396 + 203/2) / 720 ≈ 0.691
    assert abs(nums[0] - (396 + 203 / 2) / 720) < 1e-5
    # высота: nh = 46/480 ≈ 0.0958
    assert abs(nums[3] - 46 / 480) < 1e-5


def test_parse_empty_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Пустой"):
        parse_annotation(p)


def test_parse_malformed_line_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.txt"
    p.write_text("only_two\tfields\n", encoding="utf-8")
    with pytest.raises(ValueError, match="6 полей"):
        parse_annotation(p)


def test_parse_non_integer_coords_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad_int.txt"
    p.write_text("img.jpg\tabc\t340\t203\t46\tM5XSX\n", encoding="utf-8")
    with pytest.raises(ValueError, match="координаты"):
        parse_annotation(p)
