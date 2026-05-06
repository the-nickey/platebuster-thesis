"""Урезанная инференс-обвязка для облачной версии Streamlit-приложения.

Здесь только то, что реально нужно SCC-сборке: одна модель (YOLO11n-pose v2,
single-stage), размытие и наложение логотипа через гомографию по 4 углам,
рисование рамок для режима отладки.

Полный зоопарк (TwoStage, RFDETR, BboxOnly, Classical) живёт в
streamlit_app/inference.py. Дублирование — осознанное: облачная сборка
не должна тянуть из репозитория ничего, кроме своей папки."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Один найденный номер: bbox + 4 угла + уверенность."""
    bbox_xyxy: tuple[int, int, int, int]   # (x1, y1, x2, y2) в пикселях оригинала
    kpts: np.ndarray                        # (4, 2) в пикселях, порядок TL TR BR BL
    confidence: float


class SinglePosePipeline:
    """YOLO11n-pose: bbox и 4 угла из одной сети.

    Модель обучена на смеси CCPD2019 + Roboflow Universe + OpenALPR Benchmark
    с дополнительной ручной разметкой 4 углов."""

    def __init__(self, pose_path: str | Path, device: str = "cpu"):
        self.device = device
        self.pose = YOLO(str(pose_path))

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.3, imgsz: int = 640,
    ) -> list[Detection]:
        results = self.pose.predict(
            source=img_rgb, conf=conf, verbose=False,
            device=self.device, imgsz=imgsz,
        )
        if not results or not len(results[0].boxes):
            return []

        r = results[0]
        boxes = r.boxes
        kpts_data = r.keypoints.xy.cpu().numpy() if r.keypoints is not None else None

        out: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            confidence = float(boxes.conf[i])
            if kpts_data is not None and i < len(kpts_data) and len(kpts_data[i]) >= 4:
                kpts = kpts_data[i][:4].astype(np.float32)
            else:
                # запасной вариант: углы bbox
                kpts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )
            out.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2), kpts=kpts, confidence=confidence,
            ))
        return out


def auto_imgsz(W: int, H: int) -> int:
    """Подбирает разумный размер инференса по размеру оригинала.

    Большое фото — больший размер, иначе мелкие плашки потеряются.
    Маленькое — 640 быстрее без потери качества."""
    longest = max(W, H)
    if longest <= 800:
        return 640
    if longest <= 1500:
        return 1024
    if longest <= 2500:
        return 1280
    return 1600


def blur_detections(
    img_rgb: np.ndarray, dets: Iterable[Detection], kernel: int = 35,
) -> np.ndarray:
    """Размывает четырёхугольник каждой детекции маской и гауссовым размытием."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    kernel = max(3, kernel | 1)  # должен быть нечётным

    for det in dets:
        blurred = cv2.GaussianBlur(out, (kernel, kernel), 0)
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillConvexPoly(mask, det.kpts.astype(np.int32), 255)
        for c in range(3):
            out[:, :, c] = np.where(mask > 0, blurred[:, :, c], out[:, :, c])
    return out


def paste_logo_with_homography(
    img_rgb: np.ndarray,
    dets: Iterable[Detection],
    logo_rgba: np.ndarray,
) -> np.ndarray:
    """Накладывает логотип через перспективное преобразование по 4 углам.

    Порядок углов в det.kpts (TL TR BR BL) должен совпадать с ориентацией
    логотипа: левый верхний угол логотипа ложится в TL номера."""
    out = img_rgb.copy().astype(np.float32)
    H, W = out.shape[:2]

    if logo_rgba.shape[2] == 3:
        alpha = np.full((logo_rgba.shape[0], logo_rgba.shape[1], 1), 255, dtype=np.uint8)
        logo_rgba = np.concatenate([logo_rgba, alpha], axis=2)

    lh, lw = logo_rgba.shape[:2]
    src = np.array(
        [[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]],
        dtype=np.float32,
    )

    for det in dets:
        dst = det.kpts.astype(np.float32)
        H_mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(logo_rgba, H_mat, (W, H), flags=cv2.INTER_LINEAR)

        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        rgb = warped[:, :, :3].astype(np.float32)
        out = out * (1 - alpha) + rgb * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


_FONT_CANDIDATES = (
    # Linux (включая Streamlit Community Cloud)
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "DejaVuSans-Bold.ttf",
    # macOS
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "Arial Bold.ttf",
)


def _load_font(size: int):
    """Жирный sans-serif с поддержкой кириллицы. Перебирает системные пути,
    падает на дефолт Pillow (без кириллицы) только если ни один не нашёлся."""
    from PIL import ImageFont
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def make_default_logo(
    text: str = "ЗДЕСЬ МОГЛА БЫТЬ ВАША РЕКЛАМА",
) -> np.ndarray:
    """Запасная плашка: чёрная надпись на белом фоне с зелёной хромакейной рамкой.

    Размер 460×100 — ровно пропорция русского номера (~4,6 : 1), так что
    после перспективного наложения через гомографию логотип ложится без
    видимых искажений."""
    from PIL import Image as PIL_Image, ImageDraw

    W, H = 460, 100
    border = 6
    chroma_green = (0, 230, 0, 255)

    img = PIL_Image.new("RGBA", (W, H), (255, 255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W - 1, H - 1], outline=chroma_green, width=border)

    # подбираем размер шрифта под фактическую ширину текста
    inner_w = W - 2 * (border + 8)
    size = 30
    font = _load_font(size)
    while size > 8:
        font = _load_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        if (bbox[2] - bbox[0]) <= inner_w:
            break
        size -= 1

    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (W - tw) // 2 - bbox[0]
    y = (H - th) // 2 - bbox[1]
    draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

    return np.array(img)


def draw_detections(
    img_rgb: np.ndarray,
    dets: Iterable[Detection],
    show_bbox: bool = True,
    show_kpts: bool = True,
) -> np.ndarray:
    """Поверх изображения рисует bbox и угловые точки. Только для отладки."""
    out = img_rgb.copy()
    for det in dets:
        if show_bbox:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                out, f"{det.confidence:.2f}", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
            )
        if show_kpts:
            for (x, y) in det.kpts:
                cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
            pts = det.kpts.astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    return out
