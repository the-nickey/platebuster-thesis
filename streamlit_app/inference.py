"""Pipelines для скрытия / брендирования номеров. Все возвращают `list[Detection]`.

Зоопарк (по убыванию рекомендации для production):

  TwoStagePipeline         — YOLO bbox-детектор + ResNet18 keypoint head
                             (для YOLO11n-detect / YOLO12n-detect)
  SinglePosePipeline       — YOLO11n-pose (bbox + 4 угла из одной сети)
  BboxOnlyPipeline         — bbox-детектор, углы = вершины bbox'а
                             (без перспективы, как у Avito/Drom production)
  ClassicalPipeline        — контурный OpenCV-baseline без обучения

Архитектура keypoint head — ровно та, что в scripts/training/train_keypoint_head.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

from ultralytics import YOLO


# ResNet18 ImageNet normalization (та же, что в train_keypoint_head.py)
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class KeypointHead(nn.Module):
    """ResNet18 backbone + FC head → 4 keypoint × 2. Архитектура ≡ train_keypoint_head.py."""

    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        out = self.head(f)
        return out.view(-1, 4, 2)


@dataclass
class Detection:
    """Один обнаруженный номер: bbox + 4 угла + уверенность."""
    bbox_xyxy: tuple[int, int, int, int]   # (x1, y1, x2, y2) в пикселях оригинала
    kpts: np.ndarray                        # (4, 2) в пикселях оригинала, порядок TL TR BR BL
    confidence: float


def crop_with_padding(
    img_rgb: np.ndarray, bbox_xyxy: tuple[int, int, int, int],
    padding: float = 0.25, target_size: int = 192,
) -> tuple[np.ndarray | None, np.ndarray, float]:
    """Воспроизводит логику `build_keypoint_crops.py:crop_with_corners`.

    Квадратный crop вокруг центра bbox размером `max(bw, bh) * (1 + 2*padding)`.
    Если crop выходит за края — реальная часть кладётся на серый canvas (114)
    с центральным offset'ом. Возвращает (crop_resized, canvas_origin, canvas_side)."""
    H, W = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    bw, bh = x2 - x1, y2 - y1

    side = max(bw, bh) * (1 + 2 * padding)
    canvas_x1 = cx - side / 2
    canvas_y1 = cy - side / 2

    rx1 = int(round(max(canvas_x1, 0)))
    ry1 = int(round(max(canvas_y1, 0)))
    rx2 = int(round(min(canvas_x1 + side, W)))
    ry2 = int(round(min(canvas_y1 + side, H)))
    if rx2 <= rx1 or ry2 <= ry1:
        return None, np.array([canvas_x1, canvas_y1]), side

    crop = img_rgb[ry1:ry2, rx1:rx2]
    cw, ch = crop.shape[1], crop.shape[0]
    max_side = max(cw, ch)
    if cw != max_side or ch != max_side:
        canvas = np.full((max_side, max_side, 3), 114, dtype=np.uint8)
        offset_x = (max_side - cw) // 2
        offset_y = (max_side - ch) // 2
        canvas[offset_y:offset_y + ch, offset_x:offset_x + cw] = crop
        crop = canvas

    crop_rs = cv2.resize(crop, (target_size, target_size))
    return crop_rs, np.array([canvas_x1, canvas_y1]), side


def preprocess_kpt_input(crop_rgb: np.ndarray) -> torch.Tensor:
    """ImageNet normalization для ResNet18 keypoint head."""
    t = torch.from_numpy(crop_rgb).float().permute(2, 0, 1) / 255.0
    return _NORMALIZE(t).unsqueeze(0)


def _inflate_outward(corners: np.ndarray,
                     bbox_xyxy: tuple[int, int, int, int],
                     amount: float) -> np.ndarray:
    """Сдвигает каждый угол наружу от центра bbox на `amount × diag(bbox)`.

    Контролируемый over-cover: ResNet18 систематически даёт quadrilateral
    меньше GT bbox (плашка имеет физическую рамку 2-3 мм, не вошедшую в
    Roboflow-разметку bbox). Inflate компенсирует эту систематику и снижает
    риск under-cover при наложении логотипа через гомографию.

    Параметр подобран grid search'ем — см.
    `scripts/training/grid_search_keypoint_postproc.py` и главу 2 §«НАБЛЮДЕНИЕ».
    """
    if amount <= 0:
        return corners
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diag = float(np.hypot(x2 - x1, y2 - y1))
    shift = amount * diag
    out = corners.astype(np.float32).copy()
    for i in range(out.shape[0]):
        dx, dy = out[i, 0] - cx, out[i, 1] - cy
        n = float(np.hypot(dx, dy)) + 1e-6
        out[i, 0] += dx / n * shift
        out[i, 1] += dy / n * shift
    return out


# Порядок углов в YOLO-pose разметке: TL → TR → BR → BL (clockwise начиная с TL).
# После горизонтального flip'а порядок становится TR → TL → BL → BR, что соответ-
# ствует индексам [1, 0, 3, 2] относительно исходного порядка.
KPT_FLIP_IDX = (1, 0, 3, 2)


def _predict_keypoints_with_tta(
    kpt_head: "KeypointHead",
    crop_rgb: np.ndarray,
    device: str,
    use_tta: bool,
) -> np.ndarray:
    """Предсказывает 4 угла в нормализованных координатах crop'а.

    Если use_tta=True — дополнительно прогоняет горизонтально-зеркальный crop
    через ту же сеть, отражает x-координаты обратно, переупорядочивает углы
    через KPT_FLIP_IDX и усредняет с оригинальным предсказанием. Снижает шум
    регрессии за счёт двух независимых проходов через одну сеть. Цена:
    2× inference time (на CPU ~12 ms × 2 = 24 ms на крupно crop).

    Эмпирический выигрыш на 240 russian/test (см. grid_search_keypoint_postproc.py):
    coverage +1.3 п.п. в среднем, asym_loss -0.002. Маленький, но стабильный."""
    t = preprocess_kpt_input(crop_rgb).to(device)
    with torch.no_grad():
        kpts = kpt_head(t)[0].cpu().numpy().reshape(4, 2)

    if not use_tta:
        return kpts

    crop_flip = np.ascontiguousarray(crop_rgb[:, ::-1, :])
    t_flip = preprocess_kpt_input(crop_flip).to(device)
    with torch.no_grad():
        kpts_flip = kpt_head(t_flip)[0].cpu().numpy().reshape(4, 2)
    kpts_flip[:, 0] = 1.0 - kpts_flip[:, 0]
    kpts_flip = kpts_flip[list(KPT_FLIP_IDX)]
    return ((kpts + kpts_flip) / 2.0).astype(np.float32)


class TwoStagePipeline:
    """YOLO bbox-детектор (Stage 1) + ResNet18 keypoint head (Stage 2).

    Параметризуется любым YOLO-detect (yolo11n, yolo12n, ...). Голова
    общая — обучена на 60K crop'ов 192×192 в `train_keypoint_head.py`."""

    def __init__(
        self,
        detector_path: str | Path,
        kpt_head_path: str | Path,
        device: str = "cpu",
        crop_size: int = 192,
        # Дефолты подобраны grid search'ем
        # scripts/training/grid_search_keypoint_postproc.py на 240 russian/test.
        # 2026-05-04: первая итерация — ассиметричная метрика (under × 10), дала
        #   inflate=0.12, area_ratio = 1.22. Визуально лого слишком сильно вылезал.
        # 2026-05-05: пересмотрели на симметричную (under = over), новый оптимум:
        #   inflate=0.05 + tta=False, area_ratio = 1.005, cov_mean = 0.890.
        #   Лого по площади ровно как плашка, минимальный over-cover на корпусе
        #   рамки (~3 px). Цифры/буквы внутри plate inner закрываются полностью.
        crop_padding: float = 0.30,
        inflate_outward: float = 0.05,
        use_tta: bool = False,
    ):
        self.device = device
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.inflate_outward = inflate_outward
        self.use_tta = use_tta

        self.detector = YOLO(str(detector_path))
        self.kpt_head = KeypointHead().to(device)
        ckpt = torch.load(str(kpt_head_path), map_location=device, weights_only=False)
        sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        self.kpt_head.load_state_dict(sd)
        self.kpt_head.eval()

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.5, imgsz: int = 640,
    ) -> list[Detection]:
        H, W = img_rgb.shape[:2]
        results = self.detector.predict(
            source=img_rgb, conf=conf, verbose=False, device=self.device, imgsz=imgsz,
        )
        if not results or not len(results[0].boxes):
            return []

        boxes = results[0].boxes
        detections: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            confidence = float(boxes.conf[i])

            crop_canvas, canvas_origin, canvas_side = self._crop_with_padding(
                img_rgb, (x1, y1, x2, y2)
            )
            if crop_canvas is None:
                continue

            kpts_norm = _predict_keypoints_with_tta(
                self.kpt_head, crop_canvas, self.device, self.use_tta,
            )

            kpts_orig = canvas_origin + kpts_norm * canvas_side
            kpts_orig = _inflate_outward(kpts_orig, (x1, y1, x2, y2),
                                         self.inflate_outward)
            detections.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2), kpts=kpts_orig, confidence=confidence,
            ))
        return detections

    def _crop_with_padding(self, img_rgb, bbox_xyxy):
        return crop_with_padding(
            img_rgb, bbox_xyxy, self.crop_padding, self.crop_size,
        )

    def _preprocess(self, crop_rgb):
        return preprocess_kpt_input(crop_rgb)


class SinglePosePipeline:
    """YOLO11n-pose: bbox + 4 угла из одной сети (single-stage).

    Углы берутся прямо из pose-выхода Ultralytics'а, без 2-stage crop'а.
    Точнее на ccpd / russian (где обучена pose-голова), сильно проседает
    на european / openalpr (см. главу 2 §«Архитектурный обзор»)."""

    def __init__(self, pose_path: str | Path, device: str = "cpu"):
        self.device = device
        self.pose = YOLO(str(pose_path))

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.5, imgsz: int = 640,
    ) -> list[Detection]:
        results = self.pose.predict(
            source=img_rgb, conf=conf, verbose=False, device=self.device, imgsz=imgsz,
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
                # fallback: углы bbox'а
                kpts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
                )
            out.append(Detection(bbox_xyxy=(x1, y1, x2, y2), kpts=kpts, confidence=confidence))
        return out


class BboxOnlyPipeline:
    """Любой YOLO-detect, углы = вершины bbox'а (как у Avito/Drom production).

    Без 2-stage и без перспективы. Размытие/наложение делается по
    прямоугольнику. Самый быстрый pipeline — отсутствует Stage 2."""

    def __init__(self, detector_path: str | Path, device: str = "cpu"):
        self.device = device
        self.detector = YOLO(str(detector_path))

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.5, imgsz: int = 640,
    ) -> list[Detection]:
        results = self.detector.predict(
            source=img_rgb, conf=conf, verbose=False, device=self.device, imgsz=imgsz,
        )
        if not results or not len(results[0].boxes):
            return []
        boxes = results[0].boxes
        out: list[Detection] = []
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            kpts = np.array(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32,
            )
            out.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2), kpts=kpts, confidence=float(boxes.conf[i]),
            ))
        return out


class ClassicalPipeline:
    """Контурный OpenCV-baseline. Без обучения.

    Точная копия `scripts/training/eval_classical.py:detect_plates_classical()`,
    которой получены метрики F1 в таблице §5.3:
        bilateralFilter → Canny с авто-порогом по медиане →
        morph closing 13×5 (горизонтальный kernel под плашку) →
        top-50 контуров по площади → approxPolyDP до 4 углов →
        фильтры aspect/area/min-size."""

    # гиперпараметры — те же что в eval_classical.py
    ASPECT_MIN = 2.0
    ASPECT_MAX = 6.5
    MIN_AREA_FRAC = 1e-4   # >= 0.01% площади кадра
    MAX_AREA_FRAC = 0.5    # <= 50%
    APPROX_EPS_FRAC = 0.02
    TOP_CONTOURS = 50

    def __init__(self, **kwargs):
        pass

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.5, imgsz: int = 640,
    ) -> list[Detection]:
        # `conf` / `imgsz` игнорируем — rule-based не имеет confidence
        # и работает на оригинальном разрешении.
        del conf, imgsz

        H, W = img_rgb.shape[:2]
        img_area = H * W

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 75, 75)

        # Canny с автоматическими порогами по медиане яркости
        v = np.median(gray)
        lo = int(max(0, 0.66 * v))
        hi = int(min(255, 1.33 * v))
        edges = cv2.Canny(gray, lo, hi)

        # горизонтальный kernel — соединяет edges от букв номера в один блок
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:self.TOP_CONTOURS]

        out: list[Detection] = []
        for rank, cnt in enumerate(contours):
            peri = cv2.arcLength(cnt, True)
            if peri < 20:
                continue
            approx = cv2.approxPolyDP(cnt, self.APPROX_EPS_FRAC * peri, True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            if w < 8 or h < 4:
                continue

            ar = w / max(h, 1)
            if not (self.ASPECT_MIN <= ar <= self.ASPECT_MAX):
                continue

            area_frac = (w * h) / img_area
            if not (self.MIN_AREA_FRAC <= area_frac <= self.MAX_AREA_FRAC):
                continue

            corners = approx.reshape(4, 2).astype(np.float32)
            corners = _sort_corners_tlrb(corners)
            out.append(Detection(
                bbox_xyxy=(x, y, x + w, y + h),
                kpts=corners,
                confidence=1.0 - rank / self.TOP_CONTOURS,  # больше площадь → выше conf
            ))
        return out


class RFDETRPipeline:
    """RF-DETR транс­формер-детектор (Nano или Medium).

    На license-plate-задаче метрики Nano и Medium совпадают (0.986 vs 0.984
    mean mAP@50) — backbone DINOv2-small везде одинаковый, разница только в
    decoder-слоях (2 vs 4) и input resolution (384 vs 576). Nano быстрее в
    ~3× на CPU, поэтому в production выбираем Nano.

    `size`:
      - "nano"   — RFDETRNano   (default), 384 px, ~280 ms/img CPU
      - "medium" — RFDETRMedium (legacy),  576 px, ~830 ms/img CPU

    `with_kpt_head=True` — добавляет ResNet18 keypoint head как Stage 2;
    углы (а не вершины bbox) → честная гомография для перспективы."""

    def __init__(
        self,
        ckpt_path: str | Path,
        kpt_head_path: str | Path | None = None,
        device: str = "cpu",
        crop_size: int = 192,
        # см. комментарий в TwoStagePipeline.__init__ — те же значения.
        crop_padding: float = 0.30,
        inflate_outward: float = 0.05,
        use_tta: bool = False,
        size: str = "nano",
    ):
        # ленивый import — пакет тяжёлый, ставится отдельно
        from rfdetr import RFDETRMedium, RFDETRNano
        self.device = device
        self.crop_size = crop_size
        self.crop_padding = crop_padding
        self.inflate_outward = inflate_outward
        self.use_tta = use_tta
        if size.lower() == "medium":
            self.detector = RFDETRMedium(pretrain_weights=str(ckpt_path))
        elif size.lower() == "nano":
            self.detector = RFDETRNano(pretrain_weights=str(ckpt_path))
        else:
            raise ValueError(f"unknown rfdetr size: {size!r}; expected 'nano' or 'medium'")

        self.kpt_head: KeypointHead | None = None
        if kpt_head_path is not None:
            self.kpt_head = KeypointHead().to(device)
            ckpt = torch.load(str(kpt_head_path), map_location=device, weights_only=False)
            sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
            self.kpt_head.load_state_dict(sd)
            self.kpt_head.eval()

    def __call__(
        self, img_rgb: np.ndarray, conf: float = 0.5, imgsz: int = 640,
    ) -> list[Detection]:
        # rfdetr.predict понимает PIL.Image и numpy; даём numpy
        from PIL import Image
        # imgsz у RF-DETR контролируется внутри обвязки (NAS-выбранный размер),
        # параметр оставлен для совместимости интерфейса.
        del imgsz
        pil = Image.fromarray(img_rgb)
        dets = self.detector.predict(pil, threshold=conf)

        out: list[Detection] = []
        if dets is None or len(dets) == 0:
            return out

        for i in range(len(dets.xyxy)):
            x1, y1, x2, y2 = dets.xyxy[i].astype(int).tolist()
            confidence = (
                float(dets.confidence[i]) if dets.confidence is not None else 1.0
            )

            if self.kpt_head is not None:
                crop, origin, side = crop_with_padding(
                    img_rgb, (x1, y1, x2, y2),
                    self.crop_padding, self.crop_size,
                )
                if crop is not None:
                    kpts_norm = _predict_keypoints_with_tta(
                        self.kpt_head, crop, self.device, self.use_tta,
                    )
                    kpts = origin + kpts_norm * side
                    kpts = _inflate_outward(kpts, (x1, y1, x2, y2),
                                            self.inflate_outward)
                else:
                    kpts = np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.float32,
                    )
            else:
                kpts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32,
                )

            out.append(Detection(
                bbox_xyxy=(x1, y1, x2, y2), kpts=kpts, confidence=confidence,
            ))
        return out


def _sort_corners_tlrb(pts: np.ndarray) -> np.ndarray:
    """Сортирует 4 точки в порядке TL → TR → BR → BL.

    Идея: центр масс, затем для каждой точки берём знаки (dx, dy)
    относительно центра. TL = (-,-), TR = (+,-), BR = (+,+), BL = (-,+)."""
    center = pts.mean(axis=0)
    out = np.zeros_like(pts)
    for p in pts:
        dx, dy = p - center
        if dx <= 0 and dy <= 0:
            out[0] = p
        elif dx > 0 and dy <= 0:
            out[1] = p
        elif dx > 0 and dy > 0:
            out[2] = p
        else:
            out[3] = p
    return out


# ----- утилиты обработки -----

def blur_detections(
    img_rgb: np.ndarray, dets: Iterable[Detection], kernel: int = 35
) -> np.ndarray:
    """Размывает quadrilateral каждой детекции через mask + Gaussian blur."""
    out = img_rgb.copy()
    H, W = out.shape[:2]
    kernel = max(3, kernel | 1)  # должен быть нечётным

    for det in dets:
        # размытие всей картинки + копирование внутри quadrilateral
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
    """Накладывает логотип через perspective warp (cv2.findHomography).

    Порядок углов в det.kpts: TL TR BR BL — должен совпадать с тем как
    логотип ориентирован (т.е. левый-верхний угол логотипа → TL номера)."""
    out = img_rgb.copy().astype(np.float32)
    H, W = out.shape[:2]

    if logo_rgba.shape[2] == 3:
        # добавим непрозрачный alpha
        alpha = np.full((logo_rgba.shape[0], logo_rgba.shape[1], 1), 255, dtype=np.uint8)
        logo_rgba = np.concatenate([logo_rgba, alpha], axis=2)

    lh, lw = logo_rgba.shape[:2]
    src = np.array([[0, 0], [lw - 1, 0], [lw - 1, lh - 1], [0, lh - 1]], dtype=np.float32)

    for det in dets:
        dst = det.kpts.astype(np.float32)
        H_mat = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(logo_rgba, H_mat, (W, H), flags=cv2.INTER_LINEAR)

        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        rgb = warped[:, :, :3].astype(np.float32)
        out = out * (1 - alpha) + rgb * alpha

    return np.clip(out, 0, 255).astype(np.uint8)


def make_default_logo(text: str = "СКРЫТО", scale: float = 1.5) -> np.ndarray:
    """Простой запасной логотип, если пользователь не загрузил свой."""
    logo = np.zeros((100, 400, 4), dtype=np.uint8)
    logo[:, :, :3] = (40, 80, 200)        # синий фон
    logo[:, :, 3] = 230                    # 90% непрозрачно
    cv2.putText(
        logo, text, (40, 70), cv2.FONT_HERSHEY_DUPLEX, scale,
        (255, 255, 255), 2, cv2.LINE_AA,
    )
    return logo


def draw_detections(
    img_rgb: np.ndarray,
    dets: Iterable[Detection],
    show_bbox: bool = True,
    show_kpts: bool = True,
) -> np.ndarray:
    """Отрисовка bbox + угловых точек поверх изображения (для debug-режима)."""
    out = img_rgb.copy()
    for det in dets:
        if show_bbox:
            x1, y1, x2, y2 = det.bbox_xyxy
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)
            cv2.putText(
                out, f"{det.confidence:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1,
            )
        if show_kpts:
            for (x, y) in det.kpts:
                cv2.circle(out, (int(x), int(y)), 5, (0, 255, 0), -1)
            pts = det.kpts.astype(np.int32)
            cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    return out
