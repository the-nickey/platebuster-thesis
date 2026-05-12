"""
Single-image inference wrappers — те же модели, что в scripts/dump_predictions.py,
но без batching. Используются для замера latency в Streamlit Cloud (1 vCPU).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def make_inferer(cfg: dict, repo_root: Path, device: str = "cpu"):
    """Возвращает callable(img_path) -> None для одного inference-вызова."""
    t = cfg["type"]

    if t == "yolo_detect":
        from ultralytics import YOLO
        model = YOLO(str(repo_root / cfg["weights"]))
        imgsz = cfg.get("imgsz", 640)
        def run(img_path):
            _ = model.predict(source=str(img_path), imgsz=imgsz,
                              conf=0.001, device=device, verbose=False)
        return run

    if t == "yolo_pose":
        from ultralytics import YOLO
        model = YOLO(str(repo_root / cfg["weights"]))
        imgsz = cfg.get("imgsz", 640)
        def run(img_path):
            _ = model.predict(source=str(img_path), imgsz=imgsz,
                              conf=0.001, device=device, verbose=False)
        return run

    if t == "rfdetr":
        import rfdetr as rfd
        cls = getattr(rfd, cfg["rfdetr_class"])
        model = cls(pretrain_weights=str(repo_root / cfg["weights"]),
                    device=device, resolution=cfg["resolution"])
        def run(img_path):
            _ = model.predict(str(img_path), threshold=0.001)
        return run

    if t == "two_stage":
        from ultralytics import YOLO
        import torch
        from torchvision.models import resnet18
        import torch.nn as nn

        bbox_model = YOLO(str(repo_root / cfg["bbox_weights"]))

        class KeypointHead(nn.Module):
            def __init__(self):
                super().__init__()
                backbone = resnet18(weights=None)
                backbone.fc = nn.Identity()
                self.backbone = backbone
                self.head = nn.Sequential(
                    nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(0.2),
                    nn.Linear(256, 8), nn.Sigmoid(),
                )

            def forward(self, x):
                return self.head(self.backbone(x)).view(-1, 4, 2)

        kpt = KeypointHead()
        state = torch.load(str(repo_root / cfg["weights"]), map_location="cpu")
        kpt.load_state_dict(state.get("state_dict", state))
        kpt.eval()
        dev = torch.device(device if device != "auto" else "cpu")
        kpt.to(dev)
        crop_size = cfg.get("crop_size", 192)
        pad = cfg.get("crop_padding", 0.25)

        def run(img_path):
            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]
            res = bbox_model.predict(source=str(img_path), conf=0.25,
                                     device=device, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                return
            xyxy = res.boxes.xyxy.cpu().numpy()
            for x1, y1, x2, y2 in xyxy:
                bw = x2 - x1; bh = y2 - y1
                p_ = pad * max(bw, bh)
                cx1 = max(0, int(x1 - p_)); cy1 = max(0, int(y1 - p_))
                cx2 = min(W, int(x2 + p_)); cy2 = min(H, int(y2 + p_))
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                cr = cv2.resize(crop, (crop_size, crop_size))
                t_ = torch.from_numpy(cr).float().permute(2, 0, 1) / 255.0
                t_ = t_.unsqueeze(0).to(dev)
                with torch.no_grad():
                    _ = kpt(t_)

        return run

    if t == "classical":
        def run(img_path):
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.bilateralFilter(gray, 11, 17, 17)
            edges = cv2.Canny(gray, 30, 200)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return run

    raise ValueError(f"unknown model type: {t}")
