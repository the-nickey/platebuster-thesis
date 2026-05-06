"""
Обучает keypoint regression head: ResNet18 + FC → 8 координат (4 угла × 2).

Это вторая голова двухстадийного pipeline (Avito 2018-style):
  bbox detector → crop → этот head → 4 угла → гомография → скрытие/брендирование

Вход обучения: data/processed/keypoint_crops/{train,val}/{images,labels}/
  где label.txt содержит 8 нормализованных чисел: x1 y1 x2 y2 x3 y3 x4 y4
  (порядок: TL, TR, BR, BL).

Eval: per-region на data/processed/keypoint_crops/test_per_region/<region>/

Запуск:
    python scripts/training/train_keypoint_head.py
    python scripts/training/train_keypoint_head.py --epochs 30 --batch 64 --lr 1e-3
    python scripts/training/train_keypoint_head.py --eval-only \\
        --weights runs/keypoint_head_<ts>/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
import cv2

from common import pick_device, RUNS_ROOT


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CROPS_ROOT = REPO_ROOT / "data" / "processed" / "keypoint_crops"

REGIONS = ["ccpd", "russian", "european", "openalpr", "generic"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--patience", type=int, default=5,
                   help="early stop: эпох без улучшения val")
    p.add_argument("--device", default=None)
    p.add_argument("--img-size", type=int, default=192,
                   help="должно совпадать с --crop-size в build_keypoint_crops.py")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--weights", default=None,
                   help="путь к .pt — для resume или eval-only")
    return p.parse_args()


class CropKeypointsDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path,
                 img_size: int, augment: bool):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.img_size = img_size
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(images_dir.glob("*.jpg")):
            lbl = labels_dir / f"{img_path.stem}.txt"
            if lbl.exists():
                self.samples.append((img_path, lbl))

        # ImageNet normalization (для pretrained ResNet18)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        # цветовые штуки — безопасно для координат
        if np.random.rand() < 0.5:
            img = (img.astype(np.int16) + np.random.randint(-25, 25)).clip(0, 255).astype(np.uint8)
        if np.random.rand() < 0.3:
            # лёгкий gaussian noise
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = (img.astype(np.int16) + noise).clip(0, 255).astype(np.uint8)
        if np.random.rand() < 0.2:
            # лёгкий blur
            ksz = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksz, ksz), 0)
        return img
        # Заметка: НЕ делаем flip/rotation — порядок углов TL→TR→BR→BL
        # потерял бы смысл.

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = cv2.imread(str(img_path))  # BGR
        if img is None:
            raise RuntimeError(f"cannot read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img = self._augment(img)

        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img_tensor = self.normalize(img_tensor)

        # label: 8 нормализованных чисел
        parts = lbl_path.read_text(encoding="utf-8").strip().split()
        coords = torch.tensor([float(v) for v in parts[:8]], dtype=torch.float32)
        # → (4, 2)
        coords = coords.view(4, 2)

        return img_tensor, coords


class KeypointHead(nn.Module):
    """ResNet18 backbone (ImageNet pretrained) + FC head → 4 keypoint x 2."""
    def __init__(self):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # → 512 features
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 8),
            nn.Sigmoid(),  # координаты [0..1]
        )

    def forward(self, x):
        f = self.backbone(x)
        out = self.head(f)
        return out.view(-1, 4, 2)


def per_keypoint_pixel_error(pred: torch.Tensor, gt: torch.Tensor, img_size: int) -> torch.Tensor:
    """Среднее L2-расстояние между предсказанными и GT углами в пикселях crop'а."""
    diff = (pred - gt) * img_size  # из [0..1] в pixels
    dist = torch.sqrt((diff ** 2).sum(dim=-1))  # (B, 4)
    return dist.mean(dim=-1)  # (B,)


def evaluate(model: nn.Module, loader: DataLoader, device: str, img_size: int) -> dict:
    model.eval()
    all_errs = []
    with torch.no_grad():
        for imgs, gt in loader:
            imgs = imgs.to(device)
            gt = gt.to(device)
            pred = model(imgs)
            errs = per_keypoint_pixel_error(pred, gt, img_size)
            all_errs.append(errs.cpu().numpy())
    if not all_errs:
        return {"mean_px_err": float("inf"), "n": 0}
    arr = np.concatenate(all_errs)
    return {
        "mean_px_err": float(arr.mean()),
        "median_px_err": float(np.median(arr)),
        "p90_px_err": float(np.percentile(arr, 90)),
        "p95_px_err": float(np.percentile(arr, 95)),
        "n": int(len(arr)),
    }


def train_loop(args):
    device = args.device or pick_device()
    print(f"Device: {device}, img_size={args.img_size}")

    train_ds = CropKeypointsDataset(
        CROPS_ROOT / "train" / "images",
        CROPS_ROOT / "train" / "labels",
        img_size=args.img_size, augment=True,
    )
    val_ds = CropKeypointsDataset(
        CROPS_ROOT / "val" / "images",
        CROPS_ROOT / "val" / "labels",
        img_size=args.img_size, augment=False,
    )
    print(f"Train crops: {len(train_ds)}, Val crops: {len(val_ds)}")
    if not train_ds:
        raise SystemExit(f"нет train crops в {CROPS_ROOT}/train/. "
                         f"Запусти `python scripts/build_keypoint_crops.py` сначала.")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, pin_memory=False, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=False)

    model = KeypointHead().to(device)
    if args.weights:
        sd = torch.load(args.weights, map_location=device)
        model.load_state_dict(sd if isinstance(sd, dict) and "state_dict" not in sd else sd["state_dict"])
        print(f"Resumed weights: {args.weights}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_ROOT / f"keypoint_head_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    best_val_err = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss, n_seen = 0.0, 0
        for imgs, gt in train_loader:
            imgs = imgs.to(device)
            gt = gt.to(device)
            pred = model(imgs)
            loss = loss_fn(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)
            n_seen += imgs.size(0)
        scheduler.step()
        train_loss = epoch_loss / max(1, n_seen)

        val_metrics = evaluate(model, val_loader, device, args.img_size)
        history.append({"epoch": epoch, "train_loss": train_loss, **val_metrics,
                        "lr": optimizer.param_groups[0]["lr"]})
        print(f"Epoch {epoch:>3}/{args.epochs}  train_loss={train_loss:.5f}  "
              f"val mean_px_err={val_metrics['mean_px_err']:.2f}  "
              f"median={val_metrics['median_px_err']:.2f}  "
              f"p90={val_metrics['p90_px_err']:.2f}")

        # save best
        if val_metrics["mean_px_err"] < best_val_err:
            best_val_err = val_metrics["mean_px_err"]
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            no_improve += 1

        # save last
        torch.save(model.state_dict(), out_dir / "last.pt")

        if no_improve >= args.patience:
            print(f"Early stop on epoch {epoch} (no improve for {args.patience})")
            break

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nBest val mean_px_err: {best_val_err:.2f} px")
    print(f"Best weights: {out_dir / 'best.pt'}")
    return out_dir / "best.pt", out_dir


def per_region_eval(weights_path: Path, args, out_dir: Path):
    device = args.device or pick_device()
    model = KeypointHead().to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd if isinstance(sd, dict) and "state_dict" not in sd else sd["state_dict"])
    model.eval()

    print(f"\n=== per-region eval ===")
    results = {}
    for region in REGIONS:
        ds = CropKeypointsDataset(
            CROPS_ROOT / "test_per_region" / region / "images",
            CROPS_ROOT / "test_per_region" / region / "labels",
            img_size=args.img_size, augment=False,
        )
        if len(ds) == 0:
            print(f"  {region:<10} нет данных")
            continue
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, pin_memory=False)
        m = evaluate(model, loader, device, args.img_size)
        results[region] = m
        print(f"  {region:<10} n={m['n']:>4}  mean={m['mean_px_err']:.2f}px  "
              f"median={m['median_px_err']:.2f}px  p90={m['p90_px_err']:.2f}px")

    out = out_dir / "per_region_metrics.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nМетрики: {out}")


def main():
    args = parse_args()

    if args.eval_only:
        if not args.weights:
            raise SystemExit("--eval-only требует --weights <path>")
        out_dir = Path(args.weights).parent.parent  # runs/<run>/best.pt → runs/<run>/
        per_region_eval(Path(args.weights), args, out_dir)
        return

    best_path, out_dir = train_loop(args)
    per_region_eval(best_path, args, out_dir)


if __name__ == "__main__":
    main()
