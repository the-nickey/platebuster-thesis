"""Grid search оптимальных параметров post-processing для ResNet18 keypoint head.

Идея: ResNet18 даёт 4 угла, дальше до подачи в гомографию (для скрытия
номера) хочется применить 2 контролируемых post-proc'а:

  - `crop_padding` — сколько фона захватывать при cropping вокруг bbox
    перед подачей в ResNet18 (test-time, не требует переобучения)
  - `inflate_outward` — на сколько сдвинуть каждый предсказанный угол
    наружу от центра bbox (доля диагонали bbox; даёт контролируемый
    over-cover)

Метрика — асимметричная: для коммерческой задачи скрытия номера
**under-cover** (видна часть номера) гораздо хуже чем **over-cover**
(лого вылезает за рамку — некрасиво, но номер скрыт). Поэтому:

    asym_loss = w_under * max(0, 1 - coverage)^2
              + w_over  * max(0, area_ratio - 1)^2

С w_under=10, w_over=1 (10:1 в пользу overflow-tolerance).

Изолируем эффект ResNet18+post-proc от bbox-детектора: cropping идёт
вокруг GT bbox, а не предсказанного. Это убирает шум от детектора и
позволяет говорить именно про head.

Запуск:
  .venv/bin/python scripts/training/grid_search_keypoint_postproc.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "streamlit_app"))
from inference import KeypointHead, crop_with_padding, preprocess_kpt_input  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
TEST_DIR = REPO / "data" / "processed" / "unified" / "test_per_region" / "russian"
KPT_HEAD = REPO / "runs_from_cloud" / "runs" / "keypoint_head_20260503_131404" / "best.pt"

PADDINGS = [0.20, 0.25, 0.28, 0.30, 0.32, 0.35, 0.40]
INFLATES = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
TTAS = [False, True]
# 2026-05-05: пересмотрели после визуального тестирования. Раньше ставили
# W_UNDER=10, W_OVER=1 — это давало area_ratio 1.22 (лого выходит за плашку
# на 22 % площади), визуально перегруз. Теперь симметрично: лого должен
# быть «как можно ровнее по плашке». Оптимум ≈ area_ratio 1.0.
W_UNDER = 1.0
W_OVER = 1.0
FLIP_IDX = (1, 0, 3, 2)  # TL→TR→BR→BL после h-flip превращается в [TR,TL,BR,BL]


def load_samples() -> list[tuple[np.ndarray, tuple[int, int, int, int], np.ndarray | None]]:
    out = []
    for img_path in sorted((TEST_DIR / "images").glob("*.jpg")):
        lbl = TEST_DIR / "labels" / f"{img_path.stem}.txt"
        if not lbl.exists():
            continue
        parts = lbl.read_text().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        gt_bbox = (
            int((cx - bw / 2) * W), int((cy - bh / 2) * H),
            int((cx + bw / 2) * W), int((cy + bh / 2) * H),
        )

        gt_corners = None
        if len(parts) >= 17:
            kp = [
                (float(parts[5 + i * 3]) * W, float(parts[5 + i * 3 + 1]) * H,
                 int(float(parts[5 + i * 3 + 2])))
                for i in range(4)
            ]
            if all(v == 2 for _, _, v in kp):
                gt_corners = np.array([(x, y) for x, y, _ in kp], dtype=np.float32)

        out.append((img_rgb, gt_bbox, gt_corners))
    return out


def predict_corners(model: KeypointHead, img_rgb: np.ndarray,
                    gt_bbox: tuple[int, int, int, int], padding: float,
                    use_tta: bool = False) -> np.ndarray | None:
    crop, canvas_origin, canvas_side = crop_with_padding(
        img_rgb, gt_bbox, padding=padding, target_size=192,
    )
    if crop is None:
        return None
    t = preprocess_kpt_input(crop)
    with torch.no_grad():
        kpts_norm = model(t)[0].numpy().reshape(4, 2)

    if use_tta:
        # h-flip crop, predict, отразить x обратно, переупорядочить углы
        t_flip = preprocess_kpt_input(crop[:, ::-1, :].copy())
        with torch.no_grad():
            kpts_flip = model(t_flip)[0].numpy().reshape(4, 2)
        kpts_flip[:, 0] = 1.0 - kpts_flip[:, 0]
        kpts_flip = kpts_flip[list(FLIP_IDX)]
        kpts_norm = (kpts_norm + kpts_flip) / 2.0

    kpts_orig = canvas_origin + kpts_norm * canvas_side
    return kpts_orig.astype(np.float32)


def inflate_outward(corners: np.ndarray, gt_bbox: tuple[int, int, int, int],
                    inflate: float) -> np.ndarray:
    if inflate <= 0:
        return corners
    x1, y1, x2, y2 = gt_bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    diag = float(np.hypot(x2 - x1, y2 - y1))
    shift = inflate * diag
    out = corners.copy()
    for i in range(4):
        dx, dy = out[i, 0] - cx, out[i, 1] - cy
        n = float(np.hypot(dx, dy)) + 1e-6
        out[i, 0] += dx / n * shift
        out[i, 1] += dy / n * shift
    return out


def coverage_of_bbox(corners: np.ndarray, gt_bbox: tuple[int, int, int, int]
                     ) -> tuple[float, float]:
    """Возвращает (coverage, area_ratio).
       coverage = доля GT bbox площади покрытая pred quadrilateral
       area_ratio = pred_quad_area / gt_bbox_area"""
    x1, y1, x2, y2 = gt_bbox
    quad = cv2.convexHull(corners.astype(np.float32)).reshape(-1, 2)

    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return 0.0, 0.0

    mask = np.zeros((bh, bw), dtype=np.uint8)
    quad_local = (quad - np.array([x1, y1])).astype(np.int32)
    cv2.fillPoly(mask, [quad_local], 1)
    coverage = float(mask.sum()) / float(bh * bw)

    pred_area = float(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = pred_area / float(bh * bw)
    return coverage, area_ratio


def mean_corner_px_err(corners: np.ndarray, gt_corners: np.ndarray) -> float:
    """Mean px-error: для каждого pred-угла берём ближайший GT-угол.
    Это устойчиво к разному порядку углов (TL/TR/BR/BL может расходиться)."""
    if gt_corners is None:
        return float("nan")
    err = 0.0
    for pc in corners:
        d = np.hypot(gt_corners[:, 0] - pc[0], gt_corners[:, 1] - pc[1])
        err += float(d.min())
    return err / 4.0


def main():
    print(f"loading samples from {TEST_DIR.relative_to(REPO)} ...")
    samples = load_samples()
    n_with_gt_corners = sum(1 for _, _, c in samples if c is not None)
    print(f"  {len(samples)} samples ({n_with_gt_corners} with GT corners)\n")

    print(f"loading kpt head from {KPT_HEAD.relative_to(REPO)} ...")
    model = KeypointHead()
    ckpt = torch.load(str(KPT_HEAD), map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(sd)
    model.eval()
    print("  ok\n")

    rows = []
    for use_tta in TTAS:
        for p in PADDINGS:
            for inf in INFLATES:
                covs, ars, errs = [], [], []
                for img_rgb, gt_bbox, gt_corners in samples:
                    pred = predict_corners(model, img_rgb, gt_bbox, p, use_tta=use_tta)
                    if pred is None:
                        continue
                    pred = inflate_outward(pred, gt_bbox, inf)
                    cov, ar = coverage_of_bbox(pred, gt_bbox)
                    covs.append(cov)
                    ars.append(ar)
                    if gt_corners is not None:
                        errs.append(mean_corner_px_err(pred, gt_corners))

                cov_mean = float(np.mean(covs))
                cov_p10 = float(np.percentile(covs, 10))
                ar_mean = float(np.mean(ars))
                err_mean = float(np.mean(errs)) if errs else float("nan")

                under = max(0.0, 1.0 - cov_mean)
                over = max(0.0, ar_mean - 1.0)
                asym = W_UNDER * under * under + W_OVER * over * over

                cov_pct_99 = float(np.mean(np.array(covs) >= 0.99))

                rows.append((p, inf, use_tta, cov_mean, cov_p10, cov_pct_99,
                             ar_mean, err_mean, asym))

    rows.sort(key=lambda r: r[8])  # by asym_loss ascending

    print("=" * 110)
    print(f"  Grid search results — отсортированы по asym_loss (ascending = лучше)")
    print(f"  asym_loss = {W_UNDER}·under_cover² + {W_OVER}·over_cover²")
    print("=" * 110)
    print(f"{'pad':>5} {'inflate':>8} {'tta':>5} {'cov_mean':>9} {'cov_p10':>8} "
          f"{'cov≥0.99':>9} {'area_ratio':>11} {'px_err':>8} {'asym_loss':>10}")
    print("-" * 110)
    for (p, inf, tta, cov_mean, cov_p10, cov_99, ar_mean, err_mean, asym) in rows[:20]:
        print(f"{p:>5.2f} {inf:>8.2f} {str(tta):>5} {cov_mean:>9.4f} {cov_p10:>8.4f} "
              f"{cov_99:>9.3f} {ar_mean:>11.3f} {err_mean:>8.2f} {asym:>10.4f}")

    print()
    print("=" * 110)
    print("  TOP-3 кандидата по разным критериям:")
    print("=" * 110)

    by_asym = sorted(rows, key=lambda r: r[8])
    by_cov_p10 = sorted(rows, key=lambda r: -r[4])
    by_err = sorted([r for r in rows if not np.isnan(r[7])], key=lambda r: r[7])

    def fmt(r):
        p, inf, tta, cm, cp10, c99, ar, err, a = r
        return (f"  pad={p:.2f} inflate={inf:.2f} tta={str(tta):<5}  "
                f"cov={cm:.4f} (p10={cp10:.3f}) area={ar:.3f} err={err:.2f} asym={a:.4f}")

    print("[asym_loss min — самый сбалансированный]")
    for r in by_asym[:3]: print(fmt(r))
    print("\n[best worst-case coverage (p10 max) — стабильность]")
    for r in by_cov_p10[:3]: print(fmt(r))
    print("\n[best mean px_err — точность углов абсолютно]")
    for r in by_err[:3]: print(fmt(r))

    print()
    print("=" * 110)
    print("  TTA ON vs OFF на одинаковых (padding, inflate) — выигрыш в px_err:")
    print("=" * 110)
    by_key = {(r[0], r[1], r[2]): r for r in rows}
    deltas = []
    for p in PADDINGS:
        for inf in INFLATES:
            r_off = by_key.get((p, inf, False))
            r_on = by_key.get((p, inf, True))
            if r_off and r_on and not np.isnan(r_off[7]) and not np.isnan(r_on[7]):
                d_err = r_on[7] - r_off[7]
                d_cov = r_on[3] - r_off[3]
                d_asym = r_on[8] - r_off[8]
                deltas.append((p, inf, d_err, d_cov, d_asym))
    if deltas:
        avg_d_err = float(np.mean([d[2] for d in deltas]))
        avg_d_cov = float(np.mean([d[3] for d in deltas]))
        avg_d_asym = float(np.mean([d[4] for d in deltas]))
        print(f"  средний Δ px_err  (tta - no_tta): {avg_d_err:+.4f}  (отрицательный = TTA лучше)")
        print(f"  средний Δ cov_mean (tta - no_tta): {avg_d_cov:+.4f}  (положительный = TTA лучше)")
        print(f"  средний Δ asym_loss (tta - no_tta): {avg_d_asym:+.4f}  (отрицательный = TTA лучше)")


if __name__ == "__main__":
    main()
