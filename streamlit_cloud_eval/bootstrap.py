"""
Библиотечный API бутстрэпа: те же формулы, что в scripts/bootstrap_metrics.py,
но с **precompute-стратегией**, критичной для Streamlit Cloud (1 vCPU).

Идея оптимизации:
1. Для каждой картинки один раз прокатываем матчинг pred↔gt по каждому уровню
   IoU и сохраняем компактные per-image массивы: scores, tp_at_each_iou, n_gt.
2. На бутстрэп-итерации ресэмплируем картинки и просто `np.concatenate` их
   per-image массивов, сортируем по score, считаем PR-curve. Это в 100+ раз
   быстрее, чем переcчитывать матчинг.

Без этой оптимизации B=2000 × 2000 кадров CCPD занимал бы часы; с ней —
секунды на модель × регион.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

REGIONS = ["ccpd", "russian", "european", "openalpr", "generic"]
IOU_LEVELS = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def list_available_models(pred_dir: Path) -> list[str]:
    if not pred_dir.exists():
        return []
    return sorted(p.name for p in pred_dir.iterdir()
                  if p.is_dir() and any(p.glob("*.jsonl")))


def list_available_regions(pred_dir: Path, model: str) -> list[str]:
    mdir = pred_dir / model
    if not mdir.exists():
        return []
    return sorted(p.stem for p in mdir.glob("*.jsonl"))


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


# ---------------------------------------------------------------------------
# Precompute (один раз на модель × регион)
# ---------------------------------------------------------------------------


@dataclass
class PerImagePrecompute:
    """Скомпилированный кеш по одной картинке для бутстрэпа.

    image_id     — для совпадения в paired-режиме
    n_gt         — сколько было GT-боксов
    scores       — np.array длины P (числа уверенности всех predictions)
    tp           — np.ndarray (P, len(IOU_LEVELS)) булевых: tp[p, l] = 1, если
                   prediction p при IoU-пороге IOU_LEVELS[l] засчитан как TP
    kpt_err_px   — список пиксельных ошибок углов на matched парах (под IoU 0.5)
    kpt_pck      — список 0/1 (был ли угол в пределах 5% диагонали)
    latency_ms   — latency single-image inference (NaN если не замеряли)
    """
    image_id: str
    n_gt: int
    scores: np.ndarray          # (P,)
    tp: np.ndarray              # (P, len(IOU_LEVELS))
    kpt_err_px: np.ndarray      # (K,)
    kpt_pck: np.ndarray         # (K,)
    latency_ms: float = float("nan")


def precompute_one_record(rec: dict) -> PerImagePrecompute:
    gt = np.asarray(rec.get("gt_boxes_xyxyn", []), dtype=float)
    pred = np.asarray(rec.get("pred_boxes_xyxyn", []), dtype=float)
    scores = np.asarray(rec.get("pred_scores", []), dtype=float)
    gt_kpts = rec.get("gt_kpts_xyn", []) or []
    pred_kpts = rec.get("pred_kpts_xyn", []) or []
    W, H = rec.get("image_size", [1, 1])
    n_gt = len(gt)
    n_pred = len(pred)
    n_iou = len(IOU_LEVELS)

    if n_pred == 0:
        return PerImagePrecompute(
            image_id=rec.get("image_id", ""),
            n_gt=n_gt,
            scores=np.zeros(0),
            tp=np.zeros((0, n_iou), dtype=np.int8),
            kpt_err_px=np.zeros(0),
            kpt_pck=np.zeros(0, dtype=np.int8),
        )

    order = np.argsort(-scores)
    pred = pred[order]
    scores = scores[order]
    pred_kpts_sorted = [pred_kpts[i] for i in order] if pred_kpts else []

    if n_gt == 0:
        return PerImagePrecompute(
            image_id=rec.get("image_id", ""),
            n_gt=0,
            scores=scores,
            tp=np.zeros((n_pred, n_iou), dtype=np.int8),
            kpt_err_px=np.zeros(0),
            kpt_pck=np.zeros(0, dtype=np.int8),
        )

    ious = iou_xyxy(pred, gt)
    tp_matrix = np.zeros((n_pred, n_iou), dtype=np.int8)
    for li, iou_thr in enumerate(IOU_LEVELS):
        gt_used = np.zeros(n_gt, dtype=bool)
        for pi in range(n_pred):
            avail = ious[pi].copy()
            avail[gt_used] = 0
            avail[avail < iou_thr] = 0
            if avail.max() > 0:
                j = int(avail.argmax())
                gt_used[j] = True
                tp_matrix[pi, li] = 1

    kpt_err: list[float] = []
    kpt_pck: list[int] = []
    if pred_kpts_sorted and gt_kpts and not any(k is None for k in gt_kpts):
        gt_used = np.zeros(n_gt, dtype=bool)
        for pi in range(n_pred):
            avail = ious[pi].copy()
            avail[gt_used] = 0
            avail[avail < 0.5] = 0
            if avail.max() == 0:
                continue
            j = int(avail.argmax())
            gt_used[j] = True
            if pi >= len(pred_kpts_sorted) or pred_kpts_sorted[pi] is None:
                continue
            pk = pred_kpts_sorted[pi]
            gk = gt_kpts[j]
            if gk is None or len(gk) < 4:
                continue
            x1, y1, x2, y2 = gt[j]
            diag_n = float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
            for kk in range(4):
                if gk[kk][2] <= 0:
                    continue
                dx_n = pk[kk][0] - gk[kk][0]
                dy_n = pk[kk][1] - gk[kk][1]
                err_px = float(np.sqrt((dx_n * W) ** 2 + (dy_n * H) ** 2))
                err_n = float(np.sqrt(dx_n ** 2 + dy_n ** 2))
                kpt_err.append(err_px)
                kpt_pck.append(1 if (diag_n > 0 and err_n / diag_n <= 0.05) else 0)

    return PerImagePrecompute(
        image_id=rec.get("image_id", ""),
        n_gt=n_gt,
        scores=scores,
        tp=tp_matrix,
        kpt_err_px=np.asarray(kpt_err, dtype=float),
        kpt_pck=np.asarray(kpt_pck, dtype=np.int8),
    )


def precompute_records(records: list[dict],
                       progress_cb=None) -> list[PerImagePrecompute]:
    out = []
    n = len(records)
    for i, r in enumerate(records):
        out.append(precompute_one_record(r))
        if progress_cb and (i + 1) % max(1, n // 20) == 0:
            progress_cb(i + 1, n)
    return out


def attach_latencies(precompute: list[PerImagePrecompute],
                     latency_jsonl: Path) -> int:
    """Подмерживает per-image latency_ms из <region>__latency.jsonl.
    Возвращает число обновлённых записей."""
    if not latency_jsonl.exists():
        return 0
    by_id = {p.image_id: p for p in precompute}
    n = 0
    with latency_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = rec.get("image_id", "")
            if iid in by_id:
                by_id[iid].latency_ms = float(rec.get("latency_ms", float("nan")))
                n += 1
    return n


# ---------------------------------------------------------------------------
# Финализация метрик из precompute
# ---------------------------------------------------------------------------


def _ap_from_arrays(scores: np.ndarray, tp_col: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp_sorted = tp_col[order]
    cum_tp = np.cumsum(tp_sorted)
    cum_fp = np.cumsum(1 - tp_sorted)
    recalls = cum_tp / n_gt
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
    rec_levels = np.linspace(0, 1, 101)
    interp_p = np.zeros_like(rec_levels)
    for i, rec_lvl in enumerate(rec_levels):
        mask = recalls >= rec_lvl
        interp_p[i] = precisions[mask].max() if mask.any() else 0.0
    return float(interp_p.mean())


def metrics_from_precompute(pre: list[PerImagePrecompute]) -> dict:
    if not pre:
        return {"mAP50": float("nan"), "mAP50_95": float("nan"),
                "mean_kpt_err_px": float("nan"), "PCK_05": float("nan"),
                "n_kpt_matched": 0}
    has_any = any(len(p.scores) for p in pre)
    scores = np.concatenate([p.scores for p in pre]) if has_any else np.zeros(0)
    tp = (np.concatenate([p.tp for p in pre], axis=0) if has_any
          else np.zeros((0, len(IOU_LEVELS)), dtype=np.int8))
    n_gt = int(sum(p.n_gt for p in pre))
    aps = [_ap_from_arrays(scores, tp[:, li], n_gt) for li in range(len(IOU_LEVELS))]
    map50 = aps[0]
    map50_95 = float(np.nanmean(aps))

    has_kpt = any(len(p.kpt_err_px) for p in pre)
    kpt_err = np.concatenate([p.kpt_err_px for p in pre]) if has_kpt else np.zeros(0)
    kpt_pck = np.concatenate([p.kpt_pck for p in pre]) if has_kpt else np.zeros(0)
    if len(kpt_err) == 0:
        mean_err, pck05 = float("nan"), float("nan")
    else:
        mean_err = float(kpt_err.mean())
        pck05 = float(kpt_pck.mean())

    # latency
    latencies = np.asarray([p.latency_ms for p in pre], dtype=float)
    latencies = latencies[np.isfinite(latencies)]
    if len(latencies) == 0:
        lat_p50, lat_p95 = float("nan"), float("nan")
    else:
        lat_p50 = float(np.median(latencies))
        lat_p95 = float(np.percentile(latencies, 95))

    return {"mAP50": map50, "mAP50_95": map50_95,
            "mean_kpt_err_px": mean_err, "PCK_05": pck05,
            "n_kpt_matched": len(kpt_err),
            "latency_p50_ms": lat_p50,
            "latency_p95_ms": lat_p95,
            "n_latency": len(latencies)}


# ---------------------------------------------------------------------------
# Бутстрэп
# ---------------------------------------------------------------------------


def percentile_ci(values: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(finite, 100 * alpha / 2))
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return (lo, hi)


def single_bootstrap(precompute: list[PerImagePrecompute], B: int = 2000,
                     seed: int = 42, alpha: float = 0.05,
                     progress_cb=None) -> dict:
    point = metrics_from_precompute(precompute)
    rng = np.random.default_rng(seed)
    n = len(precompute)
    keys = ("mAP50", "mAP50_95", "mean_kpt_err_px", "PCK_05",
            "latency_p50_ms", "latency_p95_ms")
    samples = {k: np.zeros(B, dtype=float) for k in keys}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sub = [precompute[i] for i in idx]
        m = metrics_from_precompute(sub)
        for k in keys:
            samples[k][b] = m[k]
        if progress_cb and (b + 1) % max(1, B // 20) == 0:
            progress_cb(b + 1, B)
    out = {"_n_images": n, "_B": B}
    for k in keys:
        lo, hi = percentile_ci(samples[k], alpha)
        out[k] = {"point": point[k], "ci_low": lo, "ci_high": hi}
    return out


def paired_bootstrap(pre_a: list[PerImagePrecompute],
                     pre_b: list[PerImagePrecompute],
                     B: int = 2000, seed: int = 42, alpha: float = 0.05,
                     progress_cb=None) -> dict:
    by_id_a = {p.image_id: p for p in pre_a}
    by_id_b = {p.image_id: p for p in pre_b}
    common_ids = sorted(set(by_id_a) & set(by_id_b))
    if not common_ids:
        return {"error": "no common image_ids"}
    a_aligned = [by_id_a[i] for i in common_ids]
    b_aligned = [by_id_b[i] for i in common_ids]
    point_a = metrics_from_precompute(a_aligned)
    point_b = metrics_from_precompute(b_aligned)
    rng = np.random.default_rng(seed)
    n = len(common_ids)
    keys = ("mAP50", "mAP50_95", "mean_kpt_err_px", "PCK_05",
            "latency_p50_ms", "latency_p95_ms")
    diffs = {k: np.zeros(B, dtype=float) for k in keys}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sub_a = [a_aligned[i] for i in idx]
        sub_b = [b_aligned[i] for i in idx]
        ma = metrics_from_precompute(sub_a)
        mb = metrics_from_precompute(sub_b)
        for k in keys:
            diffs[k][b] = ma[k] - mb[k]
        if progress_cb and (b + 1) % max(1, B // 20) == 0:
            progress_cb(b + 1, B)
    out = {"_n_common": n, "_B": B,
           "point_a": point_a, "point_b": point_b}
    for k in keys:
        finite = diffs[k][np.isfinite(diffs[k])]
        if len(finite) == 0:
            out[f"diff_{k}"] = {"point": float("nan"),
                                "ci_low": float("nan"),
                                "ci_high": float("nan"),
                                "p_value": float("nan")}
            continue
        lo, hi = percentile_ci(diffs[k], alpha)
        p_pos = float((finite >= 0).mean())
        p_neg = float((finite <= 0).mean())
        p_two = min(1.0, 2.0 * min(p_pos, p_neg))
        out[f"diff_{k}"] = {
            "point": float(point_a[k] - point_b[k]),
            "ci_low": lo, "ci_high": hi, "p_value": p_two,
        }
    return out
