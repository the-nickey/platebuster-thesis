"""Бенчмарк latency всех моделей зоопарка на CPU (M3 Pro / aarch64).

Меряет два разных уровня:
  1. Чистый detector forward pass (одна сеть, без post-proc)
  2. End-to-end pipeline (то, что реально видит пользователь Streamlit):
     detect → crop → keypoint head → inflate_outward → координаты в исходных пикселях

Запуск:
  .venv/bin/python scripts/benchmark_latency.py
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "streamlit_app"))

from inference import (  # noqa: E402
    TwoStagePipeline,
    SinglePosePipeline,
    BboxOnlyPipeline,
    ClassicalPipeline,
    RFDETRPipeline,
)


WEIGHTS = {
    "yolo11n_detect_v2": REPO / "runs_from_cloud/runs/yolo11n_cuda_20260504_113721/weights/best.pt",
    "yolo12n_detect_v2": REPO / "runs_from_cloud/runs/yolo12n_cuda_20260504_130347/weights/best.pt",
    "yolo11n_pose_v2":   REPO / "runs_from_cloud/runs/yolo11n_pose_cuda_v2/weights/best.pt",
    "kpt_head":          REPO / "runs_from_cloud/runs/keypoint_head_20260503_131404/best.pt",
    "rfdetr_nano":       REPO / "runs_from_cloud/runs/rfdetr_nano_20260504_155433/checkpoint_best_ema.pth",
    "rfdetr_medium":     REPO / "runs_from_cloud/runs/rfdetr_medium_20260503_132525/checkpoint_best_ema.pth",
}

N_WARMUP = 3   # прогревочных итераций (выкидываем — JIT-компиляция, lazy load)
N_BENCH = 30   # измерительных итераций


def load_test_images() -> list[np.ndarray]:
    """Берём 20 фотографий разных регионов: ccpd / russian / european / openalpr / generic."""
    out = []
    for region in ("russian", "ccpd", "european", "openalpr", "generic"):
        img_dir = REPO / "data/processed/unified/test_per_region" / region / "images"
        if not img_dir.exists():
            continue
        for p in sorted(img_dir.glob("*.jpg"))[:4]:
            img = cv2.imread(str(p))
            if img is not None:
                out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return out


def bench(name: str, fn: Callable[[np.ndarray], object],
          imgs: list[np.ndarray]) -> dict:
    """Прогоняет fn(img) много раз, возвращает median/p95/min/max."""
    # прогрев
    for i in range(N_WARMUP):
        fn(imgs[i % len(imgs)])
    # замер
    timings = []
    for i in range(N_BENCH):
        img = imgs[i % len(imgs)]
        t0 = time.perf_counter()
        _ = fn(img)
        timings.append((time.perf_counter() - t0) * 1000)
    arr = np.asarray(timings)
    print(f"  {name:<55} median={np.median(arr):7.1f} ms  "
          f"p95={np.percentile(arr, 95):7.1f}  "
          f"min={arr.min():6.1f}  n={len(arr)}")
    return dict(
        name=name, device="CPU",
        median_ms=float(np.median(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        min_ms=float(arr.min()),
        max_ms=float(arr.max()),
        n=int(len(arr)),
    )


def main():
    print(f"loading test images...")
    imgs = load_test_images()
    print(f"  {len(imgs)} images, размеры: {[(im.shape[1], im.shape[0]) for im in imgs[:3]]} ...\n")

    results: list[dict] = []

    # ────────────────── 1. CLASSICAL (rule-based, без обучения) ──────────────────
    print("=== 1. classical (full rule-based pipeline) ===")
    cls_pipe = ClassicalPipeline()
    results.append(bench(
        "classical (Canny + morph + approxPolyDP)",
        lambda im: cls_pipe(im, conf=0.3, imgsz=640),
        imgs,
    ))

    # ────────────────── 2. YOLO детекторы (только bbox forward) ──────────────────
    print("\n=== 2. YOLO детекторы — только bbox (BboxOnlyPipeline) ===")
    for key in ("yolo11n_detect_v2", "yolo12n_detect_v2"):
        pipe = BboxOnlyPipeline(detector_path=WEIGHTS[key], device="cpu")
        results.append(bench(
            f"{key} (bbox-only)",
            lambda im, p=pipe: p(im, conf=0.3, imgsz=640),
            imgs,
        ))

    # ────────────────── 3. YOLO11n-pose (single-stage: bbox + 4 угла) ──────────────────
    print("\n=== 3. YOLO11n-pose v2 (single-stage, bbox + углы из одной сети) ===")
    pose_pipe = SinglePosePipeline(pose_path=WEIGHTS["yolo11n_pose_v2"], device="cpu")
    results.append(bench(
        "yolo11n_pose_v2 (single-stage end-to-end)",
        lambda im: pose_pipe(im, conf=0.3, imgsz=640),
        imgs,
    ))

    # ────────────────── 4. 2-stage: YOLO + ResNet18 + inflate (наш production-сетап) ──────────────────
    print("\n=== 4. 2-stage end-to-end: detect + ResNet18 keypoint + inflate_outward=0.05 ===")
    for key in ("yolo11n_detect_v2", "yolo12n_detect_v2"):
        pipe = TwoStagePipeline(
            detector_path=WEIGHTS[key],
            kpt_head_path=WEIGHTS["kpt_head"],
            device="cpu",
        )
        results.append(bench(
            f"{key} + ResNet18 + inflate (production setup)",
            lambda im, p=pipe: p(im, conf=0.3, imgsz=640),
            imgs,
        ))

    # ────────────────── 5. 2-stage с TTA ──────────────────
    print("\n=== 5. 2-stage с TTA (flip-averaging keypoint head) ===")
    pipe_tta = TwoStagePipeline(
        detector_path=WEIGHTS["yolo11n_detect_v2"],
        kpt_head_path=WEIGHTS["kpt_head"],
        device="cpu",
        use_tta=True,
    )
    results.append(bench(
        "yolo11n_detect_v2 + ResNet18 + TTA (2× kpt-forward)",
        lambda im: pipe_tta(im, conf=0.3, imgsz=640),
        imgs,
    ))

    # ────────────────── 6. RF-DETR Nano (новый) и Medium (legacy) ──────────────────
    print("\n=== 6. RF-DETR (transformer, без kpt-головы) ===")
    for size, key in (("nano", "rfdetr_nano"), ("medium", "rfdetr_medium")):
        try:
            pipe = RFDETRPipeline(ckpt_path=WEIGHTS[key], device="cpu", size=size)
            results.append(bench(
                f"rfdetr_{size} (bbox-only)",
                lambda im, p=pipe: p(im, conf=0.3),
                imgs,
            ))
        except Exception as e:
            print(f"  rfdetr_{size}: FAILED — {e}")

    # ────────────────── 7. RF-DETR Nano + ResNet18 (топ mAP кандидат с углами) ──────────────────
    print("\n=== 7. RF-DETR Nano + ResNet18 (топ-mAP transformer-кандидат) ===")
    try:
        pipe = RFDETRPipeline(
            ckpt_path=WEIGHTS["rfdetr_nano"],
            kpt_head_path=WEIGHTS["kpt_head"],
            device="cpu",
            size="nano",
        )
        results.append(bench(
            "rfdetr_nano + ResNet18 + inflate (топ-mAP с углами)",
            lambda im, p=pipe: p(im, conf=0.3),
            imgs,
        ))
    except Exception as e:
        print(f"  FAILED — {e}")

    # ────────────────── save + print summary ──────────────────
    out = REPO / "runs/_summary/latency_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n=== сохранено в {out.relative_to(REPO)} ===")

    print()
    print("=" * 80)
    print(f"{'pipeline':<55} {'median':>8} {'p95':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["median_ms"]):
        print(f"{r['name']:<55} {r['median_ms']:>7.1f} ms {r['p95_ms']:>7.1f}")


if __name__ == "__main__":
    main()
