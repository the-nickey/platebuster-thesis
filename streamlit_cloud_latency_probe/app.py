"""
Streamlit-приложение для замера latency в Streamlit Community Cloud (1 vCPU).

Поток работы:
1. Пользователь поднимает локальный HTTP-сервер в корне репо thesis и cloudflared
   туннель, копирует tunnel URL в это приложение.
2. App качает sample_image_ids.json, веса всех моделей и тестовые картинки
   из туннеля в /tmp/ (один раз за сессию контейнера).
3. Для каждой пары (модель × регион) — single-image inference c warmup,
   per-image latency пишется в /tmp/results/<model>/<region>__latency.jsonl.
4. По завершении — кнопка «Скачать zip» отдаёт результаты пользователю.
5. Пользователь кладёт распакованные JSONL в streamlit_cloud_eval/data/predictions/
   и перепушивает bootstrap-репу — там в таблицах появляются latency-колонки с ДИ.

Запуск локально:
    streamlit run streamlit_cloud_latency_probe/app.py

Деплой в Streamlit Cloud:
    1. Пушнуть содержимое этой папки как корень отдельной репы
    2. На share.streamlit.io указать `app.py` точкой входа
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import make_inferer  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
TMP_ROOT = Path(os.environ.get("LATENCY_PROBE_TMP", "/tmp/platebuster_latency"))
TMP_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = TMP_ROOT / "results"
ASSETS_DIR = TMP_ROOT / "assets"
WARMUP = 5

st.set_page_config(page_title="platebuster — latency probe",
                   page_icon="⏱", layout="wide")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_manifest() -> dict:
    return json.loads((APP_DIR / "manifest.json").read_text(encoding="utf-8"))


def fetch_file(tunnel_url: str, rel_path: str, dst_root: Path) -> Path:
    """Качает файл с tunnel_url/<rel_path> в dst_root/<rel_path>. Кеширует."""
    url = tunnel_url.rstrip("/") + "/" + rel_path.lstrip("/")
    dst = dst_root / rel_path
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=64 * 1024):
            f.write(chunk)
    return dst


def fetch_sample_ids(tunnel_url: str, manifest: dict) -> dict:
    """Качает sample_image_ids.json. Возвращает {region: [image_id]}."""
    p = fetch_file(tunnel_url, manifest["sample_image_ids"], ASSETS_DIR)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("sample", data)


def fetch_image(tunnel_url: str, manifest: dict, region: str,
                image_id: str) -> Path | None:
    """Качает картинку из test_per_region/<region>/images/. cv2 читает .jpg,
    но в исходниках может быть .jpg/.jpeg/.png — пробуем по очереди."""
    base = manifest["test_images_root"].rstrip("/")
    for ext in (".jpg", ".jpeg", ".png"):
        rel = f"{base}/{region}/images/{image_id}{ext}"
        try:
            return fetch_file(tunnel_url, rel, ASSETS_DIR)
        except requests.RequestException:
            continue
    return None


def fetch_weights(tunnel_url: str, manifest: dict, model_name: str,
                  progress=None) -> dict:
    """Качает веса для модели (для two_stage — две пары). Возвращает абсолютные
    пути в виде словаря cfg с заменёнными ключами weights/bbox_weights."""
    cfg = dict(manifest["models"][model_name])
    if "weights" in cfg and cfg.get("type") != "classical":
        p = fetch_file(tunnel_url, cfg["weights"], ASSETS_DIR)
        cfg["weights"] = str(p.relative_to(ASSETS_DIR))
        if progress: progress(f"weights: {Path(cfg['weights']).name} OK")
    if "bbox_weights" in cfg:
        p = fetch_file(tunnel_url, cfg["bbox_weights"], ASSETS_DIR)
        cfg["bbox_weights"] = str(p.relative_to(ASSETS_DIR))
        if progress: progress(f"bbox_weights: {Path(cfg['bbox_weights']).name} OK")
    return cfg


def measure_for(model_name: str, region: str, sample_ids: list[str],
                tunnel_url: str, manifest: dict, status_cb) -> dict:
    """Скачивает веса и картинки, делает warmup, замеряет per-image latency."""
    status_cb(f"[{model_name} | {region}] скачиваем веса...")
    cfg = fetch_weights(tunnel_url, manifest, model_name, progress=status_cb)

    # картинки
    status_cb(f"[{model_name} | {region}] скачиваем картинки...")
    image_paths: list[Path] = []
    for i, iid in enumerate(sample_ids):
        p = fetch_image(tunnel_url, manifest, region, iid)
        if p is not None:
            image_paths.append(p)
        if (i + 1) % 50 == 0:
            status_cb(f"[{model_name} | {region}] картинки: {i+1}/{len(sample_ids)}")

    if not image_paths:
        return {"error": "no images fetched"}

    # инференс
    status_cb(f"[{model_name} | {region}] загружаем модель в RAM...")
    runner = make_inferer(cfg, ASSETS_DIR, device="cpu")

    status_cb(f"[{model_name} | {region}] warmup × {WARMUP}...")
    for img in image_paths[:WARMUP]:
        try:
            runner(img)
        except Exception as e:
            status_cb(f"  warmup err: {e}")

    out_path = RESULTS_DIR / model_name / f"{region}__latency.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as f:
        for i, img in enumerate(image_paths):
            t1 = time.perf_counter()
            try:
                runner(img)
            except Exception as e:
                status_cb(f"  err on {img.name}: {e}")
                continue
            dt_ms = (time.perf_counter() - t1) * 1000
            latencies.append(dt_ms)
            f.write(json.dumps({"image_id": img.stem, "latency_ms": dt_ms},
                               ensure_ascii=False) + "\n")
            if (i + 1) % 20 == 0:
                eta = (time.time() - t0) / (i + 1) * (len(image_paths) - i - 1)
                status_cb(f"[{model_name} | {region}] {i+1}/{len(image_paths)} "
                          f"(ETA {eta:.0f}s)")
    arr = np.asarray(latencies)
    return {
        "n": len(latencies),
        "elapsed_s": time.time() - t0,
        "p50_ms": float(np.median(arr)) if len(arr) else float("nan"),
        "p95_ms": float(np.percentile(arr, 95)) if len(arr) else float("nan"),
    }


def make_results_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in RESULTS_DIR.rglob("*.jsonl"):
            arc = p.relative_to(RESULTS_DIR)
            z.write(p, arcname=str(arc))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


st.title("⏱ Latency probe — Streamlit Cloud (1 vCPU)")
st.caption(
    "Замер per-image latency для шести моделей-кандидатов в реальной среде "
    "развёртывания. Веса и картинки подгружаются с локального cloudflared-туннеля "
    "автора (см. README). Результаты — `__latency.jsonl` для каждой модели × "
    "региона — скачиваешь и кладёшь в репу bootstrap-сервиса."
)

manifest = load_manifest()

with st.sidebar:
    st.markdown("### Туннель")
    default_url = os.environ.get("TUNNEL_URL", "")
    tunnel_url = st.text_input(
        "cloudflared tunnel URL",
        value=default_url,
        placeholder="https://xxx-yyy-zzz.trycloudflare.com",
        help="Запусти на маке: `python -m http.server 8080` в корне thesis/ "
             "и `cloudflared tunnel --url http://localhost:8080`. URL отсюда.",
    )
    st.divider()
    st.markdown("### Модели")
    all_models = list(manifest["models"].keys())
    sel_models = st.multiselect("Прогнать", all_models, default=all_models)
    st.divider()
    st.markdown("### Регионы")
    all_regions = ["ccpd", "russian", "european", "openalpr", "generic"]
    sel_regions = st.multiselect("Прогнать", all_regions, default=all_regions)

if not tunnel_url:
    st.warning("Заполни tunnel URL в сайдбаре, чтобы начать.")
    st.stop()

# fetch sample ids
try:
    sample = fetch_sample_ids(tunnel_url, manifest)
except Exception as e:
    st.error(f"Не удалось скачать sample_image_ids.json: {e}")
    st.stop()

st.success(f"sample готов: " +
           " · ".join(f"{r}={len(ids)}" for r, ids in sample.items()))

run_all = st.button("▶ Прогнать всё выбранное", type="primary")
status = st.empty()
progress_bar = st.progress(0.0)

if run_all:
    tasks = [(m, r) for m in sel_models for r in sel_regions
             if r in sample and m in manifest["models"]]
    total = len(tasks)
    results = st.session_state.get("results", {})
    for i, (m, r) in enumerate(tasks):
        progress_bar.progress(i / total, text=f"[{i+1}/{total}] {m} / {r}")
        results.setdefault(m, {})[r] = measure_for(
            m, r, sample[r], tunnel_url, manifest,
            status_cb=lambda txt: status.text(txt),
        )
    progress_bar.progress(1.0, text="готово")
    status.text("всё прогнано")
    st.session_state["results"] = results

if "results" in st.session_state and st.session_state["results"]:
    st.divider()
    st.markdown("### Результаты")
    rows = []
    for m, by_r in st.session_state["results"].items():
        for r, payload in by_r.items():
            if "error" in payload:
                rows.append({"model": m, "region": r, "n": 0,
                             "p50_ms": None, "p95_ms": None,
                             "elapsed_s": None, "error": payload["error"]})
                continue
            rows.append({"model": m, "region": r, "n": payload["n"],
                         "p50_ms": round(payload["p50_ms"], 1),
                         "p95_ms": round(payload["p95_ms"], 1),
                         "elapsed_s": round(payload["elapsed_s"], 1),
                         "error": ""})
    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    zip_bytes = make_results_zip()
    st.download_button("⬇ Скачать все latency JSONL (zip)",
                       zip_bytes,
                       file_name="latency_jsonl.zip",
                       mime="application/zip")
    st.caption("Распакуй и положи каждый `<region>__latency.jsonl` в "
               "`streamlit_cloud_eval/data/predictions/<model>/` bootstrap-репы.")
