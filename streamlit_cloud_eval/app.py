"""
Streamlit-приложение: бутстрэп-ДИ и парные тесты значимости для шести моделей.

Развёртывается в Streamlit Community Cloud как отдельный сервис под бутстрэп.
Predictions JSONL получены заранее (в окружении эмуляции 1 vCPU) и лежат в
data/predictions/<model>/<region>.jsonl.

Запуск локально:
    streamlit run streamlit_cloud_eval/app.py

Деплой:
    1. Положить data/predictions/<model>/<region>.jsonl
    2. Запушить в репу, указать app.py точкой входа
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap as bs  # noqa: E402

# ---------------------------------------------------------------------------
# конфигурация
# ---------------------------------------------------------------------------

st.set_page_config(page_title="platebuster — bootstrap CI",
                   page_icon="📐", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PRED_DIR = APP_DIR / "data" / "predictions"

DEFAULT_PAIRS = [
    "yolo11n,yolo12n",
    "rfdetr_nano,rfdetr_medium",
    "yolo11n_pose,keypoint_head",
]


# ---------------------------------------------------------------------------
# кеши
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_and_precompute(model: str, region: str) -> tuple[list, int, int]:
    """Загрузка JSONL + per-image precompute + опциональный мерж latency.
    Кешируется по (model, region). Возвращает (precompute, n_records, n_latency)."""
    p = PRED_DIR / model / f"{region}.jsonl"
    records = bs.load_jsonl(p)
    pre = bs.precompute_records(records)
    n_lat = bs.attach_latencies(pre, PRED_DIR / model / f"{region}__latency.jsonl")
    return pre, len(records), n_lat


# ---------------------------------------------------------------------------
# вспомогалки рендера
# ---------------------------------------------------------------------------


def fmt_ci(point: float, lo: float, hi: float, fmt: str = "{:.4f}") -> str:
    if not np.isfinite(point):
        return "—"
    return f"{fmt.format(point)} [{fmt.format(lo)}; {fmt.format(hi)}]"


def fmt_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "—"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def verdict_for_pair(diff_point: float, lo: float, hi: float) -> str:
    if not np.isfinite(lo):
        return "—"
    if lo > 0:
        return "A > B (значимо)"
    if hi < 0:
        return "A < B (значимо)"
    return "ДИ накрывает 0 (различия незначимы)"


# ---------------------------------------------------------------------------
# сайдбар
# ---------------------------------------------------------------------------


with st.sidebar:
    st.markdown("### Источник predictions")
    st.code(str(PRED_DIR.relative_to(APP_DIR)), language="text")
    models_available = bs.list_available_models(PRED_DIR)
    if not models_available:
        st.error(
            "В `data/predictions/` ничего не найдено. Положи JSONL-файлы по "
            "схеме `data/predictions/<model>/<region>.jsonl` и обнови страницу."
        )
        st.stop()
    st.write(f"Найдено моделей: **{len(models_available)}**")
    st.write(f"Регионы: **{', '.join(bs.REGIONS)}**")
    st.divider()
    st.markdown("### Параметры бутстрэпа")
    B = st.slider("B — число итераций", 200, 5000, 2000, step=200,
                  help="2000 — стандарт для 95 % CI. Меньше = быстрее.")
    seed = st.number_input("seed", value=42, min_value=0, max_value=10000,
                           step=1)
    alpha = st.selectbox("уровень значимости α", [0.01, 0.05, 0.10], index=1,
                         format_func=lambda x: f"α = {x} ({int((1-x)*100)} % CI)")


# ---------------------------------------------------------------------------
# заголовок
# ---------------------------------------------------------------------------


st.title("📐 Бутстрэп-ДИ и парные тесты значимости")
st.caption(
    "Этот сервис считает доверительные интервалы для метрик детектора и "
    "keypoint-головы по сохранённым per-image предсказаниям. Парные тесты "
    "проверяют значимость разности метрик двух моделей на одной и той же "
    "ресэмплированной подвыборке картинок."
)

tab_single, tab_paired, tab_export = st.tabs(
    ["По моделям", "Парные сравнения", "Экспорт таблиц"])


# ---------------------------------------------------------------------------
# Tab 1 — Single bootstrap
# ---------------------------------------------------------------------------

with tab_single:
    col1, col2 = st.columns([2, 3])
    with col1:
        sel_models = st.multiselect("Модели", models_available,
                                    default=models_available)
        sel_regions = st.multiselect("Регионы", bs.REGIONS,
                                     default=bs.REGIONS)
    with col2:
        st.info(
            "**Что считается:**\n"
            "- mAP@50 и mAP@50:95 (одноклассовая задача, COCO-style 101-point)\n"
            "- mean keypoint error в пикселях и PCK@0.05 (для моделей с углами)\n\n"
            "**ДИ:** 2.5/97.5-перцентильный по B ресэмплам картинок с возвращением."
        )

    run_single = st.button("▶ Прогнать", type="primary", key="run_single")

    if run_single:
        results = {}
        total = len(sel_models) * len(sel_regions)
        done = 0
        progress = st.progress(0.0, text="precompute + bootstrap...")
        for m in sel_models:
            results[m] = {}
            for r in sel_regions:
                jsonl = PRED_DIR / m / f"{r}.jsonl"
                if not jsonl.exists():
                    results[m][r] = None
                    done += 1
                    progress.progress(done / total, text=f"{m} / {r}: нет файла")
                    continue
                t0 = time.time()
                pre, n_records, n_lat = load_and_precompute(m, r)
                progress.progress(done / total, text=f"{m} / {r}: бутстрэп B={B}")
                res = bs.single_bootstrap(pre, B=int(B), seed=int(seed),
                                          alpha=float(alpha))
                results[m][r] = {"res": res, "elapsed": time.time() - t0}
                done += 1
                progress.progress(done / total,
                                  text=f"{m} / {r}: готово ({results[m][r]['elapsed']:.1f}s)")
        progress.empty()
        st.session_state["single_results"] = results
        st.success(f"Готово: {total} ячеек")

    if "single_results" in st.session_state:
        rows = []
        for m, by_region in st.session_state["single_results"].items():
            for r, payload in by_region.items():
                if payload is None:
                    rows.append({"model": m, "region": r, "n": "—",
                                 "mAP@50": "—", "mAP@50:95": "—",
                                 "kpt err px": "—", "PCK@0.05": "—"})
                    continue
                res = payload["res"]
                rows.append({
                    "model": m,
                    "region": r,
                    "n": res["_n_images"],
                    "mAP@50": fmt_ci(res["mAP50"]["point"],
                                     res["mAP50"]["ci_low"],
                                     res["mAP50"]["ci_high"]),
                    "mAP@50:95": fmt_ci(res["mAP50_95"]["point"],
                                        res["mAP50_95"]["ci_low"],
                                        res["mAP50_95"]["ci_high"]),
                    "kpt err px": fmt_ci(res["mean_kpt_err_px"]["point"],
                                         res["mean_kpt_err_px"]["ci_low"],
                                         res["mean_kpt_err_px"]["ci_high"],
                                         fmt="{:.2f}"),
                    "PCK@0.05": fmt_ci(res["PCK_05"]["point"],
                                       res["PCK_05"]["ci_low"],
                                       res["PCK_05"]["ci_high"]),
                    "p50 latency ms": fmt_ci(res["latency_p50_ms"]["point"],
                                             res["latency_p50_ms"]["ci_low"],
                                             res["latency_p50_ms"]["ci_high"],
                                             fmt="{:.1f}"),
                    "p95 latency ms": fmt_ci(res["latency_p95_ms"]["point"],
                                             res["latency_p95_ms"]["ci_low"],
                                             res["latency_p95_ms"]["ci_high"],
                                             fmt="{:.1f}"),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)


# ---------------------------------------------------------------------------
# Tab 2 — Paired tests
# ---------------------------------------------------------------------------

with tab_paired:
    st.markdown("Парный бутстрэп: на каждой итерации ресэмпляются одни и те же "
                "image_id из обеих моделей. ДИ для разности `metric_A − metric_B`.")

    pair_input = st.text_area(
        "Пары моделей (по одной в строке, формат `A,B`)",
        "\n".join(DEFAULT_PAIRS), height=120,
    )
    pairs = [p.strip() for p in pair_input.splitlines() if "," in p]
    sel_regions_p = st.multiselect("Регионы", bs.REGIONS, default=bs.REGIONS,
                                    key="paired_regions")

    run_paired = st.button("▶ Прогнать парные", type="primary", key="run_paired")

    if run_paired:
        results = {}
        total = len(pairs) * len(sel_regions_p)
        done = 0
        progress = st.progress(0.0, text="paired bootstrap...")
        for pair in pairs:
            a, b = [s.strip() for s in pair.split(",")]
            results[pair] = {}
            for r in sel_regions_p:
                ja = PRED_DIR / a / f"{r}.jsonl"
                jb = PRED_DIR / b / f"{r}.jsonl"
                if not ja.exists() or not jb.exists():
                    results[pair][r] = None
                    done += 1
                    progress.progress(done / total,
                                      text=f"{pair}/{r}: нет файлов")
                    continue
                t0 = time.time()
                pa, _, _ = load_and_precompute(a, r)
                pb, _, _ = load_and_precompute(b, r)
                progress.progress(done / total,
                                  text=f"{pair}/{r}: B={B}")
                res = bs.paired_bootstrap(pa, pb, B=int(B),
                                          seed=int(seed),
                                          alpha=float(alpha))
                results[pair][r] = {"res": res, "elapsed": time.time() - t0}
                done += 1
                progress.progress(done / total,
                                  text=f"{pair}/{r}: готово")
        progress.empty()
        st.session_state["paired_results"] = results
        st.success(f"Готово: {total} ячеек")

    if "paired_results" in st.session_state:
        rows = []
        for pair, by_region in st.session_state["paired_results"].items():
            for r, payload in by_region.items():
                if payload is None or "error" in payload["res"]:
                    rows.append({"pair": pair, "region": r,
                                 "Δ mAP@50": "—", "p-value": "—",
                                 "вердикт": "—"})
                    continue
                res = payload["res"]
                d = res["diff_mAP50"]
                rows.append({
                    "pair": pair,
                    "region": r,
                    "n": res["_n_common"],
                    "Δ mAP@50": fmt_ci(d["point"], d["ci_low"], d["ci_high"]),
                    "p-value": fmt_pvalue(d["p_value"]),
                    "вердикт": verdict_for_pair(d["point"], d["ci_low"], d["ci_high"]),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)


# ---------------------------------------------------------------------------
# Tab 3 — Export
# ---------------------------------------------------------------------------

with tab_export:
    st.markdown("Готовые таблицы в Markdown для вставки в работу.")
    md = io.StringIO()
    md.write("# Таблицы со статистической строгостью\n\n")
    md.write(f"Сгенерировано: B = {int(B)}, α = {float(alpha)}, "
             f"метод = percentile bootstrap.\n\n")

    if "single_results" in st.session_state:
        md.write("## Сводная таблица (замена таблицы 13)\n\n")
        md.write("| Модель | Регион | n | mAP@50 [95 % ДИ] | mAP@50:95 [95 % ДИ] | "
                 "kpt err px [ДИ] | PCK@0.05 [ДИ] | "
                 "p50 latency ms [ДИ] | p95 latency ms [ДИ] |\n")
        md.write("|---|---|---|---|---|---|---|---|---|\n")
        for m, by_region in st.session_state["single_results"].items():
            for r, payload in by_region.items():
                if payload is None:
                    continue
                res = payload["res"]
                md.write(f"| {m} | {r} | {res['_n_images']} | "
                         f"{fmt_ci(res['mAP50']['point'], res['mAP50']['ci_low'], res['mAP50']['ci_high'])} | "
                         f"{fmt_ci(res['mAP50_95']['point'], res['mAP50_95']['ci_low'], res['mAP50_95']['ci_high'])} | "
                         f"{fmt_ci(res['mean_kpt_err_px']['point'], res['mean_kpt_err_px']['ci_low'], res['mean_kpt_err_px']['ci_high'], fmt='{:.2f}')} | "
                         f"{fmt_ci(res['PCK_05']['point'], res['PCK_05']['ci_low'], res['PCK_05']['ci_high'])} | "
                         f"{fmt_ci(res['latency_p50_ms']['point'], res['latency_p50_ms']['ci_low'], res['latency_p50_ms']['ci_high'], fmt='{:.1f}')} | "
                         f"{fmt_ci(res['latency_p95_ms']['point'], res['latency_p95_ms']['ci_low'], res['latency_p95_ms']['ci_high'], fmt='{:.1f}')} |\n")
        md.write("\n")
    else:
        md.write("_Tab «По моделям» ещё не запускался._\n\n")

    if "paired_results" in st.session_state:
        md.write("## Парные тесты значимости\n\n")
        md.write("| Пара (A vs B) | Регион | n | Δ mAP@50 [95 % ДИ] | p-value | Вердикт |\n")
        md.write("|---|---|---|---|---|---|\n")
        for pair, by_region in st.session_state["paired_results"].items():
            for r, payload in by_region.items():
                if payload is None or "error" in payload["res"]:
                    continue
                res = payload["res"]
                d = res["diff_mAP50"]
                md.write(f"| {pair} | {r} | {res['_n_common']} | "
                         f"{fmt_ci(d['point'], d['ci_low'], d['ci_high'])} | "
                         f"{fmt_pvalue(d['p_value'])} | "
                         f"{verdict_for_pair(d['point'], d['ci_low'], d['ci_high'])} |\n")
        md.write("\n")
    else:
        md.write("_Tab «Парные сравнения» ещё не запускался._\n\n")

    content = md.getvalue()
    st.code(content, language="markdown")
    st.download_button("⬇ Скачать stat_tables.md", content,
                       file_name="stat_tables.md", mime="text/markdown")
