"""Streamlit UI: размытие и брендирование автомобильных номеров.

Полная сборка (6 моделей через табы) с режимом сравнения. Slim-вариант
для Streamlit Community Cloud — в каталоге streamlit_cloud/.

Запуск локально:
  streamlit run streamlit_app/app.py
"""
from __future__ import annotations

import io
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from inference import (
    Detection,
    TwoStagePipeline,
    SinglePosePipeline,
    BboxOnlyPipeline,
    ClassicalPipeline,
    RFDETRPipeline,
    blur_detections,
    paste_logo_with_homography,
    make_default_logo,
    draw_detections,
)
from prediction_log import (
    aggregate,
    file_hash,
    log_event,
    read_events,
    LOG_FILE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS = REPO_ROOT / "runs_from_cloud" / "runs"

# веса (path в repo runs_from_cloud + опциональный override в streamlit_app/models/)
WEIGHTS = {
    "yolo11n_detect":     "yolo11n_cuda_20260504_v2/weights/best.pt",         # v2 после fix-3 — 0.984 mean
    "yolo11n_detect_v1":  "yolo11n_cuda_20260503_185701/weights/best.pt",     # broken (post-mortem demo)
    "yolo12n_detect":     "yolo12n_cuda_20260504_130347/weights/best.pt",     # v2 после fix-3 — 0.983 mean
    "yolo12n_detect_v1":  "yolo12n_cuda_20260503_202008/weights/best.pt",     # broken (post-mortem demo)
    "yolo11n_pose":       "yolo11n_pose_cuda_v2/weights/best.pt",
    "kpt_head":           "keypoint_head_20260503_131404/best.pt",
    "rfdetr_medium":      "rfdetr_medium_20260503_132525/checkpoint_best_ema.pth",
    "rfdetr_nano":        "rfdetr_nano_20260504_155433/checkpoint_best_ema.pth",  # 0.986 mean — лидер
}


def resolve_weight(key: str) -> Path:
    """Сначала ищем в streamlit_app/models/<key>.{pt,pth} (для деплоя),
    потом в runs_from_cloud/."""
    local_dir = Path(__file__).resolve().parent / "models"
    for ext in (".pt", ".pth"):
        local = local_dir / f"{key}{ext}"
        if local.exists():
            return local
    return RUNS / WEIGHTS[key]


# ---------- зоопарк моделей ----------

# default = pose, потому что YOLO11n-detect и YOLO12n-detect фактически НЕ обучились
# (см. главу 2 §«POST-MORTEM detect-моделей»). Inference выдаёт мусор. Поэтому:
DEFAULT_MODEL = "YOLO11n-pose v2 (рекомендуется)"

MODEL_CHOICES = {
    DEFAULT_MODEL: dict(
        cls="pose",
        pose="yolo11n_pose",
        notes=(
            "Основной кандидат для промышленного применения. "
            "Одноступенчатая обработка: одна и та же сеть выдаёт и "
            "ограничивающую рамку, и четыре угла плашки. "
            "На русской выборке IoU не ниже 0,5 в 240 случаях из 240; "
            "средняя точность на пересечении 0,5 равна 0,99 на CCPD и 0,77 "
            "на русской подборке. Время обработки на центральном процессоре — "
            "около 47 миллисекунд, размер весов — 5,4 мегабайта. "
            "Преимущество одноступенчатой обработки в том, что между "
            "детектором рамки и оценщиком углов нет промежуточного звена, "
            "и ошибки не накапливаются."
        ),
    ),
    "YOLO11n-detect v2 + ResNet18": dict(
        cls="two_stage",
        detector="yolo11n_detect",
        kpt_head="kpt_head",
        notes=(
            "Двухступенчатая обработка: детектор YOLO11n строит рамку, "
            "ResNet-18 предсказывает четыре угла. Переобучена 4 мая 2026 "
            "после устранения ошибки в формировании обучающей выборки "
            "(см. главу 2, §POST-MORTEM). Средняя точность на пересечении "
            "0,5 — 0,984 (CCPD — 0,994; русская — 0,988; европейская — "
            "0,986; OpenALPR — 0,992; generic — 0,959). На русской выборке "
            "пересечение не ниже 0,5 во всех 240 фотографиях. "
            "Размер обоих весов — около 50 мегабайт, время обработки — "
            "около 55 миллисекунд. Самый компактный из двухступенчатых "
            "вариантов."
        ),
    ),
    "YOLO12n-detect v2 + ResNet18": dict(
        cls="two_stage",
        detector="yolo12n_detect",
        kpt_head="kpt_head",
        notes=(
            "Двухступенчатая обработка с детектором YOLO12n, в котором "
            "вместо свёрток используется механизм внимания. Переобучена "
            "4 мая 2026. Средняя точность на пересечении 0,5 — 0,983, "
            "то есть совпадает с YOLO11n в пределах статистической "
            "погрешности. На задаче поиска одного класса (автомобильного "
            "номера) выигрыша от внимания не наблюдается, при этом "
            "обработка занимает примерно вдвое больше времени. "
            "Оставлена для сравнения архитектурных решений."
        ),
    ),
    "RF-DETR Nano + ResNet18": dict(
        cls="rfdetr",
        ckpt="rfdetr_nano",
        size="nano",
        kpt_head="kpt_head",
        notes=(
            "Двухступенчатая обработка с трансформерным детектором RF-DETR "
            "Nano и оценщиком углов на основе ResNet-18. Самая высокая "
            "средняя точность из всех испытанных моделей — 0,986 на "
            "пересечении 0,5 (CCPD — 0,988; русская — 0,997; европейская — "
            "0,996; OpenALPR — 1,000; generic — 0,948). Внутренний "
            "извлекатель признаков DINOv2-small тот же, что и у Medium, "
            "но рабочее разрешение составляет 384 пикселя вместо 576, "
            "поэтому обработка примерно втрое быстрее. Время на "
            "центральном процессоре — около 290 миллисекунд, размер "
            "весов — 116 мегабайт."
        ),
    ),
    "RF-DETR Medium + ResNet18": dict(
        cls="rfdetr",
        ckpt="rfdetr_medium",
        size="medium",
        kpt_head="kpt_head",
        notes=(
            "Предшественник варианта Nano, обученный 3 мая 2026. Средняя "
            "точность на пересечении 0,5 — 0,984. Nano даёт прирост в "
            "0,002 при примерно втрое меньшем времени обработки, поэтому "
            "Medium оставлен исключительно для иллюстрации поведения сети "
            "при увеличении разрешения и числа слоёв декодера (576 вместо "
            "384 пикселей; четыре слоя вместо двух). Размер весов — "
            "134 мегабайта, время обработки на центральном процессоре — "
            "около 830 миллисекунд."
        ),
    ),
    "Контуры + морфология (без обучения)": dict(
        cls="classical",
        notes=(
            "Базовое правиловое решение без обучения. Использует фильтр "
            "Кэнни для выделения границ, морфологическое замыкание "
            "горизонтальным ядром 13 на 5 для соединения краёв символов "
            "в единый прямоугольник, аппроксимацию контура многоугольником "
            "и фильтрацию по соотношению сторон. На русской выборке "
            "F-мера составляет 0,13 — то есть верно находит примерно одну "
            "плашку из семи. Это и есть потолок методов без обучения на "
            "пользовательских фотографиях; если ничего не нашлось, это "
            "ожидаемое поведение (см. §1.2)."
        ),
    ),
}


@st.cache_resource(show_spinner="Загружаю модель ...")
def get_pipeline(model_key: str):
    cfg = MODEL_CHOICES[model_key]
    cls = cfg["cls"]
    if cls == "two_stage":
        return TwoStagePipeline(
            detector_path=resolve_weight(cfg["detector"]),
            kpt_head_path=resolve_weight(cfg["kpt_head"]),
            device="cpu",
        )
    if cls == "pose":
        return SinglePosePipeline(
            pose_path=resolve_weight(cfg["pose"]),
            device="cpu",
        )
    if cls == "bbox_only":
        return BboxOnlyPipeline(
            detector_path=resolve_weight(cfg["detector"]),
            device="cpu",
        )
    if cls == "classical":
        return ClassicalPipeline()
    if cls == "rfdetr":
        return RFDETRPipeline(
            ckpt_path=resolve_weight(cfg["ckpt"]),
            kpt_head_path=resolve_weight(cfg["kpt_head"]) if "kpt_head" in cfg else None,
            device="cpu",
            size=cfg.get("size", "nano"),
        )
    raise ValueError(f"unknown pipeline class: {cls}")


def auto_imgsz(W: int, H: int) -> int:
    """Подбирает разумный imgsz по размеру оригинала.

    Крупное фото → больший imgsz, иначе мелкие плашки потеряются. На
    маленьких — 640 быстрее без потери качества."""
    longest = max(W, H)
    if longest <= 800:
        return 640
    if longest <= 1500:
        return 1024
    if longest <= 2500:
        return 1280
    return 1600


# ---------- compute с кэшированием ----------
#
# Detection — это dataclass с np.ndarray внутри; пытаться сложить такое в
# `@st.cache_data` напрямую падает с UnserializableReturnValueError, потому
# что Streamlit pickle'ит результат до того как сохранить, а numpy в
# нашей форме у него почему-то не проходит через сериализацию. Поэтому
# в кэше живёт plain-list-of-dict, а сразу после возврата мы разворачиваем
# его обратно в Detection — это дёшево, np.array на 4×2 int.

def _serialize(dets: list[Detection]) -> list[dict]:
    return [
        {
            "bbox_xyxy": list(d.bbox_xyxy),
            "kpts": d.kpts.tolist(),
            "confidence": float(d.confidence),
        }
        for d in dets
    ]


def _deserialize(rows: list[dict]) -> list[Detection]:
    return [
        Detection(
            bbox_xyxy=tuple(r["bbox_xyxy"]),
            kpts=np.array(r["kpts"], dtype=np.float32),
            confidence=float(r["confidence"]),
        )
        for r in rows
    ]


@st.cache_data(show_spinner=False, max_entries=50)
def compute_detections_serialized(
    model_key: str, file_id: str, conf: float, imgsz: int, _img_np: np.ndarray
) -> tuple[list[dict], float]:
    """Кэшируемый прогон pipeline. Ключ кэша = (model_key, file_id, conf, imgsz);
    `_img_np` с подчёркиванием — Streamlit не хеширует, передаётся для расчёта.

    Возвращает уже сериализованный список и latency (мс)."""
    pipeline = get_pipeline(model_key)
    t0 = time.perf_counter()
    dets = pipeline(_img_np, conf=conf, imgsz=imgsz)
    latency = (time.perf_counter() - t0) * 1000
    return _serialize(dets), latency


def compute_detections(
    model_key: str, file_id: str, conf: float, imgsz: int, img_np: np.ndarray,
) -> tuple[list[Detection], float]:
    """Обёртка: кэш-сериализация → десериализация в Detection."""
    rows, latency = compute_detections_serialized(
        model_key, file_id, conf, imgsz, img_np,
    )
    return _deserialize(rows), latency


# ---------- UI ----------

def render_advanced_sidebar(W: int, H: int) -> dict:
    """Sidebar для отладочного режима. Selectbox модели убран — в этом режиме
    каждое фото прогоняется через ВСЕ модели сразу, переключение между ними
    делается через табы прямо над картинкой (мгновенно, ничего не молотит
    повторно благодаря кэшу `compute_detections`)."""
    st.header("Отладка")
    st.caption(
        "Каждое фото будет прогнано через все 5 моделей. "
        "Переключение между ними — табы под картинкой."
    )

    auto = auto_imgsz(W, H)
    imgsz = st.select_slider(
        "Размер инференса (imgsz)",
        options=[416, 640, 960, 1024, 1280, 1600, 1920],
        value=auto,
        help=(
            f"Авто: {auto} (по размеру фото {W}×{H}). "
            "На больших фото с мелкими плашками — поднимай."
        ),
    )

    confidence = st.slider(
        "Порог уверенности детектора", 0.05, 0.95, 0.30, step=0.05,
        help="Чем ниже — тем больше «слабых» детекций возьмём в обработку.",
    )

    debug = st.checkbox(
        "Debug-overlay (рамки и точки поверх результата)", value=False,
    )

    return dict(
        model_key=DEFAULT_MODEL,    # для zip-выгрузки берём именно production-кандидата
        imgsz=imgsz,
        confidence=confidence,
        debug=debug,
        run_all_models=True,
    )


def render_simple_sidebar() -> dict:
    """Sidebar для простого режима — только режим обработки и логотип."""
    return dict(
        model_key=DEFAULT_MODEL,
        imgsz=None,           # будет выставлено по auto_imgsz позже
        confidence=0.30,
        debug=False,
        run_all_models=False,
    )


def render_processing_settings(advanced: bool) -> dict:
    """Параметры режима обработки (общие для simple/advanced)."""
    st.header("Что делать с номером")
    mode = st.radio(
        "Режим:",
        ("Размытие (Gaussian blur)", "Логотип через гомографию"),
        index=1,
        label_visibility="collapsed" if not advanced else "visible",
    )

    blur_strength = None
    logo = None

    if mode.startswith("Размытие"):
        if advanced:
            blur_strength = st.slider(
                "Сила размытия (kernel size)", 5, 99, 35, step=2,
            )
        else:
            blur_strength = 35
    else:
        logo_file = st.file_uploader(
            "Логотип (PNG / JPG)",
            type=["png", "jpg", "jpeg"],
        )
        if logo_file is not None:
            logo = np.array(Image.open(logo_file).convert("RGBA"))
        else:
            st.caption("Логотип не загружен — будет использован стандартный.")
            logo = make_default_logo("СКРЫТО")

    return dict(mode=mode, blur_strength=blur_strength, logo=logo)


def render_feedback_buttons(file_id: str, model_key: str) -> None:
    """Две кнопки «хорошо / плохо» под результатом. Голос сохраняется в
    session_state (чтобы юзер не голосовал дважды) и в JSONL-лог."""
    st.session_state.setdefault("feedback_cache", {})
    cache_key = f"{file_id}::{model_key}"
    voted = st.session_state.feedback_cache.get(cache_key)

    # ключ должен быть уникален в рамках страницы — иначе DuplicateWidgetID
    # при batch'е (несколько кнопок на странице).
    key_safe = cache_key.replace(" ", "_").replace(",", "_").replace("(", "").replace(")", "")
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button(
        "Хорошо", disabled=voted is not None, use_container_width=True,
        key=f"fb_good_{key_safe}",
    ):
        log_event({
            "type": "feedback", "file_hash": file_id,
            "model": model_key, "verdict": "good",
        })
        st.session_state.feedback_cache[cache_key] = "good"
        st.rerun()
    if c2.button(
        "Плохо", disabled=voted is not None, use_container_width=True,
        key=f"fb_bad_{key_safe}",
    ):
        log_event({
            "type": "feedback", "file_hash": file_id,
            "model": model_key, "verdict": "bad",
        })
        st.session_state.feedback_cache[cache_key] = "bad"
        st.rerun()
    if voted == "good":
        c3.success("Спасибо! Записал «норм».")
    elif voted == "bad":
        c3.warning("Спасибо! Записал «не норм».")


def render_log_panel() -> None:
    """Свёрнутый блок со статистикой логов — в advanced-режиме."""
    if not LOG_FILE.exists():
        st.caption("Лог пока пустой.")
        return
    events = read_events()
    agg = aggregate(events)

    cols = st.columns(4)
    cols[0].metric("Всего обработок", agg["total_inferences"])
    cols[1].metric("Голосов", agg["total_feedback"])
    if agg["detection_rate"] is not None:
        cols[2].metric("Hit-rate", f"{agg['detection_rate']:.0%}")
    if agg["avg_latency_ms"] is not None:
        cols[3].metric("Сред. latency", f"{agg['avg_latency_ms']:.0f} ms")

    st.markdown("**Голоса по моделям:**")
    fb_summary = []
    for model, c in agg["feedback_by_model"].items():
        good = c.get("good", 0)
        bad = c.get("bad", 0)
        total = good + bad
        rate = f"{good / total:.0%}" if total else "—"
        fb_summary.append(
            {"model": model, "хорошо": good, "плохо": bad, "good rate": rate}
        )
    if fb_summary:
        st.dataframe(fb_summary, hide_index=True, use_container_width=True)
    else:
        st.caption("Голосов пока нет.")

    with st.expander(f"Последние 20 событий из `{LOG_FILE.name}`"):
        st.json(events[-20:])


# ---------- processing one file ----------

@dataclass
class ProcessedItem:
    """Результат обработки одной фотографии — для общего zip и сводки."""
    original_name: str
    file_id: str
    detections: list[Detection]
    latency_ms: float
    processed: np.ndarray   # uint8 HxWx3 — финальная картинка с обработкой


def _run_and_render_for_model(
    model_key: str, uploaded, img: Image.Image, img_np: np.ndarray,
    fid: str, W: int, H: int, imgsz: int,
    settings: dict, proc_settings: dict, advanced: bool,
) -> tuple[list[Detection], float, np.ndarray]:
    """Прогон одной модели на одном фото и полный рендер карточки.

    Раскладка карточки сверху вниз:
        1) описание модели (notes);
        2) две колонки «исходное» и «обработанное»;
        3) кнопка скачивания и блок отзыва;
        4) разбивка времени, общая статистика, таблица найденных номеров.

    Возвращает (detections, latency_ms_total, processed_img)."""
    # 1) Описание модели — сверху, до фотографий, чтобы видно было сразу.
    st.caption(MODEL_CHOICES[model_key]["notes"])

    # 2) Инференс и обработка.
    detections, t_infer_ms = compute_detections(
        model_key, fid, settings["confidence"], imgsz, img_np,
    )

    # лог только при первом расчёте этой комбинации
    inf_state_key = (
        f"__logged_{fid}_{model_key}_{settings['confidence']}_{imgsz}"
    )
    if not st.session_state.get(inf_state_key):
        log_event({
            "type": "inference",
            "file_hash": fid,
            "file_name": uploaded.name,
            "model": model_key,
            "imgsz": imgsz,
            "conf": settings["confidence"],
            "image_size": [W, H],
            "n_detections": len(detections),
            "avg_confidence": (
                float(np.mean([d.confidence for d in detections])) if detections else None
            ),
            "latency_ms": round(t_infer_ms, 1),
        })
        st.session_state[inf_state_key] = True

    # Наложение (отдельно меряем — чтобы видеть, сколько съедают
    # размытие или перспективная гомография).
    t1 = time.perf_counter()
    if not detections:
        processed = img_np
    else:
        if proc_settings["mode"].startswith("Размытие"):
            processed = blur_detections(
                img_np, detections, kernel=proc_settings["blur_strength"],
            )
        else:
            processed = paste_logo_with_homography(
                img_np, detections, proc_settings["logo"],
            )
        if settings["debug"]:
            processed = draw_detections(
                processed, detections, show_bbox=True, show_kpts=True,
            )
    t_postproc_ms = (time.perf_counter() - t1) * 1000
    latency_total_ms = float(t_infer_ms) + t_postproc_ms

    # 3) Фотографии в две колонки.
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Исходное**")
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("**Обработанное**")
        if not detections:
            st.warning(
                "Номера не обнаружены. Можно понизить порог уверенности, "
                "увеличить размер инференса или попробовать другую модель."
            )
            st.image(img, use_container_width=True)
        else:
            st.image(processed, use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(processed).save(buf, format="PNG")
            st.download_button(
                "Скачать",
                data=buf.getvalue(),
                file_name=f"{Path(uploaded.name).stem}_{_safe_model_slug(model_key)}.png",
                mime="image/png",
                key=f"dl_{fid}_{_safe_model_slug(model_key)}",
            )

    # 4) Отзыв.
    if detections:
        st.markdown("Как результат?")
        render_feedback_buttons(fid, model_key)

    # 5) Разбивка времени и сводные показатели.
    cols = st.columns(4)
    cols[0].metric("Инференс", f"{t_infer_ms:.0f} мс")
    cols[1].metric("Наложение", f"{t_postproc_ms:.0f} мс")
    cols[2].metric("Итого", f"{latency_total_ms:.0f} мс")
    cols[3].metric("Размер инференса", str(imgsz))

    cols2 = st.columns(3)
    cols2[0].metric("Найдено номеров", len(detections))
    cols2[1].metric(
        "Средняя уверенность",
        f"{np.mean([d.confidence for d in detections]):.0%}" if detections else "—",
    )
    cols2[2].metric("Размер фото", f"{W}×{H}")

    # 6) Таблица найденных плашек: уверенность, рамка, четыре угла.
    if detections:
        rows = []
        for i, d in enumerate(detections, start=1):
            x1, y1, x2, y2 = d.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1
            rows.append(
                {
                    "№": i,
                    "Уверенность": f"{d.confidence:.0%}",
                    "Рамка, пикс.": f"{x1},{y1} → {x2},{y2}",
                    "Доля кадра, %": f"{100*bw/W:.1f} × {100*bh/H:.1f}",
                    "Углы (TL · TR · BR · BL)": " | ".join(
                        f"{int(x)},{int(y)}" for x, y in d.kpts
                    ),
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)

    return detections, latency_total_ms, processed


def _safe_model_slug(model_key: str) -> str:
    """Понятное короткое имя модели для file-name'ов и widget key'ев."""
    return (
        model_key.split("(")[0].strip()
        .replace(" ", "_").replace("+", "plus").replace("/", "_")
    )


def _short_model_label(model_key: str, max_len: int = 28) -> str:
    """Сокращённое имя для tab-заголовка (table-style)."""
    head = model_key.split("(")[0].strip()
    if len(head) > max_len:
        head = head[: max_len - 1] + "…"
    return head


def process_one(
    uploaded, settings: dict, proc_settings: dict, advanced: bool,
    position: int | None,
) -> ProcessedItem | None:
    """Обработать одну фотографию.

    Simple-режим (advanced=False) — одна production-модель, плоский layout.
    Advanced/отладочный (advanced=True, run_all_models=True) — табы по всем
    моделям; primary модель (DEFAULT_MODEL) идёт в zip-выгрузку для batch'а."""
    file_bytes = uploaded.getvalue()
    fid = file_hash(file_bytes)

    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_np = np.array(img)
    H, W = img_np.shape[:2]
    st.session_state["__last_image_size"] = (W, H)

    imgsz = settings["imgsz"] if settings["imgsz"] is not None else auto_imgsz(W, H)

    # какие модели прогоняем
    if settings.get("run_all_models"):
        models_to_run = list(MODEL_CHOICES.keys())
    else:
        models_to_run = [settings["model_key"]]

    primary_model = DEFAULT_MODEL if DEFAULT_MODEL in models_to_run else models_to_run[0]
    primary_result: tuple[list[Detection], float, np.ndarray] | None = None

    # batch: оборачиваем в expander; single — рендерим напрямую
    if position is not None:
        title = f"#{position} — {uploaded.name}"
        container = st.expander(title, expanded=(position == 1))
    else:
        container = st.container()

    with container:
        if len(models_to_run) == 1:
            primary_result = _run_and_render_for_model(
                models_to_run[0], uploaded, img, img_np, fid, W, H, imgsz,
                settings, proc_settings, advanced,
            )
        else:
            # таб на каждую модель — переключение мгновенное, кэш делает своё дело
            tab_labels = [_short_model_label(m) for m in models_to_run]
            tabs = st.tabs(tab_labels)
            for tab, model in zip(tabs, models_to_run):
                with tab:
                    res = _run_and_render_for_model(
                        model, uploaded, img, img_np, fid, W, H, imgsz,
                        settings, proc_settings, advanced,
                    )
                    if model == primary_model:
                        primary_result = res

    if primary_result is None:
        return None
    detections, latency_ms, processed = primary_result
    return ProcessedItem(
        original_name=uploaded.name,
        file_id=fid,
        detections=detections,
        latency_ms=latency_ms,
        processed=processed,
    )


# ---------- main ----------

def main():
    st.set_page_config(
        page_title="platebuster — полная сборка",
        page_icon=None,
        layout="wide",
    )
    # Шапка в стиле облачной сборки: «platebuster» + миниатюрная плашка
    # номера, у которой правый край растворяется. Здесь же — пометка
    # «полная сборка с шестью моделями» вместо bullet-каптюна.
    plate_svg = (
        "<svg width='130' height='38' viewBox='0 0 130 38' "
        "style='display:block;'>"
        "<defs>"
        " <linearGradient id='pbFade2' x1='0' y1='0' x2='1' y2='0'>"
        "  <stop offset='0'    stop-color='#ffffff' stop-opacity='1'/>"
        "  <stop offset='0.15' stop-color='#ffffff' stop-opacity='1'/>"
        "  <stop offset='0.85' stop-color='#ffffff' stop-opacity='0'/>"
        "  <stop offset='1'    stop-color='#ffffff' stop-opacity='0'/>"
        " </linearGradient>"
        " <mask id='pbMask2'>"
        "  <rect width='130' height='38' fill='url(#pbFade2)'/>"
        " </mask>"
        "</defs>"
        "<g mask='url(#pbMask2)'>"
        " <rect x='1.5' y='3.5' width='126' height='30' rx='4' "
        "ry='4' fill='#ffffff' stroke='#222222' stroke-width='1.5'/>"
        " <text x='12' y='25' "
        "font-family='Menlo, Consolas, ui-monospace, monospace' "
        "font-size='17' font-weight='700' letter-spacing='1' "
        "fill='#1B1B1B'>А 123 ВС</text>"
        "</g>"
        "<animate xlink:href='#pbFade2' attributeName='x2' "
        "values='1;0.96;1' dur='2.6s' repeatCount='indefinite'/>"
        "</svg>"
    )
    st.markdown(
        "<div style='display:flex;align-items:center;gap:14px;"
        "margin-bottom:4px;'>"
        "<div style='font-size:34px;font-weight:700;"
        "letter-spacing:-0.5px;line-height:1;'>platebuster</div>"
        f"{plate_svg}"
        "<div style='color:#888;font-size:13px;margin-left:4px;'>"
        "полная сборка · 6 моделей сразу через табы</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Скрывает автомобильные номера на фотографиях. "
        "Каждое фото прогоняется через все шесть моделей — переключение "
        "между ними мгновенное благодаря кэшу инференса. "
        "Все диагностические метрики, разбивка времени, EXIF и таблица "
        "детекций — всегда видны (research-режим, не публичная демо)."
    )

    # CSS-обёртки из облачной сборки: горизонтальный ряд кнопок в карточке,
    # иконки тултипа. Применяются глобально, безопасно.
    st.markdown(
        "<style>"
        "div[data-testid='stHorizontalBlock']"
        ":has(> div[data-testid='stColumn']:nth-child(3):last-child)"
        "{flex-wrap:nowrap !important;gap:6px !important;}"
        "div[data-testid='stHorizontalBlock']"
        ":has(> div[data-testid='stColumn']:nth-child(3):last-child) "
        "div[data-testid='stColumn']"
        "{min-width:0 !important;flex:1 1 0 !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    # ---------- sidebar ----------
    # Полная сборка живёт ТОЛЬКО в продвинутом режиме: вся диагностика
    # видна всегда, ничего не прячется. Тумблер переключения убран.
    advanced = True
    with st.sidebar:
        proc_settings = render_processing_settings(advanced)

        st.divider()
        # Узнаем размер фото для auto_imgsz, если оно уже загружено.
        uploaded_state = st.session_state.get("__last_image_size", (640, 640))
        settings = render_advanced_sidebar(*uploaded_state)


    # ---------- main: загрузка фото ----------
    uploaded_files = st.file_uploader(
        "Загрузите фото автомобилей (JPG / PNG, можно несколько)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info(
            "Перетащите файл (или несколько) в зону выше. "
            "Все будут обработаны одной кнопкой."
        )
        st.divider()
        if advanced:
            with st.expander("Лог предсказаний"):
                render_log_panel()
        return

    # все обработанные результаты (для общего zip и сводки)
    batch_results: list[ProcessedItem] = []

    if len(uploaded_files) == 1:
        item = process_one(
            uploaded_files[0], settings, proc_settings, advanced, position=None,
        )
        if item is not None:
            batch_results.append(item)
    else:
        st.subheader(f"Пакетная обработка: {len(uploaded_files)} фото")
        progress = st.progress(0.0, text="Готовлю pipeline ...")
        for i, uploaded in enumerate(uploaded_files, start=1):
            progress.progress(
                (i - 1) / len(uploaded_files),
                text=f"Обрабатываю {i}/{len(uploaded_files)}: {uploaded.name}",
            )
            item = process_one(
                uploaded, settings, proc_settings, advanced, position=i,
            )
            if item is not None:
                batch_results.append(item)
        progress.progress(1.0, text=f"Готово: {len(batch_results)} / {len(uploaded_files)}")

    # ---------- сводка по пачке + общий zip ----------
    if batch_results:
        st.divider()
        n_total = len(batch_results)
        n_with = sum(1 for it in batch_results if it.detections)
        avg_lat = float(np.mean([it.latency_ms for it in batch_results]))
        cols = st.columns(4)
        cols[0].metric("Фото обработано", n_total)
        cols[1].metric(
            "Из них с номерами",
            f"{n_with} ({n_with/n_total:.0%})" if n_total else "—",
        )
        cols[2].metric("Среднее время", f"{avg_lat:.0f} мс")
        cols[3].metric(
            "Найдено номеров",
            sum(len(it.detections) for it in batch_results),
        )

        # Гистограмма уверенности — полезна, когда модель уже прогнала
        # хотя бы пять фотографий с найденными плашками. Десять корзин
        # от 0 до 1, чтобы было видно, где сосредоточены детекции.
        all_confs = [d.confidence for it in batch_results for d in it.detections]
        if n_with >= 5 and all_confs:
            st.markdown("**Распределение уверенности**")
            counts, edges = np.histogram(all_confs, bins=10, range=(0.0, 1.0))
            chart_data = {
                f"{edges[i]:.1f}–{edges[i+1]:.1f}": int(counts[i])
                for i in range(len(counts))
            }
            st.bar_chart(chart_data)

        # Архив всех обработанных результатов.
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for it in batch_results:
                png = io.BytesIO()
                Image.fromarray(it.processed).save(png, format="PNG")
                stem = Path(it.original_name).stem
                zf.writestr(f"{stem}_processed.png", png.getvalue())
        st.download_button(
            f"Скачать все результаты одним архивом ({n_total} файлов)",
            data=zip_buf.getvalue(),
            file_name="processed_plates.zip",
            mime="application/zip",
            type="primary",
        )

    # ---------- лог-панель в advanced ----------
    if advanced:
        st.divider()
        with st.expander("Лог предсказаний", expanded=False):
            render_log_panel()

    st.divider()
    st.caption(
        "platebuster — open-source детектор автомобильных номеров. "
        "Backbone production-кандидата: YOLO11n + ResNet18 (2-stage). "
        "Исходный код и обоснование выбора моделей — в репозитории."
    )


if __name__ == "__main__":
    main()
