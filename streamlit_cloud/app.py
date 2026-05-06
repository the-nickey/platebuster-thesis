"""platebuster — облачная версия.

Streamlit Community Cloud вариант. Одна модель, одно фото за раз
(до десяти за загрузку), карточки идут одна под другой.

Запуск локально:
    .venv/bin/python -m streamlit run streamlit_cloud/app.py
"""
from __future__ import annotations

import base64
import io
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import ExifTags, Image, ImageOps

# Поддержка HEIC/HEIF, формата фото с iPhone. После регистрации
# `pillow_heif` Pillow читает их как обычные форматы — никакой
# дополнительной конвертации в коде не нужно. Если по какой-то
# причине пакет не установлен (локальная сборка без iOS-теста),
# просто игнорируем — Streamlit отфильтрует расширение.
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from inference import (
    Detection,
    SinglePosePipeline,
    auto_imgsz,
    blur_detections,
    draw_detections,
    make_default_logo,
    paste_logo_with_homography,
)

# Полная версия со всеми шестью моделями и режимом сравнения — в основной
# папке репозитория (`streamlit_app/`). Когда у проекта будет публичный URL,
# подменим эту константу на ссылку.
REPO_LINK_PLACEHOLDER = "ссылка появится после публикации репозитория"

# Force-light CSS: Streamlit на клиенте по умолчанию читает системную
# `prefers-color-scheme`, и при системной тёмной перебивает наш
# `theme.base = light` из config.toml. Этот блок применяется, когда
# тумблер «Тёмная тема» в сайдбаре выключен, и принудительно красит
# поверхности в белый — тогда не важно, что видит браузер по системе.
_LIGHT_THEME_CSS = """
<style>
:root { color-scheme: light; }
.stApp, [data-testid="stMain"], [data-testid="stHeader"]
{ background-color: #FFFFFF !important; color: #1B1B1B !important; }
[data-testid="stSidebar"]
{ background-color: #F4F4F4 !important; }
[data-testid="stSidebar"] *,
[data-testid="stMain"] *
{ color: #1B1B1B; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p,
.stApp label, .stApp .stMarkdown, .stApp .stMarkdown p
{ color: #1B1B1B !important; }
.stApp .stCaption, .stApp [data-testid="stCaptionContainer"]
{ color: #666666 !important; }
hr { border-color: #ECECEC !important; }

/* шапка Streamlit и кнопка-бургер «>>» / «<<» */
[data-testid="stHeader"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stBaseButton-header"] svg,
[data-testid="stBaseButton-headerNoPadding"] svg
{ color: #1B1B1B !important; fill: #1B1B1B !important; }

/* file_uploader */
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"]
{ background-color: #FAFAFA !important; border-color: #DDDDDD !important; }
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFileName"]
{ color: #1B1B1B !important; }
[data-testid="stFileUploaderFile"]
{ background-color: #F4F4F4 !important;
  border: 1px solid #DDDDDD !important; }
[data-testid="stFileUploader"] small
{ color: #666666 !important; }
[data-testid="stFileUploaderDeleteBtn"] svg
{ color: #1B1B1B !important; fill: #1B1B1B !important; }

/* кнопки */
button, .stDownloadButton button, .stButton button
{ background-color: #FFFFFF !important; color: #1B1B1B !important;
  border: 1px solid #DDDDDD !important; }
button p, button span, button div
{ color: inherit !important; }
button:hover, .stDownloadButton button:hover, .stButton button:hover
{ background-color: #F4F4F4 !important; border-color: #BBBBBB !important; }
button[kind="primary"], .stDownloadButton button[kind="primary"]
{ background-color: #1B1B1B !important; color: #FFFFFF !important;
  border: 1px solid #1B1B1B !important; }
button[kind="primary"] p, button[kind="primary"] span,
button[kind="primary"] div
{ color: #FFFFFF !important; }
button[data-testid="stBaseButton-tertiary"]
{ background-color: transparent !important;
  color: #444444 !important; border: none !important; }

/* радио / чекбоксы / тумблеры */
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label
{ color: #1B1B1B !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child
{ background-color: #DDDDDD !important; border: 1px solid #BBBBBB !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child
{ background-color: #1B1B1B !important; border-color: #1B1B1B !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child > div
{ background-color: #FFFFFF !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child > div
{ background-color: #FFFFFF !important; }

/* иконки тултипа `?` */
[data-testid="stTooltipHoverTarget"],
[data-testid="stTooltipHoverTarget"] svg,
.stTooltipIcon, .stTooltipIcon svg
{ color: #888888 !important; fill: #888888 !important; }

/* алерты */
[data-testid="stAlert"],
[data-testid="stAlertContainer"],
div[role="alert"]
{ background-color: #EAF1FB !important; color: #1B1B1B !important;
  border: 1px solid #C8D9F0 !important; }
[data-testid="stAlert"] *, div[role="alert"] *
{ color: #1B1B1B !important; }

/* модалка «Что это?» */
div[role="dialog"]
{ background-color: #FFFFFF !important; color: #1B1B1B !important; }
div[role="dialog"] *
{ color: #1B1B1B !important; }

/* плашка-лейбл аплоадера */
.pb-upload-label, .pb-upload-label *
{ color: #1B1B1B !important; }
</style>
"""

_DARK_THEME_CSS = """
<style>
.stApp, [data-testid="stMain"], [data-testid="stHeader"]
{ background-color: #131313 !important; color: #ECECEC !important; }
[data-testid="stSidebar"]
{ background-color: #1A1A1A !important; }
[data-testid="stSidebar"] *,
[data-testid="stMain"] *
{ color: #ECECEC; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp p,
.stApp label, .stApp .stMarkdown, .stApp .stMarkdown p
{ color: #ECECEC !important; }
.stApp .stCaption, .stApp [data-testid="stCaptionContainer"]
{ color: #9A9A9A !important; }
hr { border-color: #2A2A2A !important; }

/* шапка Streamlit и кнопка-бургер «>>» / «<<» — на тёмном фоне
   стандартный значок не виден, форсим белый */
[data-testid="stHeader"] svg,
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stBaseButton-header"] svg,
[data-testid="stBaseButton-headerNoPadding"] svg
{ color: #ECECEC !important; fill: #ECECEC !important; }

/* file_uploader: dropzone, список файлов и подписи */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *
{ color: #ECECEC !important; }
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"]
{ background-color: #1F1F1F !important; border-color: #333 !important; }
[data-testid="stFileUploaderFile"]
{ background-color: #2A2A2A !important;
  border: 1px solid #3A3A3A !important; }
[data-testid="stFileUploaderFile"] *,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] p,
[data-testid="stFileUploaderFile"] div
{ color: #ECECEC !important; }
/* размер файла и подпись «50MB per file» — приглушённый, но видимый */
[data-testid="stFileUploader"] small
{ color: #BFBFBF !important; }
/* X-кнопка удаления у файла */
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderDeleteBtn"] svg,
[data-testid="stFileUploaderDeleteBtn"] button
{ color: #ECECEC !important; fill: #ECECEC !important; }

/* кнопки */
button, .stDownloadButton button, .stButton button
{ background-color: #222 !important; color: #ECECEC !important;
  border: 1px solid #3A3A3A !important; }
button p, button span, button div
{ color: inherit !important; }
button:hover, .stDownloadButton button:hover, .stButton button:hover
{ background-color: #2C2C2C !important; border-color: #555 !important; }
/* primary — белая на тёмном; форсим тёмный текст внутри `<p>`-обёртки */
button[kind="primary"], .stDownloadButton button[kind="primary"]
{ background-color: #ECECEC !important; color: #131313 !important;
  border: 1px solid #ECECEC !important; }
button[kind="primary"] p, button[kind="primary"] span,
button[kind="primary"] div
{ color: #131313 !important; }
button[data-testid="stBaseButton-tertiary"]
{ background-color: transparent !important;
  color: #BFBFBF !important; border: none !important; }
button[data-testid="stBaseButton-tertiary"] p
{ color: #BFBFBF !important; }

/* радио / чекбоксы / тумблеры
   В Streamlit `st.toggle` рендерится тем же `data-testid="stCheckbox"`,
   что и обычный чекбокс — отличается только дизайном внутри. */
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label
{ color: #ECECEC !important; }
/* трек тумблера: первый div внутри label[data-baseweb="checkbox"] */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child
{ background-color: #2A2A2A !important; border: 1px solid #555 !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child
{ background-color: #ECECEC !important; border-color: #ECECEC !important; }
/* «thumb» внутри тумблера — внутренняя пилюля */
[data-testid="stCheckbox"] label[data-baseweb="checkbox"] > div:first-child > div
{ background-color: #ECECEC !important; }
[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input[aria-checked="true"]) > div:first-child > div
{ background-color: #131313 !important; }

/* иконки тултипа `?` рядом с лейблами */
[data-testid="stTooltipHoverTarget"],
[data-testid="stTooltipHoverTarget"] svg,
[data-testid="stTooltipIcon"],
.stTooltipIcon, .stTooltipIcon svg
{ color: #9A9A9A !important; fill: #9A9A9A !important; }
[data-baseweb="tooltip"]
{ background-color: #2A2A2A !important; color: #ECECEC !important; }

/* алерты */
[data-testid="stAlert"],
[data-testid="stAlertContainer"],
[data-testid="stAlertContentInfo"],
[data-testid="stAlertContentWarning"],
[data-testid="stAlertContentError"],
[data-testid="stAlertContentSuccess"],
div[role="alert"]
{ background-color: #1B2A3A !important; color: #ECECEC !important;
  border: 1px solid #2A3A4A !important; }
[data-testid="stAlert"] *, div[role="alert"] *
{ color: #ECECEC !important; }

/* модалка «Что это?» */
div[role="dialog"]
{ background-color: #1A1A1A !important; color: #ECECEC !important; }
div[role="dialog"] *
{ color: #ECECEC !important; }

/* тосты */
[data-testid="stToast"], [data-testid="stToastContainer"]
{ background-color: #2A2A2A !important; color: #ECECEC !important;
  border: 1px solid #3A3A3A !important; }
[data-testid="stToast"] *
{ color: #ECECEC !important; }

/* плашка-лейбл аплоадера */
.pb-upload-label, .pb-upload-label *
{ color: #ECECEC !important; }
</style>
"""

ABOUT_TEXT = (
    "Это сервис цифровой гигиены при продаже авто. "
    "А ещё он делает за вас рутину: загружаете фото машины — "
    "приложение находит плашку и закрывает её размытием или вашим "
    "логотипом. Логотип ложится с учётом перспективы "
    "(по четырём углам), а не как плоский прямоугольник.\n\n"
    "Под капотом — нейросеть YOLO11n-pose, обученная на 60 тысячах "
    "фотографий из открытых датасетов и собственной разметки. Размер модели — "
    "5,4 мегабайта, время обработки одной фотографии на сервере — около "
    "50 миллисекунд.\n\n"
    f"Полная версия со всеми шестью моделями и режимом сравнения — "
    f"в репозитории ({REPO_LINK_PLACEHOLDER}).\n\n"
    "Этот сервис сделал П. В. Очкин в рамках магистерской работы — УрФУ, 2026."
)


# --------------------------------------------------------------- настройки
HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "models" / "yolo11n-pose-v2.pt"

MAX_PHOTOS = 10
DEFAULT_CONFIDENCE = 0.30

# JSONL-журнал: одна строка на событие. Используется для счётчика
# «всего за всё время». В Streamlit Community Cloud мап /mount/src/...
# доступен на запись, но при перезапуске контейнера может быть очищен —
# это нормально для бесплатного тарифа, статистика просто стартует
# заново. Папка создаётся при первом запуске.
LOG_FILE = HERE / "logs" / "cloud.jsonl"

# ---- кастомный hold-to-zoom компонент --------------------------------------
# `components.html` не реагирует на `streamlit:setFrameHeight`, поэтому даём
# собственную папочку index.html через declare_component — там JS сам шлёт
# фактическую высоту контейнера в Streamlit. ResizeObserver внутри iframe
# держит её актуальной при ресайзе окна.
_zoom_component = components.declare_component(
    "platebuster_zoom",
    path=str(HERE / "_zoom_component"),
)

# ---- кастомный «до/после» компонент ----------------------------------------
# `streamlit_image_comparison` использует JuxtaposeJS внутри iframe с фикси-
# рованной высотой, из-за чего на узких экранах под карточкой остаётся
# огромный пустой хвост. Делаем свой простой compare-слайдер с динамической
# высотой через тот же setFrameHeight-протокол, что и у zoom-компонента.
_compare_component = components.declare_component(
    "platebuster_compare",
    path=str(HERE / "_compare_component"),
)


# --------------------------------------------------------------- модель
@st.cache_resource(show_spinner="Приложение поднимается, не переключайтесь.")
def get_pipeline() -> SinglePosePipeline:
    return SinglePosePipeline(pose_path=WEIGHTS, device="cpu")


# --------------------------------------------------------------- кэш инференса
@dataclass
class InferenceResult:
    detections: list[Detection]
    t_preproc_ms: float
    t_infer_ms: float
    t_postproc_ms: float = 0.0
    img_rgb: np.ndarray | None = None
    imgsz: int = 640


@st.cache_data(show_spinner=False, max_entries=64)
def _cached_detect(
    file_id: str, rotation: int, conf: float,
    imgsz_override: int | None, _img_np: np.ndarray,
) -> tuple[list[dict], float, int]:
    """Кэш самого инференса. Ключ — file_id + поворот + порог + imgsz.

    `_img_np` со стартовым подчёркиванием — Streamlit его не хеширует."""
    pipeline = get_pipeline()
    H, W = _img_np.shape[:2]
    imgsz = imgsz_override if imgsz_override else auto_imgsz(W, H)

    t0 = time.perf_counter()
    dets = pipeline(_img_np, conf=conf, imgsz=imgsz)
    infer_ms = (time.perf_counter() - t0) * 1000

    rows = [
        {
            "bbox_xyxy": list(d.bbox_xyxy),
            "kpts": d.kpts.tolist(),
            "confidence": float(d.confidence),
        }
        for d in dets
    ]
    return rows, infer_ms, imgsz


def detect(
    file_id: str, rotation: int, conf: float,
    img_np: np.ndarray, imgsz_override: int | None = None,
):
    rows, infer_ms, imgsz = _cached_detect(
        file_id, rotation, conf, imgsz_override, img_np,
    )
    dets = [
        Detection(
            bbox_xyxy=tuple(r["bbox_xyxy"]),
            kpts=np.array(r["kpts"], dtype=np.float32),
            confidence=float(r["confidence"]),
        )
        for r in rows
    ]
    return dets, infer_ms, imgsz


# --------------------------------------------------------------- утилиты
def file_id_from_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()[:16]


def read_exif_orientation(pil_img: Image.Image) -> int | None:
    """Достаёт тег ориентации из EXIF до автоповорота. Возвращает 1..8 или None."""
    tag = next((k for k, v in ExifTags.TAGS.items() if v == "Orientation"), None)
    if tag is None:
        return None
    try:
        exif = pil_img.getexif()
    except Exception:
        return None
    if not exif:
        return None
    val = exif.get(tag)
    if val in (1, 2, 3, 4, 5, 6, 7, 8):
        return val
    return None


_ORIENT_HUMAN = {
    1: "без поворота",
    3: "повёрнуто на 180°",
    6: "повёрнуто на 90° по часовой",
    8: "повёрнуто на 90° против часовой",
    2: "зеркально по горизонтали",
    4: "зеркально по вертикали",
    5: "зеркально с поворотом",
    7: "зеркально с поворотом",
}


def human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} Б"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.0f} КБ"
    return f"{num_bytes / 1024 / 1024:.1f} МБ"


def rotate_array_90(img_np: np.ndarray, manual_rotation: int) -> np.ndarray:
    """manual_rotation в градусах (0/90/180/270) — поворот по часовой стрелке."""
    k = (manual_rotation // 90) % 4
    if k == 0:
        return img_np
    # np.rot90 крутит против часовой; делаем k шагов против → даёт по часовой при -k
    return np.rot90(img_np, k=-k).copy()


def _np_to_data_url(img_np: np.ndarray, max_side: int = 1600) -> str:
    pil = Image.fromarray(img_np)
    if max(pil.size) > max_side:
        pil = pil.copy()
        pil.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def render_zoom_image(img_np: np.ndarray, fid: str) -> None:
    """Картинка с приближением «по нажатию» — пятикратный зум.

    Render-стороной занимается кастомный компонент `_zoom_component`:
    он сам сообщает Streamlit фактическую высоту через `setFrameHeight`,
    так что под фотографией нет пустого «хвоста»."""
    src = _np_to_data_url(img_np)
    _zoom_component(src=src, zoom=5.0, key=f"zoom_{fid}")


def render_compare_image(
    before_np: np.ndarray, after_np: np.ndarray, fid: str,
) -> None:
    """Сравнение «до/после» через ползунок. Свой компонент с динамической
    высотой — без пустот ниже карточки."""
    src_b = _np_to_data_url(before_np)
    src_a = _np_to_data_url(after_np)
    _compare_component(
        src_before=src_b,
        src_after=src_a,
        label_before="до",
        label_after="после",
        key=f"cmp_{fid}",
    )


# --------------------------------------------------------------- session counters
def init_state() -> None:
    st.session_state.setdefault("photos_processed", 0)
    st.session_state.setdefault("plates_found", 0)
    st.session_state.setdefault("manual_rotation", {})    # file_id -> deg
    st.session_state.setdefault("counted_files", set())   # file_id'ы, что уже учли
    st.session_state.setdefault("dismissed", set())        # file_id'ы, которые скрыли
    st.session_state.setdefault("toast_queue", [])         # отложенные тосты
    # стартовая тема: ?theme=dark в URL включает тёмную сразу;
    # иначе светлая. Тумблер в сайдбаре переопределяет.
    if "dark_theme" not in st.session_state:
        try:
            st.session_state["dark_theme"] = (
                st.query_params.get("theme") == "dark"
            )
        except Exception:
            st.session_state["dark_theme"] = False
    # session_id + событие открытия сессии (логируется один раз).
    if "session_id" not in st.session_state:
        import uuid
        st.session_state["session_id"] = uuid.uuid4().hex[:12]
        log_event({"type": "session_open"})


# ---- JSONL-журнал и агрегатор ----------------------------------------------

def log_event(payload: dict) -> None:
    """Кладёт событие в JSONL. Не падает, если файл недоступен — это
    статистика, а не критичные данные."""
    import time
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": int(time.time()),
            "session_id": st.session_state.get("session_id"),
            **payload,
        }
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


@st.cache_data(ttl=15, show_spinner=False)
def aggregate_log(_marker: float) -> dict:
    """Считает «всего сессий / всего фото / всего номеров» по JSONL.

    `_marker` — параметр для инвалидации кэша (mtime файла). Кэшируется
    на 15 секунд, чтобы не парсить файл при каждом rerun."""
    if not LOG_FILE.exists():
        return {"sessions": 0, "photos": 0, "plates": 0}
    sessions: set = set()
    photo_keys: set = set()
    plates = 0
    try:
        with LOG_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                sid = e.get("session_id")
                if sid:
                    sessions.add(sid)
                if e.get("type") == "photo_processed":
                    fid = e.get("file_id")
                    if sid and fid:
                        photo_keys.add((sid, fid))
                    plates += int(e.get("n_plates") or 0)
    except Exception:
        pass
    return {
        "sessions": len(sessions),
        "photos": len(photo_keys),
        "plates": plates,
    }


def get_global_stats() -> dict:
    marker = LOG_FILE.stat().st_mtime if LOG_FILE.exists() else 0.0
    return aggregate_log(marker)


def plural_ru(n: int, forms: tuple[str, str, str]) -> str:
    """Возвращает нужную форму существительного для числительного.

    forms = (1 шт., 2-4 шт., 5+ шт.). Учитывает 11-14 → 5+."""
    n_abs = abs(int(n))
    last_two = n_abs % 100
    last = n_abs % 10
    if 11 <= last_two <= 14:
        return forms[2]
    if last == 1:
        return forms[0]
    if 2 <= last <= 4:
        return forms[1]
    return forms[2]


# ---- очередь тостов ---------------------------------------------------------
# `st.rerun()` прерывает текущий рендер до того, как Streamlit успевает
# показать тост. Поэтому копим сообщения в session_state и проигрываем их
# в самом начале следующего рендера.

def queue_toast(msg: str, icon: str | None = None) -> None:
    q = st.session_state.setdefault("toast_queue", [])
    q.append({"msg": msg, "icon": icon})


def flush_toasts() -> None:
    q = st.session_state.get("toast_queue") or []
    for item in q:
        if item.get("icon"):
            st.toast(item["msg"], icon=item["icon"])
        else:
            st.toast(item["msg"])
    st.session_state["toast_queue"] = []


# --------------------------------------------------------------- бургер-меню
IMAGE_ACTION_ZOOM = "Приближение"
IMAGE_ACTION_COMPARE = "Сравнение «до и после»"

IMGSZ_OPTIONS: list = ["Авто", 416, 640, 960, 1024, 1280, 1600, 1920]


@dataclass
class Settings:
    mode: str = "Закрыть логотипом"   # или "Размыть"
    blur_strength: int = 35
    logo_rgba: np.ndarray | None = None
    image_action: str = IMAGE_ACTION_ZOOM   # что делать по нажатию на фото
    debug: bool = False
    imgsz_override: int | None = None       # явный размер инференса; None = авто
    confidence: float = 0.30                # порог уверенности детектора
    dark_theme: bool = False                # тёмная тема UI


def render_sidebar() -> Settings:
    with st.sidebar:
        st.markdown("### Что делать с номером")
        mode = st.radio(
            "Режим",
            ("Размыть", "Закрыть логотипом"),
            index=1,
            label_visibility="collapsed",
        )

        blur_strength = 35
        logo_rgba = None
        if mode == "Размыть":
            blur_strength = st.slider("Сила размытия", 5, 99, 35, step=2)
        else:
            logo_file = st.file_uploader(
                "Свой логотип (PNG или JPG)",
                type=["png", "jpg", "jpeg"],
                label_visibility="visible",
            )
            if logo_file is not None:
                logo_rgba = np.array(Image.open(logo_file).convert("RGBA"))
            else:
                st.caption("Если не загрузить — поставим стандартную плашку.")
                logo_rgba = make_default_logo()

        st.markdown("---")
        st.markdown("### Дополнительно")

        image_action = st.radio(
            "При нажатии на фото",
            (IMAGE_ACTION_ZOOM, IMAGE_ACTION_COMPARE),
            index=0,
            help=(
                "«Приближение» увеличивает точку под пальцем или курсором. "
                "«Сравнение „до и после“» прячет под результат ползунок: "
                "тянешь — видишь оригинал."
            ),
        )

        dark_theme = st.toggle(
            "Тёмная тема",
            value=st.session_state.get("dark_theme", False),
            help="Чёрный фон вместо белого — приятнее вечером.",
        )
        st.session_state["dark_theme"] = dark_theme

        debug = st.toggle(
            "Режим отладки", value=False,
            help=(
                "Показывает разбивку времени, размеры файлов, ориентацию EXIF, "
                "таблицу детекций и гистограмму уверенности."
            ),
        )

        # Размер инференса и порог уверенности. Настраиваем только в
        # отладке, но значения запоминаем и применяем и без отладки.
        if debug:
            current = st.session_state.get("imgsz_choice", "Авто")
            if current not in IMGSZ_OPTIONS:
                current = "Авто"
            sel = st.select_slider(
                "Размер инференса",
                options=IMGSZ_OPTIONS,
                value=current,
                help=(
                    "Чем больше число, тем мельче плашки замечает модель — "
                    "и тем дольше обработка. «Авто» подбирает под фото сам."
                ),
            )
            st.session_state["imgsz_choice"] = sel

            conf_current = float(st.session_state.get("conf_choice", 0.30))
            conf_sel = st.slider(
                "Порог уверенности",
                min_value=0.05, max_value=0.95,
                value=conf_current, step=0.05,
                help=(
                    "Минимальная уверенность детектора, при которой плашка "
                    "считается найденной. Ниже — больше «слабых» детекций "
                    "и риск ложных срабатываний; выше — пропустим часть "
                    "номеров под углом или в плохом свете."
                ),
            )
            st.session_state["conf_choice"] = conf_sel

        sel_saved = st.session_state.get("imgsz_choice", "Авто")
        imgsz_override = None if sel_saved == "Авто" else int(sel_saved)
        confidence = float(st.session_state.get("conf_choice", 0.30))

        st.markdown("---")
        st.caption(
            "Это публичная демо-версия. Полная сборка со всеми шестью "
            f"моделями и режимом сравнения — в репозитории "
            f"({REPO_LINK_PLACEHOLDER})."
        )

    return Settings(
        mode=mode,
        blur_strength=blur_strength,
        logo_rgba=logo_rgba,
        image_action=image_action,
        debug=debug,
        imgsz_override=imgsz_override,
        confidence=confidence,
        dark_theme=dark_theme,
    )


# --------------------------------------------------------------- модалка «Что это?»
@st.dialog("Что это?", width="large")
def _about_dialog() -> None:
    """Полноэкранная модалка с описанием сервиса.

    Размер делается через CSS-override (см. начало `main`); сюда просто
    кладём текст и кнопку закрытия. Тапа по затемнению снаружи — тоже
    закрывает (`dismissible=True` по умолчанию)."""
    st.markdown(ABOUT_TEXT)
    if st.button(
        "Понятно!",
        key="about_close",
        type="primary",
        use_container_width=True,
    ):
        st.rerun()


# --------------------------------------------------------------- шапка
def render_header() -> None:
    title_col, count_col = st.columns([3, 1])
    with title_col:
        # Бренд-блок: слово «platebuster» + миниатюрная плашка номера, которая
        # растворяется в правую сторону через linear-gradient mask.
        # Цифры/буквы внутри слегка дрожат через SVG-анимацию `dur=2.4s`,
        # что усиливает ощущение «исчезновения в воздухе».
        plate_svg = (
            "<svg width='130' height='38' viewBox='0 0 130 38' "
            "style='display:block;'>"
            "<defs>"
            # Левые 15% — чётко, потом равномерное линейное затухание,
            # к 85% уже всё растворено.
            " <linearGradient id='pbFade' x1='0' y1='0' x2='1' y2='0'>"
            "  <stop offset='0'    stop-color='#ffffff' stop-opacity='1'/>"
            "  <stop offset='0.15' stop-color='#ffffff' stop-opacity='1'/>"
            "  <stop offset='0.85' stop-color='#ffffff' stop-opacity='0'/>"
            "  <stop offset='1'    stop-color='#ffffff' stop-opacity='0'/>"
            " </linearGradient>"
            " <mask id='pbMask'>"
            "  <rect width='130' height='38' fill='url(#pbFade)'/>"
            " </mask>"
            "</defs>"
            "<g mask='url(#pbMask)'>"
            " <rect x='1.5' y='3.5' width='126' height='30' rx='4' "
            "ry='4' fill='#ffffff' stroke='#222222' stroke-width='1.5'/>"
            " <text x='12' y='25' "
            "font-family='Menlo, Consolas, ui-monospace, monospace' "
            "font-size='17' font-weight='700' letter-spacing='1' "
            "fill='#1B1B1B'>А 123 ВС</text>"
            "</g>"
            # лёгкое мерцание правого хвоста — будто цифры рассыпаются
            "<animate xlink:href='#pbFade' attributeName='x2' "
            "values='1;0.96;1' dur='2.6s' repeatCount='indefinite'/>"
            "</svg>"
        )
        st.markdown(
            "<div style='display:flex;align-items:center;gap:14px;"
            "margin-bottom:4px;'>"
            "<div style='font-size:34px;font-weight:700;"
            "letter-spacing:-0.5px;line-height:1;'>platebuster</div>"
            f"{plate_svg}"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='color:#666;margin-top:-4px;margin-bottom:8px;'>"
            "Скрывает автомобильные номера на фото — размытием или логотипом."
            "</div>",
            unsafe_allow_html=True,
        )
        # Текстовая мини-кнопка под подзаголовком открывает модалку
        # `_about_dialog` с описанием. Tap снаружи или по «Я понял» закроет.
        if st.button(
            "Что это?",
            key="about_btn",
            type="tertiary",
            help="Что делает приложение и кто его сделал.",
        ):
            _about_dialog()
    with count_col:
        n_photos = st.session_state.get("photos_processed", 0)
        n_plates = st.session_state.get("plates_found", 0)
        global_stats = get_global_stats()
        plates_word = plural_ru(n_plates, ("номер", "номера", "номеров"))
        sessions_word = plural_ru(
            global_stats["sessions"], ("сессию", "сессии", "сессий"),
        )
        st.markdown(
            "<div style='text-align:right;color:#666;font-size:13px;"
            "padding-top:22px;line-height:1.5;'>"
            f"За сессию: {n_photos} фото · {n_plates} {plates_word}"
            "<br/>"
            f"Всего: {global_stats['photos']} фото "
            f"за {global_stats['sessions']} {sessions_word}"
            "</div>",
            unsafe_allow_html=True,
        )


# --------------------------------------------------------------- одна карточка
@dataclass
class CardOutcome:
    file_id: str
    name: str
    detections: list[Detection]
    processed: np.ndarray
    original: np.ndarray
    file_bytes: int
    image_size_wh: tuple[int, int]
    fmt: str
    exif_orient: int | None
    manual_rotation: int
    timings_ms: dict
    imgsz: int


def process_card(
    uploaded, settings: Settings, position: int, total: int,
) -> CardOutcome | None:
    """Обработка одного файла + рендер карточки. Карточка идёт в потоке страницы."""
    file_bytes = uploaded.getvalue()
    fid = file_id_from_bytes(file_bytes)
    name = uploaded.name

    # пользователь скрыл это фото кнопкой «Заменить» — пропускаем
    if fid in st.session_state.get("dismissed", set()):
        return None

    # ---------- preprocessing: чтение, EXIF, поворот, конвертация ----------
    t0 = time.perf_counter()
    pil = Image.open(io.BytesIO(file_bytes))
    fmt = (pil.format or "JPEG").upper()
    exif_orient = read_exif_orientation(pil)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    img_np = np.array(pil)

    manual_rot = st.session_state["manual_rotation"].get(fid, 0)
    if manual_rot:
        img_np = rotate_array_90(img_np, manual_rot)
    H, W = img_np.shape[:2]
    t_preproc_ms = (time.perf_counter() - t0) * 1000

    # ---------- инференс ----------
    detections, t_infer_ms, imgsz = detect(
        fid, manual_rot, settings.confidence, img_np,
        imgsz_override=settings.imgsz_override,
    )

    # ---------- наложение ----------
    t1 = time.perf_counter()
    if not detections:
        processed = img_np
    elif settings.mode == "Размыть":
        processed = blur_detections(img_np, detections, kernel=settings.blur_strength)
    else:
        processed = paste_logo_with_homography(img_np, detections, settings.logo_rgba)

    if settings.debug and detections:
        processed = draw_detections(processed, detections, show_bbox=True, show_kpts=True)
    t_postproc_ms = (time.perf_counter() - t1) * 1000

    # ---------- учёт сессии ----------
    if fid not in st.session_state["counted_files"]:
        st.session_state["counted_files"].add(fid)
        st.session_state["photos_processed"] += 1
        st.session_state["plates_found"] += len(detections)
        # одна запись на (сессия, фото) — для глобальной статистики
        log_event({
            "type": "photo_processed",
            "file_id": fid,
            "n_plates": len(detections),
            "image_size": [W, H],
            "imgsz": imgsz,
        })

    # ---------- рендер карточки ----------
    timings = {
        "preproc": t_preproc_ms,
        "infer": t_infer_ms,
        "postproc": t_postproc_ms,
    }

    _render_card_body(
        fid=fid, name=name, position=position, total=total,
        original=img_np, processed=processed, detections=detections,
        settings=settings, file_bytes_len=len(file_bytes),
        image_size=(W, H), fmt=fmt, exif_orient=exif_orient,
        manual_rotation=manual_rot, timings_ms=timings, imgsz=imgsz,
    )

    return CardOutcome(
        file_id=fid, name=name, detections=detections,
        processed=processed, original=img_np,
        file_bytes=len(file_bytes), image_size_wh=(W, H), fmt=fmt,
        exif_orient=exif_orient, manual_rotation=manual_rot,
        timings_ms=timings, imgsz=imgsz,
    )


def _render_card_body(
    *, fid, name, position, total,
    original, processed, detections,
    settings: Settings,
    file_bytes_len, image_size, fmt, exif_orient,
    manual_rotation, timings_ms, imgsz,
):
    W, H = image_size

    # Заголовок «N из M — имя файла» — это диагностический шум; в обычном
    # режиме его не показываем. Тонкий разделитель между карточками поверх
    # CSS-маржинов даёт самодостаточный rhythm.
    if settings.debug:
        st.markdown(
            f"<div style='font-weight:600;font-size:18px;margin-top:20px;"
            f"margin-bottom:6px;'>{position} из {total} — {name}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style='border-top:1px solid #ECECEC;margin:18px 0 12px 0;'></div>",
            unsafe_allow_html=True,
        )

    # Картинка. Поведение управляется радио в сайдбаре:
    #   «Приближение»     — один кадр, по нажатию точка увеличивается;
    #   «Сравнение „до и после“» — ползунок поверх двух кадров.
    use_compare = settings.image_action == IMAGE_ACTION_COMPARE and detections
    if use_compare:
        render_compare_image(original, processed, fid)
    else:
        result = processed if detections else original
        render_zoom_image(result, fid)

    # Три действия. На мобильном Streamlit укладывает столбцы один под другой —
    # каждая кнопка получит всю ширину, как и хотел пользователь.
    a1, a2, a3 = st.columns(3)
    buf = io.BytesIO()
    Image.fromarray(processed).save(buf, format="PNG")
    a1.download_button(
        "Скачать",
        data=buf.getvalue(),
        file_name=f"{Path(name).stem}_platebuster.png",
        mime="image/png",
        key=f"dl_btn_{fid}",
        use_container_width=True,
    )
    if a2.button("Заменить", key=f"rep_{fid}", use_container_width=True,
                 help="Скрыть это фото из текущей сессии."):
        st.session_state["dismissed"].add(fid)
        queue_toast("Фото скрыто из списка")
        st.rerun()
    if a3.button("Повернуть", key=f"rot_{fid}", use_container_width=True,
                 help="Повернуть фото на 90° по часовой стрелке."):
        cur = st.session_state["manual_rotation"].get(fid, 0)
        st.session_state["manual_rotation"][fid] = (cur + 90) % 360
        queue_toast("Поворот применён")
        st.rerun()

    if not detections:
        st.info(
            "Номера не нашлись. Возможные причины: фото слишком тёмное, "
            "плашка очень маленькая или сильно повёрнута. Попробуйте кнопку "
            "«Повернуть на 90°» или загрузите другое фото."
        )

    if settings.debug:
        # Диагностическая строка и подробный блок.
        avg_conf = (
            f"{np.mean([d.confidence for d in detections]):.0%}"
            if detections else "—"
        )
        total_ms = sum(timings_ms.values())
        summary = (
            f"Найдено номеров: **{len(detections)}** · "
            f"уверенность: **{avg_conf}** · "
            f"размер фото: **{W}×{H}** · "
            f"размер файла: **{human_size(file_bytes_len)}** · "
            f"формат: **{fmt}** · "
            f"время: **{total_ms:.0f} мс**"
        )
        st.markdown(summary)
        _render_debug_block(
            detections=detections,
            timings_ms=timings_ms,
            imgsz=imgsz,
            image_size=image_size,
            exif_orient=exif_orient,
            manual_rotation=manual_rotation,
        )


def _render_two_columns(original, processed, detections):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Исходное**")
        st.image(original, use_container_width=True)
    with c2:
        st.markdown("**Обработанное**")
        if detections:
            st.image(processed, use_container_width=True)
        else:
            st.image(original, use_container_width=True)


def _render_debug_block(
    *, detections, timings_ms, imgsz, image_size, exif_orient, manual_rotation,
):
    W, H = image_size
    st.markdown("---")
    st.markdown("**Отладка**")

    cols = st.columns(4)
    cols[0].metric("Подготовка", f"{timings_ms['preproc']:.0f} мс")
    cols[1].metric("Инференс", f"{timings_ms['infer']:.0f} мс")
    cols[2].metric("Наложение", f"{timings_ms['postproc']:.0f} мс")
    cols[3].metric("Итого", f"{sum(timings_ms.values()):.0f} мс")

    info_lines = [
        f"Размер инференса (imgsz): **{imgsz}**",
        f"Размер фото: **{W}×{H}**",
    ]
    if exif_orient is not None:
        info_lines.append(
            f"EXIF ориентация: **{exif_orient}** "
            f"({_ORIENT_HUMAN.get(exif_orient, 'нестандарт')}, выровняли автоматически)"
        )
    if manual_rotation:
        info_lines.append(f"Ручной поворот: **{manual_rotation}°**")
    st.markdown(" · ".join(info_lines))

    if detections:
        rows = []
        for i, d in enumerate(detections, start=1):
            x1, y1, x2, y2 = d.bbox_xyxy
            bw, bh = x2 - x1, y2 - y1
            rows.append(
                {
                    "№": i,
                    "Уверенность": f"{d.confidence:.0%}",
                    "bbox, пикс.": f"{x1},{y1} → {x2},{y2}",
                    "bbox, % кадра": f"{100*bw/W:.1f}×{100*bh/H:.1f}",
                    "Углы (TL, TR, BR, BL)": " | ".join(
                        f"{int(x)},{int(y)}" for x, y in d.kpts
                    ),
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)


# --------------------------------------------------------------- main
def main() -> None:
    st.set_page_config(
        page_title="platebuster",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    init_state()
    # Проигрываем тосты, отложенные предыдущим рендером (см. queue_toast).
    flush_toasts()

    # На мобильном Streamlit штатно укладывает столбцы один под другой,
    # а у карточек хочется три кнопки в ряд при любой ширине. Целимся CSS-ом
    # в блоки `stHorizontalBlock`, у которых ровно три столбца — это и есть
    # наши тройки «Скачать / Заменить / Повернуть».
    #
    # Заодно расширяем модалку «Что это?» до 86 % ширины окна и почти всей
    # высоты — у Streamlit нативно она крошечная, читать неудобно.
    st.markdown(
        "<style>"
        # три кнопки в ряд при любой ширине
        "div[data-testid='stHorizontalBlock']"
        ":has(> div[data-testid='stColumn']:nth-child(3):last-child)"
        "{flex-wrap:nowrap !important;gap:6px !important;}"
        "div[data-testid='stHorizontalBlock']"
        ":has(> div[data-testid='stColumn']:nth-child(3):last-child) "
        "div[data-testid='stColumn']"
        "{min-width:0 !important;flex:1 1 0 !important;}"
        "div[data-testid='stHorizontalBlock']"
        ":has(> div[data-testid='stColumn']:nth-child(3):last-child) button"
        "{padding-left:8px !important;padding-right:8px !important;"
        "white-space:nowrap;font-size:13px;}"
        # текстовая кнопка «Что это?» — подчёркнута, чтобы было видно
        # что это ссылка-триггер, а не плоский подзаголовок
        "button[data-testid='stBaseButton-tertiary']"
        "{text-decoration:underline !important;"
        "text-underline-offset:3px !important;"
        "text-decoration-thickness:1px !important;}"
        # большая модалка «Что это?»
        "div[role='dialog']"
        "{width:86vw !important;max-width:86vw !important;"
        "height:calc(100vh - 24px) !important;"
        "max-height:calc(100vh - 24px) !important;"
        "display:flex !important;flex-direction:column !important;}"
        # внутренний контент-блок занимает всю высоту с прокруткой
        "div[role='dialog'] > div:nth-child(2)"
        "{flex:1 1 auto !important;overflow-y:auto !important;}"
        "</style>",
        unsafe_allow_html=True,
    )

    settings = render_sidebar()

    # Тема накатывается отдельным CSS-блоком: либо force-light, либо
    # force-dark. Streamlit на клиенте по умолчанию читает системную
    # `prefers-color-scheme`, и при системной тёмной перебивает наш
    # `theme.base = light` из config.toml. Чтобы свет был светом, а тьма —
    # тьмой ровно тогда, когда это выбрал пользователь, форсим явно.
    if settings.dark_theme:
        st.markdown(_DARK_THEME_CSS, unsafe_allow_html=True)
    else:
        st.markdown(_LIGHT_THEME_CSS, unsafe_allow_html=True)

    # Закрытие сайдбара по клику снаружи (только при узком viewport, где
    # сайдбар оверлеит контент). Streamlit-нативного поведения нет; ставим
    # один глобальный listener в parent-окне через нулевой iframe.
    components.html(
        """
        <script>
        (function(){
          const w = window.parent;
          if (w.__pbSidebarOutsideClick) return;
          w.__pbSidebarOutsideClick = true;
          w.document.addEventListener('click', function(e){
            if (w.innerWidth > 992) return;
            const sb = w.document.querySelector('[data-testid="stSidebar"]');
            if (!sb) return;
            if (sb.getAttribute('aria-expanded') === 'false') return;
            if (sb.contains(e.target)) return;
            const btn =
              w.document.querySelector('[data-testid="stSidebarCollapseButton"] button')
              || w.document.querySelector('[data-testid="stSidebarCollapseButton"]')
              || w.document.querySelector('button[kind="header"]');
            if (btn) btn.click();
          }, true);
        })();
        </script>
        """,
        height=0,
    )

    render_header()

    st.markdown("---")
    # Подменяем нативный label, чтобы разделить десктопный и мобильный текст:
    # на мобильном «перетащить» физически нельзя, оставляем «Выберите».
    st.markdown(
        "<style>"
        ".pb-upload-label{font-size:14px;color:#1B1B1B;margin-bottom:6px;}"
        ".pb-upload-label .pb-mobile{display:none;}"
        "@media (max-width: 768px){"
        " .pb-upload-label .pb-desktop{display:none;}"
        " .pb-upload-label .pb-mobile{display:inline;}"
        "}"
        "</style>"
        "<div class='pb-upload-label'>"
        f"<span class='pb-desktop'>Перетащите или выберите фотографии — до {MAX_PHOTOS} штук (JPG, PNG, HEIC)</span>"
        f"<span class='pb-mobile'>Выберите фотографии — до {MAX_PHOTOS} штук (JPG, PNG, HEIC)</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        " ",
        type=["jpg", "jpeg", "png", "heic", "heif"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if not uploaded_files:
        st.info(
            "Загрузите фото машин с видимыми номерами. "
            "JPG, PNG или HEIC, по одному или сразу несколько."
        )
        return

    # лимит
    if len(uploaded_files) > MAX_PHOTOS:
        st.warning(
            f"Загружено {len(uploaded_files)} фото — обработаем первые "
            f"{MAX_PHOTOS}, остальные пропустим."
        )
        uploaded_files = uploaded_files[:MAX_PHOTOS]

    # отфильтруем фото, которые пользователь скрыл кнопкой «Заменить»
    dismissed = st.session_state.get("dismissed", set())
    visible: list = []
    for u in uploaded_files:
        if file_id_from_bytes(u.getvalue()) not in dismissed:
            visible.append(u)

    if not visible:
        st.info(
            "Все загруженные фото скрыты. Перетащите новые в окно загрузки выше "
            "или нажмите крестик у файла и загрузите заново."
        )
        return

    # модель прогревается сейчас — это создаст cold-start spinner
    get_pipeline()

    total = len(visible)

    # «Скачать все одним архивом» хотим показать ВЫШЕ карточек: до низа
    # пачки никто не долистывает. Резервируем место плейсхолдером и
    # заполним его, когда наберём результаты.
    zip_slot = st.empty()
    # Тонкий sticky-прогресс. Прилипает к верху видимой области и держится
    # под аплоадером, пока пользователь скроллит ленту карточек. Полоса
    # высотой 3 px + микро-подпись «N из M».
    progress_slot = st.empty()
    bar_color = "#ECECEC" if settings.dark_theme else "#222"
    track_color = "#2A2A2A" if settings.dark_theme else "#EEE"
    bg_color = "#131313" if settings.dark_theme else "#FFF"
    label_color = "#9A9A9A" if settings.dark_theme else "#666"

    def _set_progress(text: str, pct: float) -> None:
        progress_slot.markdown(
            "<div style='position:sticky;top:0;z-index:50;"
            f"background:{bg_color};padding:6px 0 8px 0;'>"
            "<div style='display:flex;align-items:center;gap:10px;'>"
            f"<span style='font-size:12px;color:{label_color};"
            "white-space:nowrap;'>"
            f"{text}</span>"
            "<div style='flex:1 1 auto;height:3px;border-radius:2px;"
            f"background:{track_color};overflow:hidden;'>"
            f"<div style='width:{pct*100:.1f}%;height:100%;"
            f"background:{bar_color};transition:width .15s;'></div>"
            "</div></div></div>",
            unsafe_allow_html=True,
        )

    _set_progress(f"Готовлю обработку — 0 из {total}", 0.0)

    outcomes: list[CardOutcome] = []
    for i, uploaded in enumerate(visible, start=1):
        _set_progress(
            f"Обрабатываю {i} из {total}",
            (i - 1) / total,
        )
        outcome = process_card(uploaded, settings, position=i, total=total)
        if outcome is not None:
            outcomes.append(outcome)
    # после прогона убираем sticky-полосу, чтобы не висела бесполезно
    progress_slot.empty()

    if not outcomes:
        return

    n_total = len(outcomes)
    n_with = sum(1 for o in outcomes if o.detections)

    # Заполняем плейсхолдер архива «постфактум» — фото уже отрисованы ниже,
    # а кнопка появится сверху.
    if n_total >= 2:
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for o in outcomes:
                png = io.BytesIO()
                Image.fromarray(o.processed).save(png, format="PNG")
                stem = Path(o.name).stem
                zf.writestr(f"{stem}_platebuster.png", png.getvalue())
        with zip_slot.container():
            st.download_button(
                f"Скачать все результаты одним архивом ({n_total} файлов)",
                data=zip_buf.getvalue(),
                file_name="platebuster_results.zip",
                mime="application/zip",
                type="primary",
                use_container_width=False,
            )

    # ---------- сводка по пачке — только в отладке ----------
    if settings.debug:
        st.markdown("---")
        avg_total_ms = float(np.mean([sum(o.timings_ms.values()) for o in outcomes]))
        n_plates = sum(len(o.detections) for o in outcomes)
        cols = st.columns(4)
        cols[0].metric("Фото обработано", n_total)
        cols[1].metric(
            "С номерами",
            f"{n_with} из {n_total}" if n_total else "—",
        )
        cols[2].metric("Найдено номеров", n_plates)
        cols[3].metric("Среднее время", f"{avg_total_ms:.0f} мс")

        if n_with >= 5:
            st.markdown("**Распределение уверенности**")
            all_confs = [d.confidence for o in outcomes for d in o.detections]
            if all_confs:
                counts, edges = np.histogram(all_confs, bins=10, range=(0.0, 1.0))
                chart_data = {
                    f"{edges[i]:.1f}–{edges[i+1]:.1f}": int(counts[i])
                    for i in range(len(counts))
                }
                st.bar_chart(chart_data)

    st.markdown("---")
    st.caption(
        "Сделал П. В. Очкин, магистерская работа УрФУ ИРИТ-РТФ 2026. "
        f"Архитектура моделей, разбор экспериментов и полная версия "
        f"со всеми шестью моделями — в репозитории "
        f"({REPO_LINK_PLACEHOLDER})."
    )


if __name__ == "__main__":
    main()
