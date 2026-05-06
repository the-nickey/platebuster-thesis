# `streamlit_app/` — полная сборка для локального запуска

Локальное приложение со всеми шестью моделями и режимом сравнения.
Production-кандидат — 2-stage pipeline:

**YOLO11n-detect (bbox)** → **ResNet18 keypoint head (4 угла)** →
**`cv2.findHomography` + наложение** (размытие или логотип).

Slim-вариант для Streamlit Community Cloud (одна модель, минимальный UI) — в каталоге [`streamlit_cloud/`](../streamlit_cloud/).

## Структура

```
streamlit_app/
├── app.py              UI (Streamlit)
├── inference.py        2-stage pipeline + утилиты обработки
├── prediction_log.py   локальный лог предсказаний для UI
├── requirements.txt    pinned зависимости для деплоя
├── models/             веса для деплоя (опционально, копии из weights/)
└── assets/             логотипы по умолчанию
```

`app.py` сначала ищет веса в `streamlit_app/models/`. Если там пусто — fallback на корневую `weights/` (см. [`weights/README.md`](../weights/README.md)).

## Локальный запуск

```bash
# из корня репозитория
python3 -m venv .venv
source .venv/bin/activate
pip install -r streamlit_app/requirements.txt

streamlit run streamlit_app/app.py
# откроется http://localhost:8501
```

На macOS можно двойным кликом запустить [`platebuster-full.command`](../platebuster-full.command) — он поднимет приложение на порту 8600 и откроет браузер.

## Метрики production-кандидата

| Стадия                          | Bbox mAP@50 | Pose mean px error | Latency CPU |
|---------------------------------|------------:|-------------------:|------------:|
| Stage 1: YOLO11n-detect         | **0.908**   | —                  | 42.7 ms     |
| Stage 2: ResNet18 keypoint head | —           | 2.1 px (CCPD) / 3.4 px (Russian) | 12.1 ms |
| **Pipeline total**              | **0.908**   | **~3 px на номер** | **~55 ms**  |

Размер production-сборки — около 50 МБ (yolo11n 5.6 МБ + ResNet18 44 МБ).
