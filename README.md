# platebuster — open-source веб-сервис для скрытия автомобильных номеров

> 🇷🇺 Русская версия (основная). 🇬🇧 [English](README.en.md).

Сравнительное исследование подходов к детекции автомобильных номеров (классический CV vs нейросети разных классов: CNN, attention-centric, transformer-based) и open-source веб-сервис, который скрывает номера на фото с учётом перспективных искажений через гомографию по 4 угловым точкам.

**Лицензия:** [AGPL-3.0](LICENSE) — следствие лицензии Ultralytics YOLO. См. [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Что внутри

- Сравнение шести моделей в одном UI: классический baseline (контуры + морфология), ResNet18-regression, YOLO11n-detect + ResNet18-keypoint, YOLO11n-pose, YOLO12-pose, RF-DETR Nano/Small/Medium.
- Гомография по 4 углам — номер закрывается с учётом ракурса, а не плоским прямоугольником.
- Два режима: размытие или наложение логотипа.
- Две сборки: полная локальная (6 моделей, режим сравнения) и slim для Streamlit Community Cloud (одна модель, до 10 фото за раз).

## Стек

- **Модели:** YOLO11/YOLO26-pose (Ultralytics, AGPL-3.0), RF-DETR (Apache-2.0), ResNet18-regression baseline, классический baseline.
- **Обучение:** PyTorch 2.4+ с MPS (Apple Silicon) и CUDA в облаке.
- **Разметка:** CVAT (см. [`infra/cvat/`](infra/cvat/)) + лёгкие annotator-ы в [`scripts/annotation/`](scripts/annotation/).
- **Веб:** Streamlit.

## Быстрый старт — локальное приложение

```bash
# из корня репозитория
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# скачать веса в weights/ — см. weights/README.md

# полная сборка (6 моделей)
streamlit run streamlit_app/app.py --server.port 8600
```

На macOS можно вместо команды двойным кликом запустить [`platebuster-full.command`](platebuster-full.command) (полная сборка) или [`platebuster-cloud.command`](platebuster-cloud.command) (slim).

## Воспроизведение эксперимента с нуля

```bash
pip install -r requirements-dev.txt

# 1. Датасеты — см. data/README.md
python scripts/download_ccpd.py
python scripts/build_openalpr_dataset.py
python scripts/download_roboflow.py        # нужен ROBOFLOW_API_KEY в .env
python scripts/build_unified_dataset.py

# 2. Обучение
python scripts/training/train_yolo_pose.py
python scripts/training/train_yolo_detect.py
python scripts/training/train_rfdetr.py
python scripts/training/train_keypoint_head.py

# 3. Оценка
python scripts/training/eval_classical.py
python scripts/training/eval_two_stage_pipeline.py
python scripts/benchmark_latency.py
python scripts/collect_results.py
```

## Структура репозитория

```
platebuster-thesis/
├── src/plates/             # импортируемый Python-пакет (datasets, утилиты)
├── scripts/                # CLI-утилиты
│   ├── download_*.py       # скачивание публичных датасетов
│   ├── build_*.py          # сборка/конверсия в общий формат
│   ├── qc_*.py             # quality control разметки
│   ├── annotation/         # corner_annotator, mobile_corner_annotator
│   └── training/           # обучение и оценка моделей
├── streamlit_app/          # локальное приложение (полная сборка, 6 моделей)
├── streamlit_cloud/        # slim для Streamlit Community Cloud
├── notebooks/              # EDA / эксперименты
├── cloud/                  # инструкции деплоя обучения в облако
├── infra/cvat/             # инструкции по локальному CVAT
├── tests/                  # pytest unit-тесты
├── data/                   # пусто в репо; см. data/README.md
├── weights/                # пусто в репо; см. weights/README.md
├── requirements.txt        # production (streamlit cloud)
├── requirements-dev.txt    # +training/jupyter/eval
├── LICENSE                 # AGPL-3.0
└── THIRD_PARTY_NOTICES.md  # лицензии моделей и датасетов
```

## Что НЕ лежит в репозитории и почему

- **Датасеты** (`data/`) — десятки гигабайт публичных данных с разными лицензиями. Скачиваются скриптами, а не перераспространяются.
- **Веса** (`weights/`, `*.pt`, `*.pth`) — публикуются отдельно, в Releases / на Hugging Face Hub. См. [`weights/README.md`](weights/README.md).
- **CVAT-исходники** — это отдельный open-source проект; здесь только инструкция по запуску, см. [`infra/cvat/README.md`](infra/cvat/README.md).
- **Артефакты обучения** (`runs/`, выходы `qc_*.py` и т.п.) — пересобираются скриптами.

## Лицензия и атрибуция

Проект под [AGPL-3.0](LICENSE). Это требование Ultralytics YOLO, который используется в составе сравнения и в production-сборке.

Полная матрица лицензий моделей и датасетов: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
