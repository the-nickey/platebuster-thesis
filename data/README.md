# `data/` — датасеты

В репозитории нет данных — все датасеты скачиваются скриптами из `scripts/`. Это сделано умышленно: датасеты тяжёлые (десятки гигабайт), у каждого своя лицензия, и они не могут перераспространяться как часть этой репы.

## Структура

```
data/
├── ccpd/         CCPD2019 / CCPD2020 — китайские номера, 4 угла в имени файла
├── openalpr/     OpenALPR Benchmark — три региона: br, eu, us
├── roboflow/     три публичных датасета с Roboflow Universe
├── manual/       (опционально) пользовательские фотографии для cross-domain
└── processed/    объединённый train / val / test в YOLO-keypoints формате
```

Лицензии каждого датасета и условия атрибуции: [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Сборка с нуля

```bash
# из корня репозитория
pip install -r requirements-dev.txt

# 1. CCPD2019 (~12 ГБ)
python scripts/download_ccpd.py

# 2. OpenALPR Benchmark (~400 МБ)
python scripts/build_openalpr_dataset.py

# 3. Roboflow Universe (нужен ROBOFLOW_API_KEY в .env, см. .env.example)
python scripts/download_roboflow.py

# 4. Объединение в YOLO-keypoints формат
python scripts/build_unified_dataset.py
```

После этого `data/processed/unified/` содержит готовые train / val / test для обучения.

## Доразметка 4 углов

CCPD приходит с 4 углами уже размеченными в имени файла. У публичных датасетов с Roboflow и OpenALPR — только bbox. Доразметить углы можно двумя путями:

- **CVAT** — стандартный инструмент. См. [`infra/cvat/README.md`](../infra/cvat/README.md). Перед загрузкой подготовь архивы скриптом `scripts/prepare_cvat_uploads.py`.
- **scripts/annotation/** — собственные лёгкие annotator-ы (быстрее CVAT для одной задачи): `corner_annotator.py` (десктоп) и `mobile_corner_annotator.py` (веб для тачскрина).

## `data/manual/` — опциональная папка

Зарезервирована для собственных фото, на которых хочется проверить cross-domain поведение моделей. По умолчанию пуста и игнорируется git-ом. Кладёшь туда свои изображения + разметку — и можешь использовать в evaluation pipeline.
