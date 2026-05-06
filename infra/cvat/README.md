# Локальный CVAT для разметки 4 углов

Часть разметки в этом проекте шла через CVAT (для bbox + keypoints). CVAT — отдельный open-source проект, поэтому в этой репе он **не вендорится** — клонируется отдельно.

## Запуск

```bash
# в любом каталоге за пределами этой репы
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d
# UI на http://localhost:8080
```

Дальше — стандартный CVAT-флоу:

1. Создать проект в UI.
2. Подготовить архивы для загрузки скриптом `scripts/prepare_cvat_uploads.py` (он положит готовые `.zip` в `data/processed/cvat_uploads/`).
3. Импортировать каждый архив как task в CVAT (по одному на регион).
4. Поверх bbox-разметки расставить skeleton с 4 точками: TL, TR, BR, BL.
5. Экспортировать обратно — формат COCO Keypoints.

## Альтернатива — лёгкие annotator-ы

Для разметки только 4 углов часто быстрее использовать собственные тулы из этого репо:

- [`scripts/annotation/corner_annotator.py`](../../scripts/annotation/corner_annotator.py) — десктопный (cv2-окно, мышь).
- [`scripts/annotation/mobile_corner_annotator.py`](../../scripts/annotation/mobile_corner_annotator.py) — веб-сервер для тачскрина (Steam Deck / iPad).

См. docstring каждого скрипта для конкретных команд запуска.
