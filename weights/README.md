# `weights/` — обученные веса моделей

Веса в репозиторий не коммитятся (десятки и сотни мегабайт каждый). Скачиваются отдельно.

## Список весов

| Файл                  | Размер  | Архитектура                  | Лицензия    |
|-----------------------|---------|------------------------------|-------------|
| `yolo11n-pose.pt`     | ~6 МБ   | YOLO11n-pose (Ultralytics)   | AGPL-3.0    |
| `yolo11n.pt`          | ~5.6 МБ | YOLO11n-detect (Ultralytics) | AGPL-3.0    |
| `yolo26n-pose.pt`     | ~8 МБ   | YOLO26n-pose (Ultralytics)   | AGPL-3.0    |
| `rf-detr-nano.pth`    | ~366 МБ | RF-DETR Nano (Roboflow)      | Apache-2.0  |
| `rf-detr-small.pth`   | ~386 МБ | RF-DETR Small (Roboflow)     | Apache-2.0  |
| `rf-detr-medium.pth`  | ~405 МБ | RF-DETR Medium (Roboflow)    | Apache-2.0  |
| `resnet18-corners.pt` | ~44 МБ  | ResNet18 keypoint regressor  | AGPL-3.0    |

Все веса fine-tuned на смеси CCPD2019, OpenALPR Benchmark и публичных датасетов с Roboflow Universe. Подробности про каждый датасет — в [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## Где взять

Веса публикуются как [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) и/или на Hugging Face Hub.

Куда положить файлы: прямо в `weights/<filename>.pt`. Если веса есть локально — `streamlit_app/inference.py` подхватит их автоматически.

```bash
# вариант 1: GitHub Releases (после первого релиза)
# скачай нужный файл со страницы Releases репозитория и положи в weights/

# вариант 2: Hugging Face Hub
huggingface-cli download <repo-id> --local-dir weights/
```

## Размеры и лимиты

Полная локальная сборка задействует все модели — суммарно ~1.2 ГБ весов. Slim-сборка для Streamlit Community Cloud использует только `yolo11n-pose.pt` (~6 МБ), что укладывается в free-tier лимиты Streamlit без LFS.
