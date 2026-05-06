# Third-party Notices

Этот проект использует и/или fine-тюнит несколько сторонних компонентов. Ниже — их лицензии и условия атрибуции.

## Модели

| Компонент                 | Лицензия     | Источник                                              |
|---------------------------|--------------|--------------------------------------------------------|
| YOLO11 / YOLO26 (poses)   | **AGPL-3.0** | https://github.com/ultralytics/ultralytics             |
| RF-DETR                   | Apache-2.0   | https://github.com/roboflow/rf-detr                    |
| ResNet18 (предобучение)   | BSD-3-Clause | https://github.com/pytorch/vision                      |

Использование Ultralytics YOLO под AGPL-3.0 определяет лицензию проекта в целом.

## Датасеты

| Датасет             | Лицензия / условия                                            | Источник                                              |
|---------------------|---------------------------------------------------------------|--------------------------------------------------------|
| CCPD2019 / CCPD2020 | Research-only (см. оригинальный репо)                         | https://github.com/detectRecog/CCPD                    |
| OpenALPR Benchmark  | См. `data/openalpr/LICENSE` после скачивания                  | https://github.com/openalpr/benchmarks                 |
| Roboflow Universe   | Per-dataset (часто CC-BY-4.0); см. карточку каждого датасета | https://universe.roboflow.com                          |

## Библиотеки (runtime)

| Библиотека            | Лицензия         |
|-----------------------|------------------|
| PyTorch               | BSD-3-Clause     |
| OpenCV (headless)     | Apache-2.0       |
| Streamlit             | Apache-2.0       |
| Pillow                | HPND             |
| NumPy                 | BSD-3-Clause     |
| Ultralytics (YOLO)    | AGPL-3.0         |

Полный список зависимостей — в `requirements.txt` и `requirements-dev.txt`.

## Цитирование

При использовании работ, упомянутых выше, корректно цитировать оригинальные публикации. Ссылки и BibTeX — в README соответствующих репозиториев.
