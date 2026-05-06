# platebuster — облачная сборка

Минимальная версия приложения для Streamlit Community Cloud. Одна модель
(YOLO11n-pose v2, 5,4 МБ), до десяти фотографий за раз, режимы — размытие
или логотип через гомографию.

Полная версия с шестью моделями и режимом сравнения — в каталоге
`streamlit_app/` репозитория.

## Что внутри

```
streamlit_cloud/
├── app.py                    UI и оркестрация
├── inference.py              урезанный pose-pipeline (одна модель)
├── models/
│   └── yolo11n-pose-v2.pt    веса 5,4 МБ
├── requirements.txt          CPU-wheel'ы Torch + минимум зависимостей
├── runtime.txt               python-3.11
├── .streamlit/config.toml    тема и параметры сервера
└── assets/                   зарезервировано под логотип бренда
```

## Локальный запуск

```bash
# из корня репозитория
python -m streamlit run streamlit_cloud/app.py
```

Откроется на `http://localhost:8501`.

## Развёртывание на Streamlit Community Cloud

1. Закоммитить каталог `streamlit_cloud/` в публичный репозиторий на
   GitHub. Веса (`models/yolo11n-pose-v2.pt`) — через GitHub LFS,
   иначе репозиторий распухнет.
2. На [share.streamlit.io](https://share.streamlit.io) — New app:
   - Repository: `<user>/<repo>`
   - Branch: `main`
   - Main file path: `streamlit_cloud/app.py`
   - Python version: 3.11 (берётся из `runtime.txt`)
3. Дождаться первого билда — около 5–7 минут.
4. После деплоя первый «холодный» запуск страницы — около 30 секунд
   (поднимается контейнер и грузится модель). Дальше обработка одной
   фотографии — около 50 миллисекунд.

## Лимиты Community Cloud

Free-tier рассчитан так:

- хранилище приложения — 1 ГБ
- оперативная память — 1 ГБ
- LFS-трафик — 1 ГБ в месяц

Сборка укладывается с запасом: torch CPU + torchvision + ultralytics +
opencv-headless ≈ 310 МБ, веса 5,4 МБ.
