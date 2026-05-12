# streamlit_cloud_latency_probe — замер latency в реальной облачной среде

Отдельный Streamlit-сервис для замера per-image latency шести моделей-кандидатов
в среде Streamlit Community Cloud (1 vCPU). Закрывает комментарий руководителя
id=17 (репрезентативность среды) и id=33 (время инференса в облаке не измерялось).

## Архитектура

В репе только код (`app.py`, `inference.py`, `manifest.json`, `requirements.txt`)
— тяжёлые артефакты (веса ~310 МБ + 813 sample-картинок ~60 МБ) подгружаются
**с локального HTTP-сервера автора через cloudflared-туннель**, чтобы не
выкладывать их во внешние хранилища и не превысить лимит 1 ГБ Streamlit Cloud.

```
streamlit_cloud_latency_probe/
├── app.py                # Streamlit UI + загрузка артефактов + замер
├── inference.py          # single-image inference wrappers для 6 моделей
├── manifest.json         # пути к весам и картинкам относительно tunnel URL
├── requirements.txt      # torch CPU + ultralytics + rfdetr + …
└── README.md
```

## Запуск (поток работы автора)

### 1. На своём маке: поднять HTTP-сервер и cloudflared-туннель

```bash
# терминал 1 — в корне репо thesis
cd /Users/pvochkin/Desktop/vibecode/thesis
python -m http.server 8080

# терминал 2 — туннель (cloudflared должен быть установлен)
brew install cloudflared  # один раз
cloudflared tunnel --url http://localhost:8080
```

cloudflared выдаст URL вида `https://xxx-yyy-zzz.trycloudflare.com` — копируй.

### 2. В Streamlit Cloud (этот сервис)

Открываешь развёрнутое приложение, в сайдбаре вставляешь tunnel URL,
выбираешь модели и регионы (по умолчанию — все), нажимаешь «Прогнать».

Прогон ~22 минут на 1 vCPU при полном комплекте (813 кадров × 6 моделей).
По завершении — кнопка «Скачать все latency JSONL (zip)».

### 3. Положить результаты в bootstrap-репу

Распакуй zip, скопируй каждый `<region>__latency.jsonl` в:

```
streamlit_cloud_eval/data/predictions/<model>/
```

Перепушь репу bootstrap-сервиса — он автоматически подхватит latency и
покажет в таблицах две новых колонки: `p50 latency ms [ДИ]` и `p95 latency ms [ДИ]`.

## Деплой

1. Запушить содержимое этой папки как корень отдельной репы
   (например, `platebuster-latency-probe`).
2. На share.streamlit.io указать `app.py` точкой входа.
3. Зависимости из `requirements.txt` подхватятся при build.

CPU-сборка PyTorch указана явно через `--extra-index-url` — без этого pip
тянет CUDA-сборку (~750 МБ), и приложение не помещается в 1 ГБ.

## Что закрывается в работе

- Комм. id=17: «замер на Apple M3 Pro не репрезентативен для Streamlit
  Community Cloud, где используется виртуальный процессор» → теперь замер
  именно там, на 1 vCPU.
- Комм. id=33: «время собственно инференса в облачной среде не измерялось»
  → измерено per-image на 813 кадрах × 6 моделей.
- Доверительные интервалы для p50/p95 latency считаются в bootstrap-сервисе
  по этим JSONL (бутстрэп по картинкам, B=2000).

## Sample

Использует тот же sample, что и bootstrap-сервис
(`streamlit_cloud_eval/data/sample_image_ids.json`): 200 + 200 + 146 + 67 + 200
= **813 кадров**, seed=42. Это даёт прямое соответствие между метриками
качества (mAP, PCK) и метриками latency на одних и тех же кадрах.

## Ограничения

- Туннель работает пока запущен `cloudflared` на маке. Если сессия закрылась
  до окончания прогона — повторно запусти туннель и нажми «Прогнать» (уже
  скачанные веса/картинки лежат в `/tmp/` контейнера и не перекачиваются).
- При полной очистке контейнера (Streamlit Cloud перезапустил процесс) `/tmp/`
  чистится и потребуется повторное скачивание ~370 МБ.
- На rfdetr_medium на 1 vCPU расчёт ~8 минут — если сессия начнёт виснуть,
  прогоняй модели по очереди (multiselect в сайдбаре).
