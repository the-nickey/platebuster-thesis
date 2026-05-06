# Запуск обучений в Yandex Cloud (A100)

Этап-в-этап инструкция как поднять VM, перенести данные и обучить все 4 модели
сравнения. Ожидаемое время на A100 при `epochs=300/200`: ~6-7 часов на всё.
Бюджет ~700 ₽ при ~74 ₽/ч.

## Маршрут одним взглядом

```
[локально]    bash cloud/build_bundle.sh                   ← один файл /tmp/training_bundle_*.tar.gz (~2 GB)
[YC console]  Object Storage → drag&drop архива в bucket   ← без CLI, прямо в браузере
[YC console]  правый клик на объекте → "Получить ссылку" → "Сделать публичным"  ← на 1 час
[YC console]  Compute Cloud → создать VM (A100, DLI образ)
[ssh на VM]   wget '<публ. ссылка>' && tar xzf … && bash run_all.sh
              ждать ~6-7 часов
[ssh на VM]   tar czf /tmp/runs.tar.gz runs/ logs/         ← упаковать результаты
[YC console]  scp с локалки забрать /tmp/runs.tar.gz       (или upload в bucket)
[локально]    распаковать → thesis/runs/
[YC console]  УДАЛИТЬ VM!                                  ← обязательно, иначе тикает оплата
```

Никакого `yc` CLI на локалке не требуется. Всё — через web-консоль.

---

## 1. На локалке: собрать bundle

Из корня `thesis/`:

```bash
bash cloud/build_bundle.sh
```

Скрипт:
1. пересоздаёт `cloud/bundle/` — папка с симлинками на нужные скрипты и данные
2. упаковывает её через `tar -h` (символлинки → реальные файлы) с исключением
   `unified/pretrain/` (~4 GB CCPD-Stage-A, не используется в 2-stage архитектуре)
3. печатает путь к `/tmp/training_bundle_<ts>.tar.gz` (~2 GB)

Под капотом архив содержит:
```
scripts/                       (~250 KB)
data/processed/unified/        (~1.3 GB — finetune + test_per_region)
data/processed/keypoint_crops/ (~1 GB — crop'ы для keypoint head)
run_all.sh, requirements.txt
```

Лимит загрузки в YC console — 5 GB на объект, мы укладываемся.

---

## 2. В YC console: залить bundle в Object Storage

1. Открой https://console.cloud.yandex.ru → **Object Storage**
2. Используем существующий bucket `number-plate-destroyer` (если ещё нет — создать новый, имя глобально-уникальное, класс Standard, доступ ограниченный).
3. Зайди в bucket → кнопка **"Загрузить"** → drag&drop файл `/tmp/training_bundle_*.tar.gz`.
   Загрузка ~2 GB на хорошем интернете 1-3 минуты.
4. После загрузки кликни на объект → **"Получить ссылку"** → **"Публичный"**.
   Скопируй URL вида `https://storage.yandexcloud.net/number-plate-destroyer/training_bundle_*.tar.gz`.
   Этот URL понадобится для `wget` на VM.

---

## 3. В YC console: создать VM с A100

Compute Cloud → **"Создать ВМ"**:

- **Имя:** `plate-thesis-a100`
- **Зона:** `ru-central1-a`
- **Платформа:** `gpu-standard-v3` (Intel Ice Lake) — выбираем "1× NVIDIA A100"
- **vCPU:** 28, **RAM:** 119 GB (приходит вместе с A100)
- **Образ:** Marketplace → **"Deep Learning Toolkit, Ubuntu 22.04"** (с CUDA + PyTorch + cuDNN preinstalled)
- **Диск:** 200 GB **SSD network-ssd**
- **Публичный IP:** **временный** (без него на VM не зайдёшь)
- **SSH-ключ:** вставить `cat ~/.ssh/id_ed25519.pub` с локалки
- **Пользователь:** `ubuntu`
- **Прерываемая (preemptible):** включить — даёт 50% скидку, риск отключения через 24 ч (нам хватает с большим запасом)

Запомни **публичный IP** после создания.

---

## 4. На VM: запустить обучение

```bash
ssh ubuntu@<vm-ip>

# проверь, что GPU виден
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# скачай bundle (URL из шага 2)
wget -O bundle.tar.gz '<публичная ссылка>'

# распакуй и запусти
mkdir -p ~/training_bundle && tar -xzf bundle.tar.gz -C ~/training_bundle
cd ~/training_bundle

# в tmux чтобы пережить разрыв ssh
tmux new -s train
bash run_all.sh 2>&1 | tee run_all.log

# detach: Ctrl+B затем D
# вернуться: tmux attach -t train
```

Из соседнего терминала на локалке можно мониторить:

```bash
ssh ubuntu@<vm-ip> 'tail -f ~/training_bundle/run_all.log'
ssh ubuntu@<vm-ip> 'nvidia-smi -l 5'
```

### Что выполнится в run_all.sh

| шаг | что делает | прим. время на V100 |
|---|---|---:|
| `deps` | `pip install -r requirements.txt` (включая CUDA torch) | 3-5 мин |
| `prepare` | патчит абсолютные пути в `data.yaml`, удаляет `*.cache` | < 1 сек |
| `coco` | конвертирует unified → COCO для RF-DETR | 1-2 мин |
| `classical` | контурный baseline на 5 регионах | 1-2 мин |
| `keypoint` | ResNet18 head, 60 эпох на 60K crops | ~25-30 мин |
| `yolo11` | YOLO11n-detect, до 300 эпох (early stop ~120) | ~70-90 мин |
| `yolo12` | YOLO12n-detect, до 300 эпох (early stop ~120) | ~80-100 мин |
| `yolo_pose` | YOLO11n-pose, до 300 эпох (early stop ~150) | ~90-120 мин |
| `rfdetr` | RF-DETR Medium, до 200 эпох | ~3-5 ч |
| **итого** | | **~7-9 ч** |

`per_region_metrics.json` пишется сразу после каждого шага в свою папку
`runs/<step>_<ts>/` — даже если RF-DETR упадёт, метрики предыдущих моделей не пропадут.

---

## 5. Забрать результаты

После завершения `run_all.sh` (или досрочно — если тебя устраивают результаты, не дожидаясь
RF-DETR):

```bash
# на VM
cd ~/training_bundle
tar czf /tmp/runs_$(date +%Y%m%d_%H%M).tar.gz runs/ logs/
ls -lh /tmp/runs_*.tar.gz
```

Простейший путь — **scp с локалки**:

```bash
# на локалке (из корня репо)
scp ubuntu@<vm-ip>:/tmp/runs_*.tar.gz ./
tar xzf runs_*.tar.gz
ls runs/
```

Альтернатива через console — на VM `wget --post-file ...` сложновато, проще scp.

---

## 6. Удалить VM (обязательно)

Иначе оплата идёт за каждый час:

```
YC console → Compute Cloud → ВМ → "Удалить"
```

Bucket с архивом тоже можно почистить — иначе хранение бесплатное только в первые 5 GB.

---

## 7. Если что-то пошло не так

- **`run_all.sh` упал на одном шаге** — остальные шаги продолжатся (каждый `step` изолирован).
  Логи в `~/training_bundle/logs/<step>.log`. Можно перезапустить только сломанный:
  ```bash
  SKIP_DEPS=1 SKIP_PREPARE=1 SKIP_COCO=1 SKIP_CLASSICAL=1 SKIP_KEYPOINT=1 \
  SKIP_YOLO11=1 SKIP_YOLO12=1 bash run_all.sh
  ```
- **`nvidia-smi` ругается / `torch.cuda.is_available() == False`** — на DLI образе обычно достаточно `sudo reboot` после первого ssh-входа.
- **`pip install rfdetr` ругается на torch версию** — DLI образ имеет свой torch, не трогай его. Если конфликт — `pip install rfdetr --no-deps` потом доставить вручную: `pip install pycocotools tqdm transformers safetensors timm`.
- **VM-preemptible отняли посреди ран-а** — checkpoint'ы YOLO/RF-DETR сохраняются каждые N эпох. Создай новую VM, скачай тот же bundle, и в `run_all.sh` поправь `--resume` для YOLO или `pretrain_weights` для RF-DETR на путь к последнему checkpoint'у.
- **Не хватило диска** — увеличь boot-диск в консоли без пересоздания VM.
- **YOLO кричит "path does not exist"** — `prepare` шаг должен был перепатчить пути. Проверь `cat data/processed/unified/finetune/data.yaml` на VM, что там `path:` указывает на актуальный путь под `~/training_bundle/`. Если нет — запусти `bash run_all.sh` заново, шаг `prepare` идёт первым.

---

## 8. Что в итоге выйдет в `runs/`

```
runs/
├── classical_<ts>/
│   ├── per_region_metrics.json
│   └── visualizations/
├── keypoint_head_<ts>/
│   ├── best.pt
│   ├── history.json
│   └── per_region_metrics.json
├── yolo11n_cuda_<ts>/
│   ├── weights/best.pt
│   └── per_region_metrics.json
├── yolo12n_cuda_<ts>/
│   ├── weights/best.pt
│   └── per_region_metrics.json
└── rfdetr_medium_<ts>/
    ├── checkpoint_best_total.pth
    └── per_region_metrics.json
```

Эти `per_region_metrics.json` идут прямо в финальную таблицу главы 2 §5.3.
