#!/usr/bin/env bash
# Orchestrator для Yandex Cloud A100 VM. Запускается из распакованного training_bundle/.
#
# Окружение по умолчанию: Ubuntu 22.04 + CUDA 12.x + Python 3.11 (Yandex DLI image
# "Ubuntu 22.04 + Deep Learning Toolkit").
#
# Управление через env vars (по умолчанию все шаги ВЫПОЛНЯЮТСЯ):
#   SKIP_DEPS=1         не ставить pip-зависимости (если уже стоят)
#   SKIP_PREPARE=1      не патчить пути в data.yaml и не чистить .cache
#   SKIP_COCO=1         не пересобирать COCO для RF-DETR
#   SKIP_CLASSICAL=1    пропустить classical baseline
#   SKIP_KEYPOINT=1     пропустить keypoint head
#   SKIP_YOLO11=1       пропустить YOLO11n
#   SKIP_YOLO12=1       пропустить YOLO12n
#   SKIP_RFDETR=1       пропустить RF-DETR
#   YOLO_EPOCHS=300     эпохи YOLO (default 300, см. главу 2 §5.4)
#   RFDETR_EPOCHS=200   эпохи RF-DETR (default 200)
#   RFDETR_MODEL=medium nano|small|medium|large
#
# Запуск:
#   cd ~/training_bundle
#   bash run_all.sh 2>&1 | tee run_all.log
#
# По окончании веса и метрики всех моделей лежат в ~/training_bundle/runs/.
# Упакуй: tar czf runs.tar.gz runs/ logs/

set -u
cd "$(dirname "$0")"
BUNDLE_ROOT="$(pwd)"

mkdir -p runs logs

# ────────────────────── шаги ──────────────────────

do_deps() {
  # snapshot torch ДО установки — если pip-rfdetr притащит другой torch и сломает CUDA,
  # увидим разницу и сообщим пользователю.
  python3 -c "import torch; print('PRE: torch=', torch.__version__, 'cuda=', torch.cuda.is_available())" 2>&1 || echo "PRE: torch не установлен"

  python3 -m pip install --upgrade pip

  # Сначала ставим CUDA-сборку torch с нужного индекса. На чистой Ubuntu 22.04 LTS
  # без DLI-образа `pip install ultralytics` иначе притащит CPU-only torch.
  # На DLI-образе torch уже стоит — pip скажет "already satisfied" и не тронет.
  python3 -m pip install --retries 5 \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.org/simple \
    torch torchvision

  python3 -m pip install --retries 5 -r requirements.txt

  python3 -c "import torch
print('POST: torch=', torch.__version__, 'cuda=', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('!!! CUDA не виден после pip install. Скорее всего pip-rfdetr перезатёр conda-torch на CPU-only. '
                     'Перезапусти: pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121')
print('device=', torch.cuda.get_device_name(0))"

  # rfdetr extras проверка — без [train] model.train() не работает
  python3 -c "
try:
    from rfdetr.training import train_one_epoch  # noqa
    print('rfdetr[train] extras: OK')
except ImportError as e:
    raise SystemExit(f'!!! rfdetr[train] extras не установлены: {e}. Запусти: pip install \"rfdetr[train,loggers]>=1.6.5\"')
"
}

do_prepare() {
  # 1) переписываем абсолютные пути в data.yaml на путь под этой VM
  for yml in data/processed/unified/finetune/data.yaml data/processed/unified/pretrain/data.yaml; do
    if [[ -f "$yml" ]]; then
      target_dir="$(dirname "$(realpath "$yml")")"
      python3 -c "
import sys, pathlib
p = pathlib.Path('$yml')
out = []
for line in p.read_text().splitlines():
    if line.startswith('path:'):
        out.append(f'path: $target_dir')
    else:
        out.append(line)
p.write_text('\n'.join(out) + '\n')
print(f'patched {p} → path: $target_dir')
"
    fi
  done
  # 2) удаляем Ultralytics .cache файлы — они содержат старые пути с локалки автора
  find data/processed/unified -name "*.cache" -delete -print
  echo "prepare done"
}

do_coco() {
  python3 scripts/training/convert_yolo_to_coco.py
}

do_classical() {
  python3 scripts/training/eval_classical.py --visualize 20
}

do_keypoint() {
  python3 scripts/training/train_keypoint_head.py \
    --epochs "${KEYPOINT_EPOCHS:-60}" --batch 128 --workers 8
}

do_yolo11() {
  python3 scripts/training/train_yolo_detect.py \
    --model yolo11n.pt --preset cuda \
    --epochs "${YOLO_EPOCHS:-300}" \
    --patience "${YOLO_PATIENCE:-25}" \
    --batch "${YOLO_BATCH:-64}" \
    --name yolo11n_cuda
}

do_yolo12() {
  python3 scripts/training/train_yolo_detect.py \
    --model yolo12n.pt --preset cuda \
    --epochs "${YOLO_EPOCHS:-300}" \
    --patience "${YOLO_PATIENCE:-25}" \
    --batch "${YOLO_BATCH:-64}" \
    --name yolo12n_cuda
}

do_yolo_pose() {
  python3 scripts/training/train_yolo_pose.py \
    --model yolo11n-pose.pt --preset cuda \
    --epochs "${YOLO_EPOCHS:-300}" \
    --patience "${YOLO_PATIENCE:-25}" \
    --batch "${YOLO_BATCH:-64}" \
    --name yolo11n_pose_cuda
}

do_rfdetr() {
  python3 scripts/training/train_rfdetr.py \
    --model "${RFDETR_MODEL:-medium}" \
    --epochs "${RFDETR_EPOCHS:-200}" \
    --batch-size "${RFDETR_BATCH:-8}" \
    --grad-accum "${RFDETR_GRAD_ACCUM:-2}" \
    --patience "${RFDETR_PATIENCE:-20}"
}

# ────────────────────── runner ──────────────────────

step() {
  local name="$1"
  local skip_var="SKIP_$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')"
  if [[ "${!skip_var:-0}" == "1" ]]; then
    echo ""
    echo "=== [$name] SKIPPED (${skip_var}=1) ==="
    return 0
  fi
  echo ""
  echo "============================================================"
  echo "=== [$name] start: $(date +%H:%M:%S)"
  echo "============================================================"
  local t0=$(date +%s)
  set +e
  "do_$name" 2>&1 | tee "logs/${name}.log"
  local rc=${PIPESTATUS[0]}
  set -e
  local t1=$(date +%s)
  if [[ $rc -ne 0 ]]; then
    echo "=== [$name] FAILED (rc=$rc) after $((t1-t0))s ==="
  else
    echo "=== [$name] done in $((t1-t0))s ==="
  fi
}

# ────────────────────── pipeline ──────────────────────

echo "=== bundle root: $BUNDLE_ROOT ==="
echo "=== started at: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

step deps
step prepare
step coco

# быстрые шаги первыми (если что — раньше получим ошибку)
step classical
step keypoint

# тяжёлые
step yolo11
step yolo12
step yolo_pose
step rfdetr

# ────────────────────── сводный отчёт ──────────────────────
echo ""
echo "============================================================"
echo "=== СТАТУС ШАГОВ ==="
echo "============================================================"
for step_name in deps prepare coco classical keypoint yolo11 yolo12 yolo_pose rfdetr; do
  log="logs/${step_name}.log"
  if [[ ! -f "$log" ]]; then
    echo "  ⊘ $step_name — не запускался (SKIP)"
  elif tail -50 "$log" 2>/dev/null | grep -q -E "Traceback|FAILED|Error:|SystemExit:|CUDA out of memory"; then
    echo "  ✗ $step_name — УПАЛ. Смотри: cat $log | tail -80"
  else
    echo "  ✓ $step_name — ОК"
  fi
done

echo ""
echo "============================================================"
echo "=== МЕТРИКИ ==="
echo "============================================================"
find runs -maxdepth 3 -name "per_region_metrics.json" -print | while read f; do
  echo ""
  echo "--- $f ---"
  cat "$f"
done

echo ""
echo "=== finished at: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo ""

# автоупаковка результатов — на случай если ты заберёшь сонный
RUNS_TARBALL="/tmp/runs_$(date +%Y%m%d_%H%M).tar.gz"
echo "пакую результаты в $RUNS_TARBALL ..."
tar czf "$RUNS_TARBALL" runs/ logs/ 2>&1 | tail -5
ls -lh "$RUNS_TARBALL"
echo ""
echo "забери на локалке:"
echo "  scp ubuntu@<vm-ip>:$RUNS_TARBALL ./"
echo ""

# ────────────────────── auto-shutdown ──────────────────────
# Окно SHUTDOWN_DELAY минут чтобы успеть забрать. 15 — компромисс между
# «не платить за сон» и «есть шанс сделать scp». Если просыпаешься позже —
# просто включишь VM обратно в YC console, диск никуда не делся.
# AUTO_SHUTDOWN=0 чтобы отключить совсем.
SHUTDOWN_DELAY="${SHUTDOWN_DELAY:-15}"
if [[ "${AUTO_SHUTDOWN:-1}" == "1" ]]; then
  echo "==================================================================="
  echo "VM выключится через $SHUTDOWN_DELAY минут"
  echo "  отмена:     sudo shutdown -c"
  echo "  немедленно: sudo shutdown -h now"
  echo "  ВАЖНО: shutdown останавливает VM, но не удаляет её. Диск+IP"
  echo "         тикают ~5 ₽/ч. После того как забрал $RUNS_TARBALL —"
  echo "         удали VM в YC console (Compute → ВМ → Удалить)."
  echo "==================================================================="
  sudo shutdown -h "+$SHUTDOWN_DELAY" 2>&1 || echo "(не удалось запланировать shutdown — выключи VM руками)"
fi
