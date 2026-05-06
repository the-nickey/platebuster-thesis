#!/usr/bin/env bash
# Собирает training_bundle.tar.gz для заливки в Yandex Cloud Object Storage.
#
# Использование:
#   bash cloud/build_bundle.sh
#
# Результат:
#   1) пересоздаёт cloud/bundle/ — папку с симлинками на скрипты + данные
#   2) пакует её через `tar -h` (символлинки → реальные файлы) в /tmp/training_bundle_<ts>.tar.gz
#
# Этот один файл и заливается в YC Storage через drag&drop в web-консоли.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THESIS_ROOT="$(dirname "$SCRIPT_DIR")"
BUNDLE_DIR="$SCRIPT_DIR/bundle"

echo "thesis_root: $THESIS_ROOT"
echo "bundle_dir:  $BUNDLE_DIR"

# 1. (re)create bundle/ structure with absolute symlinks
rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR/data/processed"

ln -s "$THESIS_ROOT/scripts"                          "$BUNDLE_DIR/scripts"
ln -s "$SCRIPT_DIR/run_all.sh"                        "$BUNDLE_DIR/run_all.sh"
ln -s "$SCRIPT_DIR/requirements.txt"                  "$BUNDLE_DIR/requirements.txt"
ln -s "$THESIS_ROOT/data/processed/unified"           "$BUNDLE_DIR/data/processed/unified"
ln -s "$THESIS_ROOT/data/processed/keypoint_crops"    "$BUNDLE_DIR/data/processed/keypoint_crops"

echo ""
echo "=== bundle/ структура ==="
ls -la "$BUNDLE_DIR"
echo ""
echo "data/processed:"
ls -la "$BUNDLE_DIR/data/processed"

# 2. tar.gz с разворачиванием симлинков
# Кладём в cloud/ (не в /tmp), чтобы Finder показывал и файл не пропадал после reboot.
TS=$(date +%Y%m%d_%H%M)
OUT="$SCRIPT_DIR/training_bundle_${TS}.tar.gz"

echo ""
echo "=== пакую в $OUT ==="
echo "это ~3 GB и займёт 1-3 минуты на M3 Pro"

tar -czhf "$OUT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.cache' \
    --exclude='.DS_Store' \
    --exclude='unified/pretrain' \
    -C "$BUNDLE_DIR" .

# unified/pretrain/ исключён намеренно (~4 GB CCPD pretrain_mix):
# в 2-stage Avito-style архитектуре Stage A pretrain не используется
# (см. thesis/docs/resume_state.md §1.2). Если понадобится — убрать
# --exclude=unified/pretrain.

echo ""
echo "=== готово ==="
ls -lh "$OUT"
echo ""
echo "файл лежит здесь:"
echo "  $OUT"
echo ""
echo "в Finder открыть папку:"
echo "  open $SCRIPT_DIR"
echo ""
echo "следующий шаг:"
echo "  YC console → Object Storage → number-plate-destroyer → 'Загрузить'"
echo "  drag&drop файла $(basename "$OUT") в окно браузера"
echo "  после загрузки сделай объект публичным и скопируй URL"
echo ""
echo "на VM:"
echo "  wget -O bundle.tar.gz '<публичная ссылка>'"
echo "  mkdir -p ~/training_bundle && tar -xzf bundle.tar.gz -C ~/training_bundle"
echo "  cd ~/training_bundle && bash run_all.sh"
