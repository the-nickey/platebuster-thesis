#!/bin/zsh
# platebuster — полная сборка (6 моделей через табы, вечный режим отладки).
# Двойной клик в Finder запускает Streamlit и открывает страницу в браузере.

set -euo pipefail
cd "$(dirname "$0")"

PORT=8600
URL="http://localhost:${PORT}"
PYTHON=".venv/bin/python"
APP="streamlit_app/app.py"

if [[ ! -x "$PYTHON" ]]; then
  echo "Не нашёл $PYTHON — нужен venv в корне репозитория."
  echo "Нажми Enter, чтобы закрыть."
  read -r
  exit 1
fi

# Если порт занят — освобождаем (старый запуск не закрылся корректно).
if lsof -ti:${PORT} >/dev/null 2>&1; then
  echo "Порт ${PORT} занят, останавливаю предыдущий запуск."
  lsof -ti:${PORT} | xargs kill -9 2>/dev/null || true
  sleep 1
fi

echo "Запускаю platebuster (полная сборка) на ${URL}"
# Откроем браузер сразу, не дожидаясь cold-start модели.
( sleep 3 && open "${URL}" ) &

exec "$PYTHON" -m streamlit run "$APP" \
  --server.port "$PORT" \
  --browser.gatherUsageStats false
