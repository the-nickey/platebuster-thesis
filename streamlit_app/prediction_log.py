"""Лог предсказаний и фидбэка пользователя.

Пишет JSONL в `streamlit_app/logs/predictions.jsonl`. Каждая строка — событие:

  {"type": "inference", ...}  — один запуск pipeline на одной картинке
  {"type": "feedback",  ...}  — кнопка 👍 / 👎 под результатом

Используется и для production-наблюдаемости, и для «норм / ненорм» eval'а
во время демо. Намеренно простой формат — `tail -f`, `jq`, `wc -l` всё
работает из коробки.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FILE = LOG_DIR / "predictions.jsonl"


def file_hash(file_bytes: bytes, length: int = 12) -> str:
    """Короткий стабильный hash файла (для группировки событий по фото)."""
    return hashlib.md5(file_bytes).hexdigest()[:length]


def log_event(event: dict) -> None:
    """Дописывает событие в JSONL. Создаёт каталог если его нет."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    event = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event}
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def read_events(limit: int | None = None) -> list[dict]:
    """Читает все события из лога, опционально ограничивает последними N."""
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if limit is not None:
        out = out[-limit:]
    return out


def aggregate(events: Iterable[dict]) -> dict:
    """Сводка по событиям для UI-панели."""
    inf = [e for e in events if e.get("type") == "inference"]
    fb = [e for e in events if e.get("type") == "feedback"]

    by_model: Counter = Counter()
    fb_by_model: dict[str, Counter] = {}
    n_with_detections = 0
    latencies: list[float] = []

    for e in inf:
        model = e.get("model", "?")
        by_model[model] += 1
        if e.get("n_detections", 0) > 0:
            n_with_detections += 1
        if "latency_ms" in e:
            latencies.append(float(e["latency_ms"]))

    for e in fb:
        model = e.get("model", "?")
        fb_by_model.setdefault(model, Counter())[e.get("verdict", "?")] += 1

    return {
        "total_inferences": len(inf),
        "total_feedback": len(fb),
        "by_model": dict(by_model),
        "feedback_by_model": {m: dict(c) for m, c in fb_by_model.items()},
        "detection_rate": (n_with_detections / len(inf)) if inf else None,
        "avg_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_ms": (sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None),
    }
