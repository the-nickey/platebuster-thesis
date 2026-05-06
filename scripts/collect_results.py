"""
Собирает per_region_metrics.json от всех 5 моделей (classical, keypoint head,
yolo11n, yolo12n, rfdetr) в одну сводную таблицу — готовую вставку в главу 2 §5.3.

Запуск:
    python scripts/collect_results.py
    python scripts/collect_results.py --runs-dir runs/  --out chapter_2_table.md

По умолчанию ищет последние run-ы каждого типа в `runs/`. Выводит:
1. CSV-таблицу (для Excel / Google Sheets)
2. Markdown-таблицу (для главы 2)
3. JSON со всеми метриками (для скриптов-аналитики дальше)
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REGIONS = ["ccpd", "russian", "european", "openalpr", "generic"]

# В каком порядке показываем модели в итоговой таблице — сначала baseline'ы,
# потом 2-stage детекторы, в конце single-stage YOLO-pose
MODEL_ORDER = ["classical", "keypoint_head", "yolo11n", "yolo12n", "rfdetr", "yolo_pose"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--out-dir", default=str(REPO_ROOT / "runs" / "_summary"))
    return p.parse_args()


def find_latest_run(runs_dir: Path, prefix: str) -> Path | None:
    """Возвращает свежайшую папку runs/<prefix>_*/ с per_region_metrics.json."""
    candidates = []
    for d in runs_dir.iterdir():
        if not d.is_dir():
            continue
        if not d.name.startswith(prefix):
            continue
        metrics = d / "per_region_metrics.json"
        if metrics.exists():
            candidates.append((d.stat().st_mtime, d))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_metrics(runs_dir: Path) -> OrderedDict[str, dict]:
    out = OrderedDict()
    # имена в run_all.sh передаются через --name, поэтому совпадают с этими prefix'ами
    mapping = {
        "classical":    "classical",
        "keypoint_head": "keypoint_head",
        "yolo11n":      "yolo11n_cuda",        # из run_all.sh: --name yolo11n_cuda
        "yolo12n":      "yolo12n_cuda",        # из run_all.sh: --name yolo12n_cuda
        "rfdetr":       "rfdetr_",
        "yolo_pose":    "yolo11n_pose_cuda",   # из run_all.sh: --name yolo11n_pose_cuda
    }
    for model_name, prefix in mapping.items():
        run = find_latest_run(runs_dir, prefix)
        if run is None:
            print(f"  [{model_name}]: не найден run в {runs_dir} (prefix={prefix}*)")
            out[model_name] = None
            continue
        metrics_file = run / "per_region_metrics.json"
        if not metrics_file.exists():
            print(f"  [{model_name}]: найдена папка {run.name}, но нет per_region_metrics.json")
            out[model_name] = None
            continue
        metrics = json.loads(metrics_file.read_text())
        # str(run) — устойчиво к относительным/абсолютным путям, не ломается на cross-prefix
        out[model_name] = {"run": str(run), "metrics": metrics}
        print(f"  [{model_name}]: {run.name}")
    return out


def primary_score(model_name: str, region_metrics: dict) -> str:
    """Вернёт основную метрику одной строкой для столбца таблицы."""
    if region_metrics is None or "error" in region_metrics:
        return "—"
    # классика без mAP — F1 + pixel error
    if model_name == "classical":
        f1 = region_metrics.get("f1")
        pe = region_metrics.get("mean_pixel_error_corners")
        if f1 is None:
            return "—"
        s = f"F1={f1:.3f}"
        if pe is not None:
            s += f" / px={pe:.1f}"
        return s
    # keypoint head — pixel error
    if model_name == "keypoint_head":
        m = region_metrics.get("mean_px_err")
        return f"px={m:.2f}" if m is not None else "—"
    # YOLO-pose — bbox mAP50 и pose mAP50 одной строкой
    if model_name == "yolo_pose":
        bb = region_metrics.get("mAP50")
        ps = region_metrics.get("pose_mAP50")
        if bb is None:
            return "—"
        s = f"box={bb:.3f}"
        if ps is not None:
            s += f" / pose={ps:.3f}"
        return s
    # детекторы — mAP50
    m = region_metrics.get("mAP50")
    return f"{m:.3f}" if m is not None else "—"


def build_md_table(data: OrderedDict[str, dict]) -> str:
    header = ["Модель"] + REGIONS + ["Mean"]
    sep = ["---"] + [":-:" for _ in REGIONS] + [":-:"]

    rows = []
    for model_name in MODEL_ORDER:
        entry = data.get(model_name)
        if entry is None:
            rows.append([model_name, *(["—"] * (len(REGIONS) + 1))])
            continue
        metrics = entry["metrics"]
        cells = []
        ndices = []
        for r in REGIONS:
            cells.append(primary_score(model_name, metrics.get(r)))
            # для среднего считаем только числа из mAP/F1
            mr = metrics.get(r)
            if mr and "error" not in mr:
                v = mr.get("mAP50") if model_name not in ("classical", "keypoint_head") \
                    else mr.get("f1") if model_name == "classical" \
                    else mr.get("mean_px_err")
                if v is not None:
                    ndices.append(v)
        mean = f"{sum(ndices) / len(ndices):.3f}" if ndices else "—"
        rows.append([model_name, *cells, mean])

    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def build_csv_table(data: OrderedDict[str, dict]) -> str:
    """Полная CSV — все доступные числовые метрики, не только primary score."""
    rows = [["model", "region", "mAP50", "mAP50_95", "precision", "recall", "f1",
             "pose_mAP50", "pose_mAP50_95",
             "mean_pixel_error_corners", "mean_px_err", "median_px_err", "p90_px_err",
             "n_images", "n_predictions"]]
    for model_name in MODEL_ORDER:
        entry = data.get(model_name)
        if entry is None:
            continue
        metrics = entry["metrics"]
        for r in REGIONS:
            mr = metrics.get(r) or {}
            rows.append([
                model_name, r,
                mr.get("mAP50"), mr.get("mAP50_95"),
                mr.get("precision"), mr.get("recall"),
                mr.get("f1"),
                mr.get("pose_mAP50"), mr.get("pose_mAP50_95"),
                mr.get("mean_pixel_error_corners"),
                mr.get("mean_px_err"),
                mr.get("median_px_err"),
                mr.get("p90_px_err"),
                mr.get("n_images"), mr.get("n_predictions"),
            ])

    out = []
    for row in rows:
        out.append(",".join("" if x is None else str(x) for x in row))
    return "\n".join(out)


def main():
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== collect_results from {runs_dir} ===")
    data = load_metrics(runs_dir)

    md = build_md_table(data)
    csv = build_csv_table(data)
    raw = {
        m: (e["metrics"] if e else None) for m, e in data.items()
    }

    (out_dir / "table.md").write_text(md, encoding="utf-8")
    (out_dir / "table.csv").write_text(csv, encoding="utf-8")
    (out_dir / "all_metrics.json").write_text(
        json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n=== итоговая таблица (markdown) ===")
    print(md)
    print(f"\nфайлы:")
    print(f"  {out_dir / 'table.md'}")
    print(f"  {out_dir / 'table.csv'}")
    print(f"  {out_dir / 'all_metrics.json'}")


if __name__ == "__main__":
    main()
