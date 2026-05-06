"""
QC-инструмент для проверки ручной разметки 4 углов номера.

Делает две вещи:
  1) Визуальный обзор: рендерит фото с overlay (bbox + 4 угла + подписи TL/TR/BR/BL)
     в qc_output/, чтобы пробежаться глазами.
  2) Автоматическое выявление аномалий по эвристикам:
       - угол выходит за обрамляющий bbox > 10% его размера → ошибка кропа/клика
       - четырёхугольник не выпуклый (поломан порядок TL→TR→BR→BL)
       - аспект (длинная/короткая сторона) не в [1.8, 8] — нерегулярная плашка
       - площадь четырёхугольника < 40% от площади bbox — углы не дотянуты до краёв
       - угол вне [0..1] нормализованной системы — порча координат

Запуск:
    python scripts/qc_corners.py --dataset russian --split train
    python scripts/qc_corners.py --dataset russian --split valid --anomaly-only
    python scripts/qc_corners.py --dataset russian --split train --limit 100

Результат:
    qc_output/{dataset}_{split}/
      ├── ok/        нормальные (если без --anomaly-only)
      ├── flagged/   подозрительные
      └── report.txt отчёт со списком и причинами

После просмотра — любые ошибочные стемы добавляешь в .corners/_skipped.txt
(или удаляешь .corners/<stem>.txt и перерасмечаешь через mobile_corner_annotator).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent

ROBOFLOW_DATASETS = {
    "russian": REPO_ROOT / "data" / "roboflow" / "russian",
    "european": REPO_ROOT / "data" / "roboflow" / "european",
    "generic": REPO_ROOT / "data" / "roboflow" / "generic",
}
OPENALPR_REGIONS = {
    "br": REPO_ROOT / "data" / "processed" / "openalpr" / "br",
    "eu": REPO_ROOT / "data" / "processed" / "openalpr" / "eu",
    "us": REPO_ROOT / "data" / "processed" / "openalpr" / "us",
}
SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}

POINT_NAMES = ["TL", "TR", "BR", "BL"]
POINT_COLORS = [(0, 0, 255), (0, 200, 0), (255, 0, 0), (0, 215, 255)]  # BGR

# пороги аномалий
OOB_TOL_FRAC = 0.10        # допустимый выход угла за bbox, доля от размера bbox
ASPECT_RATIO_MIN = 0.4     # quad_aspect / bbox_aspect ниже этого → подозрение
ASPECT_RATIO_MAX = 2.5     # выше этого тоже подозрение (перспектива не растягивает в 2.5×)
AREA_RATIO_MIN = 0.40      # min(полезная_площадь_4угольника / площадь_bbox)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="russian",
                   choices=["russian", "european", "generic", "openalpr"])
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=str(REPO_ROOT / "qc_output"))
    p.add_argument("--limit", type=int, default=None,
                   help="обработать не больше N фото (default: все)")
    p.add_argument("--anomaly-only", action="store_true",
                   help="сохранять только flagged-фото, ok пропускать")
    p.add_argument("--remark", action="store_true",
                   help="не рендерить картинки, а УДАЛИТЬ corners-файлы для всех flagged "
                        "(и убрать их из _skipped.txt). "
                        "После этого они снова появятся как pending в mobile_corner_annotator.")
    return p.parse_args()


def resolve_split_dir(dataset: str, split: str) -> Path:
    if dataset == "openalpr":
        return OPENALPR_REGIONS[split]
    return ROBOFLOW_DATASETS[dataset] / SPLIT_ALIASES[split]


def parse_yolo_bboxes(label_path: Path):
    """[(class, cx, cy, w, h), ...] — нормализованные координаты"""
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = map(float, parts[1:5])
            out.append((cls, cx, cy, w, h))
        except ValueError:
            continue
    return out


def parse_corners(path: Path):
    """Список четвёрок углов: [((x1,y1),(x2,y2),(x3,y3),(x4,y4)), ...]"""
    if not path.exists() or path.name.startswith("_"):
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 8:
            continue
        try:
            v = list(map(float, parts))
            out.append(((v[0], v[1]), (v[2], v[3]), (v[4], v[5]), (v[6], v[7])))
        except ValueError:
            continue
    return out


def is_convex_quad(pts: list[tuple[float, float]]) -> bool:
    """Проверка выпуклости: все cross-product одного знака."""
    signs = []
    n = len(pts)
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        cx, cy = pts[(i + 2) % n]
        cross = (bx - ax) * (cy - by) - (by - ay) * (cx - bx)
        signs.append(np.sign(cross))
    signs = [s for s in signs if s != 0]
    return len(signs) == 0 or all(s == signs[0] for s in signs)


def quad_area(pts: list[tuple[float, float]]) -> float:
    """Shoelace area."""
    s = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def quad_aspect(pts: list[tuple[float, float]]) -> float:
    """Соотношение длинная/короткая сторона усреднённое: верх+низ vs левая+правая."""
    def dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
    top_bot = (dist(pts[0], pts[1]) + dist(pts[3], pts[2])) / 2
    lft_rgt = (dist(pts[0], pts[3]) + dist(pts[1], pts[2])) / 2
    short, long = sorted((top_bot, lft_rgt))
    return long / short if short > 0 else float("inf")


def find_anomalies(corners_norm, bboxes_norm) -> list[str]:
    """Возвращает список названий выявленных аномалий для этой четвёрки."""
    issues = []

    # 1) точки в [0..1]?
    for x, y in corners_norm:
        if not (0 <= x <= 1 and 0 <= y <= 1):
            issues.append("oob_normalized")
            break

    # 2) выпуклость
    if not is_convex_quad(list(corners_norm)):
        issues.append("non_convex")

    quad_aspect_v = quad_aspect(list(corners_norm))

    # 3) совпадение с bbox: ищем ближайший по центру bbox
    if bboxes_norm:
        xs = [p[0] for p in corners_norm]
        ys = [p[1] for p in corners_norm]
        cx_c, cy_c = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2

        best_b = None
        best_d = 1e9
        for _, cx, cy, w, h in bboxes_norm:
            d = (cx - cx_c) ** 2 + (cy - cy_c) ** 2
            if d < best_d:
                best_b, best_d = (cx, cy, w, h), d

        if best_b:
            cx, cy, w, h = best_b
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            tol_x, tol_y = w * OOB_TOL_FRAC, h * OOB_TOL_FRAC

            # out-of-bbox углы
            for (px, py) in corners_norm:
                if px < x1 - tol_x or px > x2 + tol_x or py < y1 - tol_y or py > y2 + tol_y:
                    issues.append("corner_out_of_bbox")
                    break

            # area ratio
            quad_a = quad_area(list(corners_norm))
            bbox_a = w * h
            if bbox_a > 0 and quad_a / bbox_a < AREA_RATIO_MIN:
                issues.append(f"small_area_ratio({quad_a/bbox_a:.2f})")

            # 4) аспект четырёхугольника не должен ДИКО отличаться от bbox.
            #    допуск намеренно широкий — перспектива 3/4 даёт большой разброс
            bbox_aspect = max(w, h) / max(min(w, h), 1e-6)
            if bbox_aspect > 0:
                ratio = quad_aspect_v / bbox_aspect
                if ratio < ASPECT_RATIO_MIN or ratio > ASPECT_RATIO_MAX:
                    issues.append(f"aspect_mismatch(quad={quad_aspect_v:.1f}/bbox={bbox_aspect:.1f})")

    return issues


def render_overlay(img, corners_norm, bboxes_norm, anomalies):
    h, w = img.shape[:2]
    canvas = img.copy()

    # все bbox-ы — magenta
    for _, cx, cy, bw, bh in bboxes_norm:
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 255), 2)

    # 4 угла + линии между ними
    pts_px = [(int(x * w), int(y * h)) for x, y in corners_norm]
    for i, (x, y) in enumerate(pts_px):
        cv2.circle(canvas, (x, y), 6, POINT_COLORS[i], -1)
        cv2.putText(canvas, POINT_NAMES[i], (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, POINT_COLORS[i], 2, cv2.LINE_AA)
    for i in range(4):
        cv2.line(canvas, pts_px[i], pts_px[(i + 1) % 4], (255, 255, 255), 2)

    # верхняя плашка с инфо
    if anomalies:
        text = "FLAGGED: " + ", ".join(anomalies)
        bar_color = (0, 0, 255)
    else:
        text = "OK"
        bar_color = (0, 180, 0)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 32), bar_color, -1)
    cv2.putText(canvas, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


def collect_flagged(corners_dir: Path, labels_dir: Path) -> list[tuple[str, list[str]]]:
    """Возвращает [(stem, [issues])] для всех flagged corner-файлов."""
    flagged = []
    for cf in sorted(corners_dir.glob("*.txt")):
        if cf.name.startswith("_"):
            continue
        bboxes = parse_yolo_bboxes(labels_dir / f"{cf.stem}.txt")
        corners_groups = parse_corners(cf)
        all_issues = []
        for cn in corners_groups:
            all_issues.extend(find_anomalies(cn, bboxes))
        if all_issues:
            flagged.append((cf.stem, all_issues))
    return flagged


def remove_stems_from_skipped(skipped_file: Path, stems_to_remove: set[str]) -> int:
    """Возвращает кол-во удалённых строк."""
    if not skipped_file.exists():
        return 0
    lines = skipped_file.read_text(encoding="utf-8").splitlines()
    kept = [l for l in lines if l.strip() and l.strip() not in stems_to_remove]
    removed = len(lines) - len(kept)
    skipped_file.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
    return removed


def remark_mode(corners_dir: Path, labels_dir: Path):
    """Удаляет corners-файлы для flagged + чистит _skipped.txt от их stem'ов."""
    flagged = collect_flagged(corners_dir, labels_dir)
    if not flagged:
        print("Аномалий не найдено — нечего перерасмечать.")
        return

    stems = {stem for stem, _ in flagged}
    removed_corners = 0
    for stem, issues in flagged:
        cf = corners_dir / f"{stem}.txt"
        if cf.exists():
            cf.unlink()
            removed_corners += 1

    removed_skipped = remove_stems_from_skipped(corners_dir / "_skipped.txt", stems)

    print(f"\n=== REMARK MODE ===")
    print(f"Flagged total:           {len(flagged)}")
    print(f"Удалено corners-файлов:  {removed_corners}")
    print(f"Удалено из _skipped.txt: {removed_skipped}")
    print(f"\nЭти фото снова попадут как pending в mobile_corner_annotator.")
    print(f"\nДетали (что и почему flagged):")
    for stem, issues in flagged[:20]:
        print(f"  {stem}: {', '.join(issues)}")
    if len(flagged) > 20:
        print(f"  ... и ещё {len(flagged) - 20}")


def main():
    args = parse_args()
    split_dir = resolve_split_dir(args.dataset, args.split)
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    corners_dir = split_dir / "corners"

    if not corners_dir.exists():
        raise SystemExit(f"нет корнеров в {corners_dir}")

    if args.remark:
        remark_mode(corners_dir, labels_dir)
        return

    out_root = Path(args.out) / f"{args.dataset}_{args.split}"
    flagged_dir = out_root / "flagged"
    ok_dir = out_root / "ok"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    if not args.anomaly_only:
        ok_dir.mkdir(parents=True, exist_ok=True)

    corner_files = sorted(p for p in corners_dir.glob("*.txt") if not p.name.startswith("_"))
    if args.limit:
        corner_files = corner_files[: args.limit]

    report_lines = []
    n_ok = n_flag = 0

    for cf in corner_files:
        stem = cf.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            report_lines.append(f"{stem}: no image found")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            report_lines.append(f"{stem}: cv2 cannot read")
            continue

        bboxes = parse_yolo_bboxes(labels_dir / f"{stem}.txt")
        corners_groups = parse_corners(cf)

        all_issues = []
        for cn in corners_groups:
            all_issues.extend(find_anomalies(cn, bboxes))

        if not corners_groups:
            continue
        overlay = render_overlay(img, corners_groups[0], bboxes, all_issues)

        if all_issues:
            cv2.imwrite(str(flagged_dir / f"{stem}.jpg"), overlay)
            report_lines.append(f"FLAGGED  {stem}: {', '.join(all_issues)}")
            n_flag += 1
        else:
            n_ok += 1
            if not args.anomaly_only:
                cv2.imwrite(str(ok_dir / f"{stem}.jpg"), overlay)

    report = (
        f"Dataset: {args.dataset}/{args.split}\n"
        f"Total corners: {len(corner_files)}\n"
        f"OK: {n_ok}\n"
        f"FLAGGED: {n_flag} ({n_flag * 100 / max(1, len(corner_files)):.1f}%)\n"
        f"\nFlagged details:\n" + "\n".join(report_lines if any("FLAGGED" in l for l in report_lines) else ["(нет)"])
    )
    (out_root / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nFlagged: {flagged_dir}")
    if not args.anomaly_only:
        print(f"OK: {ok_dir}")


if __name__ == "__main__":
    main()
