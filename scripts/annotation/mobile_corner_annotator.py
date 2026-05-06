"""
Веб-сервер для разметки 4 углов номера с тачскрина (Steam Deck / iPad).

Запуск:
    python mobile_corner_annotator.py --dataset russian --split valid
    python mobile_corner_annotator.py --dataset russian --split test
    python mobile_corner_annotator.py --dataset european --split train

Для openalpr split интерпретируется как регион (br/eu/us):
    python mobile_corner_annotator.py --dataset openalpr --split br

Открыть с тачскрина: http://<IP-мака>:8501
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from io import BytesIO

from flask import Flask, jsonify, send_file, request, Response
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent

# roboflow-датасеты с одинаковой структурой {train,valid,test}/{images,labels}/
ROBOFLOW_DATASETS = {
    "russian": REPO_ROOT / "data" / "roboflow" / "russian",
    "european": REPO_ROOT / "data" / "roboflow" / "european",
    "generic": REPO_ROOT / "data" / "roboflow" / "generic",
}

# openalpr — без split'ов, разбит на регионы br/eu/us
OPENALPR_REGIONS = {
    "br": REPO_ROOT / "data" / "processed" / "openalpr" / "br",
    "eu": REPO_ROOT / "data" / "processed" / "openalpr" / "eu",
    "us": REPO_ROOT / "data" / "processed" / "openalpr" / "us",
}

# алиасы val/valid — Roboflow зовёт это "valid"
SPLIT_ALIASES = {
    "train": "train",
    "val": "valid",
    "valid": "valid",
    "test": "test",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# насколько расширять bbox при кропе (доля от размеров bbox)
CROP_PADDING = 0.25

# из множества bbox в одном фото берём самый крупный
USE_LARGEST_BOX = True


def resolve_split_dir(dataset: str, split: str) -> Path:
    if dataset == "openalpr":
        if split not in OPENALPR_REGIONS:
            raise SystemExit(
                f"для openalpr split должен быть одним из {list(OPENALPR_REGIONS)}, получено {split!r}"
            )
        return OPENALPR_REGIONS[split]

    if dataset not in ROBOFLOW_DATASETS:
        raise SystemExit(
            f"неизвестный dataset {dataset!r}. ожидается один из "
            f"{list(ROBOFLOW_DATASETS) + ['openalpr']}"
        )

    if split not in SPLIT_ALIASES:
        raise SystemExit(f"неизвестный split {split!r}. ожидается train/val/valid/test")

    return ROBOFLOW_DATASETS[dataset] / SPLIT_ALIASES[split]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="russian",
                   choices=["russian", "european", "generic", "openalpr"])
    p.add_argument("--split", default="train",
                   help="train/valid/test для roboflow-датасетов; br/eu/us для openalpr")
    p.add_argument("--port", type=int, default=8501)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--min-bbox-px", type=int, default=30,
                   help="не показывать фото, где самый крупный bbox по короткой стороне меньше этого порога. "
                        "не прошедшие фото авто-добавляются в _skipped.txt и попадают в датасет как bbox-only.")
    p.add_argument("--min-image-px", type=int, default=480,
                   help="не показывать фото, где короткая сторона меньше этого порога")
    return p.parse_args()


ARGS = parse_args()
DATASET_SPLIT_DIR = resolve_split_dir(ARGS.dataset, ARGS.split)

IMAGES_DIR = DATASET_SPLIT_DIR / "images"
YOLO_LABELS_DIR = DATASET_SPLIT_DIR / "labels"
OUT_DIR = DATASET_SPLIT_DIR / "corners"
SKIPPED_FILE = OUT_DIR / "_skipped.txt"

if not IMAGES_DIR.exists():
    raise SystemExit(f"не найдено {IMAGES_DIR}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)


def find_images() -> list[Path]:
    return sorted(p for p in IMAGES_DIR.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def get_skipped_stems() -> set[str]:
    if not SKIPPED_FILE.exists():
        return set()
    return {
        line.strip()
        for line in SKIPPED_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def get_done_stems() -> set[str]:
    return {p.stem for p in OUT_DIR.glob("*.txt") if not p.name.startswith("_")}


def label_path_for_image(image_path: Path) -> Path:
    return YOLO_LABELS_DIR / f"{image_path.stem}.txt"


def corners_path_for_image(image_path: Path) -> Path:
    return OUT_DIR / f"{image_path.stem}.txt"


def read_yolo_boxes(image_path: Path, img_w: int, img_h: int) -> list[dict]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            xc, yc, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue

        x1 = max(0, min(img_w - 1, int((xc - bw / 2) * img_w)))
        y1 = max(0, min(img_h - 1, int((yc - bh / 2) * img_h)))
        x2 = max(0, min(img_w, int((xc + bw / 2) * img_w)))
        y2 = max(0, min(img_h, int((yc + bh / 2) * img_h)))

        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append({"class": cls, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "area": (x2 - x1) * (y2 - y1)})
    return boxes


def choose_box(boxes: list[dict]) -> dict | None:
    if not boxes:
        return None
    if USE_LARGEST_BOX:
        return max(boxes, key=lambda b: b["area"])
    return boxes[0]


def add_padding_to_box(box: dict, img_w: int, img_h: int, padding: float = CROP_PADDING):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * padding), int(bh * padding)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(img_w, x2 + pad_x), min(img_h, y2 + pad_y))


def get_pending_images() -> list[Path]:
    images = find_images()
    done, skipped = get_done_stems(), get_skipped_stems()
    return [img for img in images if img.stem not in done and img.stem not in skipped]


def auto_skip_too_small(min_image_px: int, min_bbox_px: int) -> dict[str, int]:
    """Прогоняет все pending-фото через фильтры размера. Не прошедшие — добавляет в _skipped.txt.
    Возвращает {already_skipped, newly_skipped, kept}."""
    images = find_images()
    done, skipped = get_done_stems(), get_skipped_stems()

    newly_skipped = []
    kept = 0
    for img_path in images:
        if img_path.stem in done or img_path.stem in skipped:
            continue
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            newly_skipped.append(img_path.stem)
            continue

        if min(w, h) < min_image_px:
            newly_skipped.append(img_path.stem)
            continue

        boxes = read_yolo_boxes(img_path, w, h)
        if not boxes:
            newly_skipped.append(img_path.stem)
            continue
        box = choose_box(boxes)
        bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
        if min(bw, bh) < min_bbox_px:
            newly_skipped.append(img_path.stem)
            continue

        kept += 1

    if newly_skipped:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with SKIPPED_FILE.open("a", encoding="utf-8") as f:
            for stem in newly_skipped:
                f.write(stem + "\n")

    return {"already_skipped": len(skipped), "newly_skipped": len(newly_skipped), "kept": kept}


def get_image_info(image_path: Path) -> dict:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size

    boxes = read_yolo_boxes(image_path, img_w, img_h)
    box = choose_box(boxes)

    if box is None:
        crop = {"x1": 0, "y1": 0, "x2": img_w, "y2": img_h}
        has_box = False
    else:
        cx1, cy1, cx2, cy2 = add_padding_to_box(box, img_w, img_h)
        crop = {"x1": cx1, "y1": cy1, "x2": cx2, "y2": cy2}
        has_box = True

    return {"filename": image_path.name, "stem": image_path.stem,
            "image_width": img_w, "image_height": img_h,
            "crop": crop, "has_box": has_box, "boxes_count": len(boxes)}


@app.route("/")
def index():
    title = f"{ARGS.dataset}/{ARGS.split}"
    html = """
<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Разметка углов — __TITLE__</title>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
  <style>
    body { margin: 0; padding: 12px; font-family: -apple-system, BlinkMacSystemFont, system-ui, sans-serif;
           background: #111; color: #f2f2f2; }
    .top { display: flex; gap: 8px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
    button { border: 0; border-radius: 10px; padding: 12px 14px; font-size: 16px; font-weight: 700;
             color: white; background: #333; }
    button.primary { background: #2f8f46; }
    button.warn { background: #c0392b; }
    button.secondary { background: #555; }
    #status { font-size: 14px; line-height: 1.35; opacity: 0.95; margin-bottom: 8px; }
    #hint { font-size: 18px; font-weight: 800; margin: 8px 0 10px; }
    .tl { color: #ff4040; } .tr { color: #35d04f; } .br { color: #40a0ff; } .bl { color: #ffd22d; }
    #canvasWrap { width: 100%; touch-action: none; user-select: none; }
    canvas { width: 100%; height: auto; display: block; border-radius: 12px; background: #222; touch-action: none; }
    .small { font-size: 13px; opacity: 0.75; margin-top: 10px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 6px; background: #2f8f46; font-size: 13px; font-weight: 700; }
  </style>
</head>
<body>
  <div class="top">
    <span class="badge">__TITLE__</span>
    <button class="secondary" onclick="undoPoint()">↩ Назад</button>
    <button class="secondary" onclick="resetPoints()">⟲ Сброс</button>
    <button class="warn" onclick="skipImage()">Пропустить</button>
  </div>
  <div id="status">Загрузка...</div>
  <div id="hint">
    Порядок: <span class="tl">TL</span> → <span class="tr">TR</span> →
    <span class="br">BR</span> → <span class="bl">BL</span>
  </div>
  <div id="canvasWrap"><canvas id="canvas"></canvas></div>
  <div class="small">Тапни 4 угла номера: TL → TR → BR → BL. После 4-го тапа файл сохранится автоматически.</div>
<script>
let current = null;
let img = new Image();
let points = [];
const pointNames = ["TL", "TR", "BR", "BL"];
const pointColors = ["#ff4040", "#35d04f", "#40a0ff", "#ffd22d"];
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

function setStatus(text) { document.getElementById("status").innerHTML = text; }

async function loadNext() {
  points = [];
  const res = await fetch("/api/next");
  current = await res.json();
  if (current.done) {
    setStatus("🎉 Всё размечено в этом split'е");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  setStatus("<b>" + current.filename + "</b><br>Осталось: " + current.remaining +
            " | bbox: " + (current.has_box ? "есть" : "нет") + " | boxes: " + current.boxes_count);
  img = new Image();
  img.onload = () => { canvas.width = img.naturalWidth; canvas.height = img.naturalHeight; draw(); };
  img.src = "/api/crop/" + encodeURIComponent(current.filename) + "?t=" + Date.now();
}

function draw() {
  if (!current) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);
  if (points.length >= 2) {
    ctx.lineWidth = 3; ctx.strokeStyle = "white";
    ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) ctx.lineTo(points[i].x, points[i].y);
    if (points.length === 4) ctx.closePath();
    ctx.stroke();
  }
  for (let i = 0; i < points.length; i++) {
    const p = points[i];
    ctx.fillStyle = pointColors[i]; ctx.strokeStyle = "black"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
    ctx.font = "bold 22px sans-serif"; ctx.fillStyle = pointColors[i];
    ctx.strokeStyle = "black"; ctx.lineWidth = 4;
    ctx.strokeText(pointNames[i], p.x + 12, p.y - 12);
    ctx.fillText(pointNames[i], p.x + 12, p.y - 12);
  }
  if (points.length < 4) {
    const next = pointNames[points.length];
    document.getElementById("hint").innerHTML = "Клик " + (points.length + 1) +
      "/4: <span style='color:" + pointColors[points.length] + "'>" + next + "</span>";
  } else {
    document.getElementById("hint").innerHTML = "Сохраняю...";
  }
}

canvas.addEventListener("pointerdown", async (event) => {
  event.preventDefault();
  if (!current || current.done || points.length >= 4) return;
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  points.push({ x, y }); draw();
  if (points.length === 4) await savePoints();
});

async function savePoints() {
  const res = await fetch("/api/save", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: current.filename, points: points })
  });
  const data = await res.json();
  if (!data.ok) { alert("Ошибка сохранения: " + data.error); return; }
  await loadNext();
}

function undoPoint() { if (points.length > 0) { points.pop(); draw(); } }
function resetPoints() { points = []; draw(); }

async function skipImage() {
  if (!current || current.done) return;
  await fetch("/api/skip", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: current.filename })
  });
  await loadNext();
}
loadNext();
</script>
</body>
</html>
""".replace("__TITLE__", title)
    return Response(html, mimetype="text/html")


@app.route("/api/next")
def api_next():
    pending = get_pending_images()
    if not pending:
        return jsonify({"done": True})
    info = get_image_info(pending[0])
    info["done"] = False
    info["remaining"] = len(pending)
    return jsonify(info)


@app.route("/api/crop/<path:filename>")
def api_crop(filename):
    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return jsonify({"error": "image not found"}), 404

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size
        boxes = read_yolo_boxes(image_path, img_w, img_h)
        box = choose_box(boxes)
        crop_box = (0, 0, img_w, img_h) if box is None else add_padding_to_box(box, img_w, img_h)
        cropped = img.crop(crop_box)
        buffer = BytesIO()
        cropped.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
    return send_file(buffer, mimetype="image/jpeg")


@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    filename = data.get("filename")
    points = data.get("points")
    if not filename or not points or len(points) != 4:
        return jsonify({"ok": False, "error": "bad payload"})

    image_path = IMAGES_DIR / filename
    if not image_path.exists():
        return jsonify({"ok": False, "error": "image not found"})

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img_w, img_h = img.size
        boxes = read_yolo_boxes(image_path, img_w, img_h)
        box = choose_box(boxes)
        crop_x1, crop_y1, _, _ = (0, 0, img_w, img_h) if box is None else add_padding_to_box(box, img_w, img_h)

    normalized = []
    for p in points:
        x_original = max(0, min(img_w - 1, crop_x1 + float(p["x"])))
        y_original = max(0, min(img_h - 1, crop_y1 + float(p["y"])))
        normalized.append(x_original / img_w)
        normalized.append(y_original / img_h)

    out_path = corners_path_for_image(image_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(" ".join(f"{v:.6f}" for v in normalized), encoding="utf-8")
    return jsonify({"ok": True, "saved_to": str(out_path)})


@app.route("/api/skip", methods=["POST"])
def api_skip():
    data = request.get_json(force=True)
    filename = data.get("filename")
    if not filename:
        return jsonify({"ok": False, "error": "bad payload"})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with SKIPPED_FILE.open("a", encoding="utf-8") as f:
        f.write(Path(filename).stem + "\n")
    return jsonify({"ok": True})


def main():
    print(f"Dataset:  {ARGS.dataset}/{ARGS.split}")
    print(f"Папка:    {DATASET_SPLIT_DIR}")
    print(f"Фильтры:  min_image_px={ARGS.min_image_px}, min_bbox_px={ARGS.min_bbox_px}")

    if ARGS.min_bbox_px > 0 or ARGS.min_image_px > 0:
        stats = auto_skip_too_small(ARGS.min_image_px, ARGS.min_bbox_px)
        print(f"Авто-фильтр: уже было в skipped {stats['already_skipped']}, "
              f"добавлено {stats['newly_skipped']}, проходят фильтр {stats['kept']}")

    pending = get_pending_images()
    done = get_done_stems()
    skipped = get_skipped_stems()
    print(f"Всего:    {len(find_images())} | размечено: {len(done)} | "
          f"пропущено: {len(skipped)} | осталось показать: {len(pending)}")
    print(f"\nОткрой на ноутбуке: http://127.0.0.1:{ARGS.port}")
    print(f"С тачскрина — IP мака в Wi-Fi, например: http://192.168.1.10:{ARGS.port}\n")
    app.run(host=ARGS.host, port=ARGS.port, debug=False)


if __name__ == "__main__":
    main()
