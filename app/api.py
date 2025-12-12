from PIL import Image
import io
import base64
import json
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse

from app.services.yolo_service import YoloSegService
from app.services.sam_service import MobileSamService
from app.services.stats_service import compute_void_rates_from_detections

from app.sam.main import get_sam_device

app = FastAPI(title="Electronics Active Learning - Stage 1")

sam_service = MobileSamService()
yolo_service = None

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _read_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


@app.get("/", response_class=HTMLResponse)
async def index():
    """
    Simple static UI for local testing.
    Serves app/frontend/index.html.
    """
    index_path = FRONTEND_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.on_event("startup")
def _startup():
    global yolo_service
    try:
        yolo_service = YoloSegService()
    except Exception as e:
        # Allow running SAM-only even if YOLO deps not ready
        print(f"[WARN] YOLO service not loaded: {e}")


@app.post("/yolo/predict")
async def yolo_predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25),
):
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO service not available.")
    contents = await file.read()
    img = _read_image(contents)
    out = yolo_service.predict(img, conf=conf)
    return JSONResponse(out)


@app.post("/sam/segment-overlay-with-stats")
async def sam_segment_overlay_with_stats(
    file: UploadFile = File(...),
    inside_bbox0: bool = Query(True),
):
    contents = await file.read()
    img = _read_image(contents)

    overlay, stats = sam_service.analyze(img, inside_bbox0=inside_bbox0)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {"image_png_base64": b64, "segments": stats}


@app.post("/analyze/void-rate")
async def analyze_void_rate(
    file: UploadFile = File(...),
    conf: float = Query(0.25),
):
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO service not available.")

    contents = await file.read()
    img = _read_image(contents)

    pred = yolo_service.predict(img, conf=conf)
    rates = compute_void_rates_from_detections(pred)

    return {"prediction": pred, "void_rates": rates}


@app.get("/healthz")
async def healthz():
    """Simple health check with device info."""
    global yolo_service
    yolo_dev = None
    if yolo_service is not None:
        yolo_dev = yolo_service.device

    return {
        "status": "ok",
        "sam_device": get_sam_device(),
        "yolo_device": yolo_dev,
    }


@app.post("/feedback/roi-example")
async def feedback_roi_example(
    file: UploadFile = File(...),
    label: str = Form(...),
    notes: str = Form(""),
):
    """
    Store a user-selected ROI patch + metadata for future (offline) training.

    - Saves the ROI image under feedback/YYYYMMDD/.
    - Writes a JSON sidecar with label, notes, timestamp, and basic image info.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    # Timestamp + date-based directory
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    day_dir = FEEDBACK_DIR / ts[:8]  # e.g. feedback/20251212
    day_dir.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "roi"
    base_stem = Path(original_name).stem

    # Make label filesystem-safe
    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")) or "unlabeled"

    img_filename = f"{base_stem}_{safe_label}_{ts}.png"
    img_path = day_dir / img_filename

    # Save PNG ROI image (and grab size if possible)
    height = width = None
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img.save(img_path, format="PNG")
        height, width = img.height, img.width
    except Exception as e:
        # If PIL fails, at least persist raw bytes for offline inspection
        with open(img_path, "wb") as f:
            f.write(contents)
        print(f"[feedback] Warning: PIL decode failed for ROI: {e}")

    meta = {
        "schema_version": 1,
        "source": "roi-example",
        "timestamp_utc": ts,
        "label": label,
        "notes": notes,
        "original_filename": original_name,
        "saved_image": str(img_path),
        "image_size_hw": [height, width],
    }

    meta_path = day_dir / f"{base_stem}_{safe_label}_{ts}.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "saved_dir": str(day_dir),
        "image_path": str(img_path),
        "meta_path": str(meta_path),
    }

