#!/usr/bin/env python3
"""
api.py

Stage 1 backend API.

Provides:
- YOLO segmentation inference (/yolo/predict)   [local or remote via YOLO_MODE]
- Optional MobileSAM overlay+stats (/sam/segment-overlay-with-stats) [ENABLE_SAM]
- Optional SAM refinement (/sam/refine) [ENABLE_SAM]
- Void-rate computation (/analyze/void-rate)
- Feedback capture endpoint (/feedback/roi-example)
- Health check (/healthz)

Uses response_model contracts for strict interface validation.
"""

import base64
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from anyio import to_thread
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Form
from fastapi.responses import HTMLResponse
from PIL import Image

from app.schemas import (
    YoloPredictResponse,
    VoidRateResponse,
    SamSegmentResponse,
    SamRefineResponse,
    HealthzResponse,
)
from app.settings import ENABLE_SAM, CHIP_CLASS, HOLE_CLASS, YOLO_MODE
from app.services.yolo_service import build_yolo_service
from app.services.sam_service import MobileSamService
from app.services.stats_service import compute_void_rates_from_detections

app = FastAPI(title="Electronics Active Learning - Stage 1")

# Services (initialized on startup)
yolo_service = None
sam_service: Optional[MobileSamService] = None

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
FEEDBACK_DIR = Path("feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


def _read_image(contents: bytes) -> Image.Image:
    """Decode bytes -> PIL Image (RGB)."""
    try:
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")


@app.on_event("startup")
def _startup() -> None:
    """Initialize services once at startup."""
    global yolo_service, sam_service

    try:
        yolo_service = build_yolo_service()
    except Exception as e:
        yolo_service = None
        print(f"[WARN] YOLO service not loaded: {e}")

    if ENABLE_SAM:
        try:
            sam_service = MobileSamService()
        except Exception as e:
            sam_service = None
            print(f"[WARN] SAM service not loaded: {e}")
    else:
        sam_service = None


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serves app/frontend/index.html."""
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found.")
    return index_path.read_text(encoding="utf-8")


@app.post("/yolo/predict", response_model=YoloPredictResponse)
async def yolo_predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25),
):
    """Run YOLO segmentation inference (local or remote)."""
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO service not available.")

    contents = await file.read()
    img = _read_image(contents)

    try:
        out = await to_thread.run_sync(yolo_service.predict, img, conf)
        return out
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")


@app.post("/sam/segment-overlay-with-stats", response_model=SamSegmentResponse)
async def sam_segment_overlay_with_stats(
    file: UploadFile = File(...),
    inside_bbox0: bool = Query(True),
):
    """Run SAM segmentation overlay + stats (only if ENABLE_SAM=true)."""
    if sam_service is None:
        raise HTTPException(status_code=501, detail="SAM is disabled in this deployment.")

    contents = await file.read()
    img = _read_image(contents)

    try:
        overlay, stats = await to_thread.run_sync(sam_service.analyze, img, inside_bbox0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM analysis failed: {e}")

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return {"image_png_base64": b64, "segments": stats}


@app.post("/sam/refine", response_model=SamRefineResponse)
async def sam_refine(
    file: UploadFile = File(...),
    x1: float = Form(...),
    y1: float = Form(...),
    x2: float = Form(...),
    y2: float = Form(...),
    points_json: str = Form(""),
    labels_json: str = Form(""),
):
    """Refine a region using SAM predictor with a bbox prompt."""
    if sam_service is None:
        raise HTTPException(status_code=501, detail="SAM is disabled in this deployment.")

    contents = await file.read()
    img = _read_image(contents)

    if x2 <= x1 or y2 <= y1:
        raise HTTPException(status_code=422, detail="Invalid bbox: x2/y2 must be > x1/y1.")

    bbox = [x1, y1, x2, y2]

    point_coords = None
    point_labels = None

    if points_json.strip():
        try:
            point_coords = json.loads(points_json)
        except Exception:
            raise HTTPException(status_code=400, detail="points_json is not valid JSON.")
    if labels_json.strip():
        try:
            point_labels = json.loads(labels_json)
        except Exception:
            raise HTTPException(status_code=400, detail="labels_json is not valid JSON.")

    try:
        poly, score, area_px = await to_thread.run_sync(
            sam_service.refine, img, bbox, point_coords, point_labels
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM refine failed: {e}")

    if len(poly) < 3 or area_px <= 0:
        raise HTTPException(status_code=422, detail="SAM refinement produced empty mask.")

    return {"bbox_xyxy": bbox, "polygon": poly, "score": score, "area_px": area_px}


@app.post("/analyze/void-rate", response_model=VoidRateResponse)
async def analyze_void_rate(
    file: UploadFile = File(...),
    conf: float = Query(0.25),
):
    """Run YOLO + compute void-rates per chip."""
    if yolo_service is None:
        raise HTTPException(status_code=500, detail="YOLO service not available.")

    contents = await file.read()
    img = _read_image(contents)

    try:
        pred = await to_thread.run_sync(yolo_service.predict, img, conf)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference failed: {e}")

    try:
        rates = compute_void_rates_from_detections(pred, chip_class=CHIP_CLASS, hole_class=HOLE_CLASS)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Void-rate computation failed: {e}")

    return {"prediction": pred, "void_rates": rates}


@app.get("/healthz", response_model=HealthzResponse)
async def healthz():
    """Health check with mode/device info."""
    yolo_dev = getattr(yolo_service, "device", None)

    sam_dev = "disabled"
    if ENABLE_SAM and sam_service is not None:
        try:
            sam_dev = sam_service.get_device()
        except Exception:
            sam_dev = "unknown"

    return {
        "status": "ok",
        "sam_device": sam_dev,
        "yolo_device": yolo_dev,
        "sam_enabled": bool(ENABLE_SAM),
        "yolo_mode": str(YOLO_MODE),
    }


@app.post("/feedback/roi-example")
async def feedback_roi_example(
    file: UploadFile = File(...),
    label: str = Form(...),
    notes: str = Form(""),
):
    """Store a user-selected ROI patch + metadata for future (offline) training."""
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    day_dir = FEEDBACK_DIR / ts[:8]
    day_dir.mkdir(parents=True, exist_ok=True)

    original_name = file.filename or "roi"
    base_stem = Path(original_name).stem

    safe_label = "".join(c for c in label if c.isalnum() or c in ("-", "_")) or "unlabeled"

    img_filename = f"{base_stem}_{safe_label}_{ts}.png"
    img_path = day_dir / img_filename

    height = width = None
    try:
        im = Image.open(io.BytesIO(contents)).convert("RGB")
        im.save(img_path, format="PNG")
        height, width = im.height, im.width
    except Exception as e:
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

    return {"status": "ok", "saved_dir": str(day_dir), "image_path": str(img_path), "meta_path": str(meta_path)}

