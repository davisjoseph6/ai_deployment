#!/usr/bin/env python3
"""
api.py

Unified Stage 1 local backend connecting:
  - YOLO auto prediction
  - MobileSAM correction/analysis
  - Void-rate computation
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64

from app.services.yolo_service import YoloSegService
from app.services.sam_service import MobileSamService
from app.services.stats_service import compute_void_rates_from_detections

from app.sam.main import get_sam_device

app = FastAPI(title="Electronics Active Learning - Stage 1")

sam_service = MobileSamService()
yolo_service = None


def _read_image(contents: bytes) -> Image.Image:
    return Image.open(io.BytesIO(contents)).convert("RGB")


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
    conf: float = Query(0.25)
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
    inside_bbox0: bool = Query(True)
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
    conf: float = Query(0.25)
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


