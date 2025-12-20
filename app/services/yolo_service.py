#!/usr/bin/env python3
"""
yolo_service.py

YOLO segmentation service supporting:
- local inference via Ultralytics (YOLO_MODE=local)
- remote inference via HTTP (YOLO_MODE=remote)

Key design goal:
- In remote mode, this module MUST NOT require torch/ultralytics to be installed.
  Heavy imports happen only inside LocalYoloSegService.
"""

import io
import os
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from app.settings import (
    YOLO_CHECKPOINT,
    YOLO_MODE,
    YOLO_ENDPOINT,
    YOLO_AUTH_HEADER,
    YOLO_AUTH_PREFIX,
    YOLO_AUTH_TOKEN,
)


def _auth_headers() -> Dict[str, str]:
    """Build auth headers for remote inference, if configured."""
    token = (YOLO_AUTH_TOKEN or "").strip()
    if not token:
        return {}
    header = (YOLO_AUTH_HEADER or "Authorization").strip()
    prefix = (YOLO_AUTH_PREFIX or "Bearer").strip()
    return {header: f"{prefix} {token}".strip()}


def _poly_to_python_list(poly) -> Optional[List[List[float]]]:
    if poly is None:
        return None
    try:
        pts = poly.tolist()
        if not pts:
            return None
        return [[float(x), float(y)] for x, y in pts]
    except Exception:
        return None


def _normalize_prediction(pred: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce the contract expected by YoloPredictResponse.
    If the remote endpoint already returns the exact contract, this is a no-op
    (except it strips unexpected fields).
    """
    model = pred.get("model") or {}
    image_shape = pred.get("image_shape")

    dets_in = pred.get("detections") or []
    dets_out: List[Dict[str, Any]] = []

    for i, d in enumerate(dets_in):
        if not isinstance(d, dict):
            continue
        bbox = d.get("bbox_xyxy")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            # If bbox missing/invalid, skip to avoid response validation failure
            continue

        poly = d.get("polygon")
        if poly is not None and not isinstance(poly, list):
            poly = None

        dets_out.append(
            {
                "id": int(d.get("id", i)),
                "category": str(d.get("category", "")),
                "class_id": int(d.get("class_id", 0)),
                "score": float(d.get("score", 0.0)),
                "bbox_xyxy": [float(x) for x in bbox],
                "polygon": poly,
            }
        )

    out = {
        "model": {
            "name": str(model.get("name", "yolo-seg")),
            "version": str(model.get("version", "remote")),
        },
        "image_shape": image_shape if isinstance(image_shape, list) else [0, 0],
        "detections": dets_out,
        "timing_ms": pred.get("timing_ms", None),
    }
    return out


class LocalYoloSegService:
    """Local Ultralytics YOLO segmentation wrapper (requires torch + ultralytics)."""

    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None):
        # Heavy imports only here
        import torch
        from ultralytics import YOLO  # type: ignore

        ckpt = checkpoint or str(YOLO_CHECKPOINT)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"YOLO checkpoint not found: {ckpt}")

        self.device = device or self._get_yolo_device(torch)
        self.model = YOLO(ckpt)

        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.names = getattr(self.model, "names", {})

    @staticmethod
    def _get_yolo_device(torch_mod) -> str:
        """
        Decide which device to pass to Ultralytics.

        - If YOLO_DEVICE/DEVICE=cpu => cpu
        - Else CUDA only if available AND compute capability >= 7.0
        - Otherwise cpu
        """
        env = (os.getenv("YOLO_DEVICE") or os.getenv("DEVICE") or "").strip().lower()
        if env == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            return "cpu"

        if getattr(torch_mod, "cuda", None) is not None and torch_mod.cuda.is_available():
            try:
                major, minor = torch_mod.cuda.get_device_capability(0)
                if major < 7:
                    return "cpu"
                return "cuda"
            except Exception:
                return "cpu"

        return "cpu"

    def _name_for_class(self, cls_id: int) -> str:
        try:
            if hasattr(self.names, "get"):
                return str(self.names.get(cls_id, str(cls_id)))
        except Exception:
            pass
        return str(cls_id)

    def predict(self, image: Image.Image, conf: float = 0.25) -> Dict[str, Any]:
        t0 = time.perf_counter()
        img = image.convert("RGB")

        results = self.model.predict(img, conf=conf, device=self.device, verbose=False)

        if not results:
            return {
                "model": {"name": "yolo-seg", "version": os.path.basename(str(YOLO_CHECKPOINT))},
                "image_shape": [img.height, img.width],
                "detections": [],
                "timing_ms": (time.perf_counter() - t0) * 1000.0,
            }

        r = results[0]
        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return {
                "model": {"name": "yolo-seg", "version": os.path.basename(str(YOLO_CHECKPOINT))},
                "image_shape": [img.height, img.width],
                "detections": [],
                "timing_ms": (time.perf_counter() - t0) * 1000.0,
            }

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        polygons: List[Optional[List[List[float]]]] = [None] * len(xyxy)
        if masks is not None and hasattr(masks, "xy") and masks.xy is not None:
            raw = masks.xy
            for i in range(min(len(raw), len(polygons))):
                polygons[i] = _poly_to_python_list(raw[i])

        detections: List[Dict[str, Any]] = []
        for i in range(len(xyxy)):
            cls_id = int(clses[i])
            detections.append(
                {
                    "id": i,
                    "category": self._name_for_class(cls_id),
                    "class_id": cls_id,
                    "score": float(scores[i]),
                    "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
                    "polygon": polygons[i],
                }
            )

        return {
            "model": {"name": "yolo-seg", "version": os.path.basename(str(YOLO_CHECKPOINT))},
            "image_shape": [img.height, img.width],
            "detections": detections,
            "timing_ms": (time.perf_counter() - t0) * 1000.0,
        }


class RemoteYoloSegService:
    """
    Calls a remote inference endpoint.
    Recommended: have MLOps expose the SAME contract as /yolo/predict returns,
    so the app stays stable.
    """

    def __init__(self, endpoint: str, timeout: tuple = (10.0, 600.0)):
        ep = endpoint.strip().rstrip("/")
        if not ep:
            raise ValueError("YOLO_ENDPOINT is empty but YOLO_MODE=remote.")
        self.device = "remote"
        self.timeout = timeout

        # Accept either base URL or full route
        if ep.endswith("/yolo/predict") or ep.endswith("/score"):
            self.predict_url = ep
        else:
            self.predict_url = ep + "/yolo/predict"

        self.headers = _auth_headers()

    def predict(self, image: Image.Image, conf: float = 0.25) -> Dict[str, Any]:
        img = image.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        payload = buf.getvalue()

        files = {"file": ("image.png", payload, "image/png")}
        resp = requests.post(
            self.predict_url,
            params={"conf": conf},
            files=files,
            headers=self.headers,
            timeout=self.timeout,
        )
        if not resp.ok:
            raise RuntimeError(f"Remote YOLO failed: HTTP {resp.status} {resp.text[:500]}")

        data = resp.json()
        return _normalize_prediction(data)


def build_yolo_service():
    """Factory: returns local or remote YOLO service based on settings/env."""
    mode = (YOLO_MODE or "local").strip().lower()
    if mode == "remote":
        return RemoteYoloSegService(YOLO_ENDPOINT)
    return LocalYoloSegService()

