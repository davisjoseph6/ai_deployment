#!/usr/bin/env python3
"""
yolo_service.py

Local YOLO segmentation wrapper for Stage 1.

Loads a YOLO segmentation checkpoint and exposes predict() returning a
normalized JSON bundle usable by the app pipeline.
"""

from typing import Any, Dict, List, Optional
import os

from PIL import Image
import torch

from app.settings import YOLO_CHECKPOINT

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


def _get_yolo_device() -> str:
    """
    Decide which device string to pass to Ultralytics:

    1) If YOLO_DEVICE or DEVICE is 'cpu' -> force CPU.
       If 'cuda' -> try GPU, but validate capability.
    2) Else, auto: if a *supported* CUDA GPU exists, use it; otherwise 'cpu'.
    """
    env = os.getenv("YOLO_DEVICE") or os.getenv("DEVICE")
    requested = None
    if env:
        env = env.strip().lower()
        if env in ("cpu", "cuda", "cuda:0"):
            requested = env
            print(f"[YOLO] Device requested from env: {env}")
        else:
            print(f"[YOLO] Ignoring unsupported YOLO_DEVICE={env}, auto-selecting.")

    if requested == "cpu":
        print("[YOLO] Forcing CPU because YOLO_DEVICE=cpu")
        return "cpu"

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            major, minor = torch.cuda.get_device_capability(0)
            print(f"[YOLO] Detected CUDA device {name} (sm_{major}{minor})")

            if major < 7:
                print(
                    f"[YOLO] WARNING: GPU sm_{major}{minor} is not supported by this "
                    "PyTorch build. Using 'cpu' instead."
                )
                return "cpu"

            if requested in ("cuda", "cuda:0") or requested is None:
                print("[YOLO] Using 'cuda' for YOLO segmentation.")
                return "cuda"

        except Exception as e:
            print(f"[YOLO] CUDA available but capability query failed ({e}); using 'cpu'.")

    print("[YOLO] Using 'cpu'.")
    return "cpu"

class YoloSegService:
    """Thin wrapper around a YOLO segmentation model."""

    def __init__(self, checkpoint: str = None, device: Optional[str] = None):
        if YOLO is None:
            raise ImportError(
                "ultralytics is required. Install with: pip install ultralytics"
            )

        ckpt = checkpoint or str(YOLO_CHECKPOINT)
        self.device = device or _get_yolo_device()

        print(f"[YOLO] Loading model from: {ckpt}")
        self.model = YOLO(ckpt)
        # Ultralytics will honor device argument in predict(); this is extra safety:
        try:
            self.model.to(self.device)
        except Exception as e:
            print(f"[YOLO] Warning: model.to({self.device}) failed: {e}")

        # class names mapping if available
        self.names = getattr(self.model, "names", {})

    def predict(self, image: Image.Image, conf: float = 0.25) -> Dict[str, Any]:
        """Run inference and return a normalized JSON structure."""
        img = image.convert("RGB")

        results = self.model.predict(
            img, conf=conf, device=self.device, verbose=False
        )
        r = results[0]

        boxes = getattr(r, "boxes", None)
        masks = getattr(r, "masks", None)

        detections: List[Dict[str, Any]] = []

        if boxes is None:
            return {
                "model": {"name": "yolo-seg", "version": YOLO_CHECKPOINT.name},
                "image_shape": [img.height, img.width],
                "detections": [],
            }

        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        # Polygons, if available (Ultralytics provides masks.xy)
        polygons = []
        if masks is not None and hasattr(masks, "xy"):
            polygons = masks.xy
        else:
            polygons = [None] * len(xyxy)

        for i in range(len(xyxy)):
            cls_id = int(clses[i])
            category = self.names.get(cls_id, str(cls_id))
            detections.append({
                "id": i,
                "category": category,
                "class_id": cls_id,
                "score": float(scores[i]),
                "bbox_xyxy": [float(x) for x in xyxy[i].tolist()],
                "polygon": polygons[i] if polygons[i] is not None else None,
            })

        return {
            "model": {"name": "yolo-seg", "version": YOLO_CHECKPOINT.name},
            "image_shape": [img.height, img.width],
            "detections": detections,
        }

