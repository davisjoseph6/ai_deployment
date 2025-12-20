#!/usr/bin/env python3
"""
settings.py

Central configuration for Stage 1 backend.

- Supports local YOLO checkpoint path (default: yolo_+_segmentation/best.pt)
- Optional remote YOLO mode (YOLO_MODE=remote + YOLO_ENDPOINT)
- Optional SAM enable/disable (ENABLE_SAM)
"""

import os
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = ROOT / "ai_model_artifacts"
ARTIFACTS_DIR = Path(os.getenv("AI_ARTIFACTS_DIR", DEFAULT_ARTIFACTS))

# --- Modes ---
YOLO_MODE = os.getenv("YOLO_MODE", "local").strip().lower()  # local|remote
YOLO_ENDPOINT = os.getenv("YOLO_ENDPOINT", "").strip()       # used if remote

# --- Model paths ---
SAM_CHECKPOINT = Path(
    os.getenv("SAM_CHECKPOINT", str(ARTIFACTS_DIR / "segmentation_model" / "mobile_sam.pt"))
)

# Default to the combined model you want to deploy
YOLO_CHECKPOINT = Path(
    os.getenv("YOLO_CHECKPOINT", str(ARTIFACTS_DIR / "yolo_+_segmentation" / "best.pt"))
)

# --- Feature flags ---
ENABLE_SAM = _env_bool("ENABLE_SAM", default=False)

# --- Class names used in void-rate logic ---
CHIP_CLASS = os.getenv("CHIP_CLASS", "chip")
HOLE_CLASS = os.getenv("HOLE_CLASS", "hole")

# --- Remote auth (for Azure ML endpoint, etc.) ---
YOLO_AUTH_HEADER = os.getenv("YOLO_AUTH_HEADER", "Authorization").strip()
YOLO_AUTH_PREFIX = os.getenv("YOLO_AUTH_PREFIX", "Bearer").strip()
YOLO_AUTH_TOKEN = os.getenv("YOLO_AUTH_TOKEN", "").strip()

