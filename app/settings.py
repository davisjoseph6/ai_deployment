#!/usr/bin/env python3
"""
settings.py

Central configuration for Stage 1 local backend.
Loads model artifact paths via environment variables with safe defaults.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# If you renamed the folder, this will be correct:
DEFAULT_ARTIFACTS = ROOT / "ai_model_artifacts"

# If you did NOT rename, override by exporting:
# export AI_ARTIFACTS_DIR=~/ai_deployment/ai_model_artifiacts
ARTIFACTS_DIR = Path(os.getenv("AI_ARTIFACTS_DIR", DEFAULT_ARTIFACTS))

SAM_CHECKPOINT = Path(
    os.getenv(
        "SAM_CHECKPOINT",
        ARTIFACTS_DIR / "segmentation_model" / "mobile_sam.pt"
    )
)

YOLO_CHECKPOINT = Path(
    os.getenv(
        "YOLO_CHECKPOINT",
        ARTIFACTS_DIR / "yolo_model" / "optimal_model.pt"
    )
)

