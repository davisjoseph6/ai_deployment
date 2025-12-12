#!/usr/bin/env python3
"""
sam_service.py

MobileSAM wrapper for Stage 1.

Assumes your previous code was copied into:
  - app/sam/main.py
  - app/sam/tools.py
"""

from PIL import Image
from typing import Tuple, List, Dict, Any

from app.sam.main import (
    segment_everything,
    analyze_segments,
    analyze_segments_in_bbox0,
)


class MobileSamService:
    """Thin wrapper around your MobileSAM pipeline."""

    def segment(self, image: Image.Image):
        return segment_everything(image)

    def analyze(self, image: Image.Image, inside_bbox0: bool = True):
        if inside_bbox0:
            overlay, stats = analyze_segments_in_bbox0(image)
        else:
            overlay, stats = analyze_segments(image)
        return overlay, stats

