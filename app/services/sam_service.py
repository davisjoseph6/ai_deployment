#!/usr/bin/env python3
"""
sam_service.py

MobileSAM wrapper.

IMPORTANT:
This module does NOT import app.sam.main at import time.
All heavy imports happen only when MobileSamService() is instantiated.
"""

from typing import List, Optional, Tuple
from PIL import Image


class MobileSamService:
    """Thin wrapper around the MobileSAM pipeline (lazy-loaded)."""

    def __init__(self):
        # Heavy imports happen here only when SAM is enabled
        from app.sam.main import (
            segment_everything,
            analyze_segments,
            analyze_segments_in_bbox0,
            refine_mask_with_box_prompt,
            get_sam_device,
        )
        from app.sam.tools import mask_to_polygon

        self._segment_everything = segment_everything
        self._analyze_segments = analyze_segments
        self._analyze_segments_in_bbox0 = analyze_segments_in_bbox0
        self._refine_mask_with_box_prompt = refine_mask_with_box_prompt
        self._mask_to_polygon = mask_to_polygon
        self._get_sam_device = get_sam_device

    def get_device(self) -> str:
        return str(self._get_sam_device())

    def segment(self, image: Image.Image):
        return self._segment_everything(image)

    def analyze(self, image: Image.Image, inside_bbox0: bool = True):
        if inside_bbox0:
            overlay, stats = self._analyze_segments_in_bbox0(image)
        else:
            overlay, stats = self._analyze_segments(image)
        return overlay, stats

    def refine(
        self,
        image: Image.Image,
        bbox_xyxy: List[float],
        point_coords: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
    ) -> Tuple[List[List[float]], float, int]:
        out = self._refine_mask_with_box_prompt(
            image=image,
            bbox_xyxy=bbox_xyxy,
            point_coords=point_coords,
            point_labels=point_labels,
        )
        mask = out["mask"]
        poly = self._mask_to_polygon(mask)
        score = float(out["score"])
        area_px = int(mask.sum())
        return poly, score, area_px

