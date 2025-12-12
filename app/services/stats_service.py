#!/usr/bin/env python3
"""
stats_service.py

Utilities for computing areas and void rates from YOLO detections.

Expected classes:
  - chip (component)
  - hole (void)
"""

from typing import Dict, Any, List
import numpy as np
import cv2


def _polygon_to_mask(poly, h: int, w: int) -> np.ndarray:
    """Rasterize polygon points into a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if poly is None or len(poly) < 3:
        return mask

    pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], 1)
    return mask


def _centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return (-1, -1)
    return (int(np.round(xs.mean())), int(np.round(ys.mean())))


def compute_void_rates_from_detections(
    pred: Dict[str, Any],
    chip_class: str = "chip",
    hole_class: str = "hole",
) -> List[Dict[str, Any]]:
    """
    Compute per-chip void rate:
        void_rate = total_void_area_inside_chip / chip_area
    """
    h, w = pred["image_shape"]
    dets = pred.get("detections", [])

    chips = [d for d in dets if d.get("category") == chip_class]
    holes = [d for d in dets if d.get("category") == hole_class]

    chip_masks = [
        _polygon_to_mask(c.get("polygon"), h, w) for c in chips
    ]
    hole_masks = [
        _polygon_to_mask(v.get("polygon"), h, w) for v in holes
    ]
    hole_centroids = [_centroid(m) for m in hole_masks]

    results: List[Dict[str, Any]] = []

    for i, cm in enumerate(chip_masks):
        comp_area = int(cm.sum())
        if comp_area == 0:
            results.append({
                "chip_id": i,
                "component_area_px": 0,
                "void_area_px": 0,
                "void_rate": 0.0,
                "hole_ids": [],
            })
            continue

        void_area = 0
        hole_ids = []

        for j, hm in enumerate(hole_masks):
            cx, cy = hole_centroids[j]
            if cx < 0:
                continue

            # centroid-in-chip heuristic
            if cm[cy, cx] == 1:
                inter = int((hm & cm).sum())
                if inter > 0:
                    void_area += inter
                    hole_ids.append(j)

        results.append({
            "chip_id": i,
            "component_area_px": comp_area,
            "void_area_px": void_area,
            "void_rate": float(void_area) / float(comp_area),
            "hole_ids": hole_ids,
        })

    return results

