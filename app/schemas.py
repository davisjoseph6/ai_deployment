#!/usr/bin/env python3
"""
schemas.py

Pydantic response/contract models for Stage 1.

Strict contracts (extra keys forbidden).
Designed to work with both Pydantic v1 and v2.
"""

from typing import List, Optional, Type
from pydantic import BaseModel

try:
    from pydantic import ConfigDict  # v2
except Exception:
    ConfigDict = None  # type: ignore


BBoxXYXY: Type
ShapeHW: Type
BBoxXYWH: Type
HSV3: Type

try:
    from pydantic import conlist  # type: ignore
    try:
        BBoxXYXY = conlist(float, min_length=4, max_length=4)  # type: ignore
        ShapeHW = conlist(int, min_length=2, max_length=2)  # type: ignore
        BBoxXYWH = conlist(float, min_length=4, max_length=4)  # type: ignore
        HSV3 = conlist(int, min_length=3, max_length=3)  # type: ignore
    except TypeError:
        BBoxXYXY = conlist(float, min_items=4, max_items=4)  # type: ignore
        ShapeHW = conlist(int, min_items=2, max_items=2)  # type: ignore
        BBoxXYWH = conlist(float, min_items=4, max_items=4)  # type: ignore
        HSV3 = conlist(int, min_items=3, max_items=3)  # type: ignore
except Exception:
    BBoxXYXY = List[float]
    ShapeHW = List[int]
    BBoxXYWH = List[float]
    HSV3 = List[int]


class StrictBaseModel(BaseModel):
    """Base model that forbids extra keys (contract strictness)."""

    if ConfigDict is not None:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"


# ----------------------------
# YOLO contracts
# ----------------------------
class ModelInfo(StrictBaseModel):
    name: str
    version: str


class Detection(StrictBaseModel):
    id: int
    category: str
    class_id: int
    score: float
    bbox_xyxy: BBoxXYXY
    polygon: Optional[List[List[float]]] = None


class YoloPredictResponse(StrictBaseModel):
    model: ModelInfo
    image_shape: ShapeHW
    detections: List[Detection]
    timing_ms: Optional[float] = None


# ----------------------------
# Void-rate contracts
# ----------------------------
class VoidRateItem(StrictBaseModel):
    chip_id: int
    component_area_px: int
    void_area_px: int
    void_rate: float
    hole_ids: List[int]


class VoidRateResponse(StrictBaseModel):
    prediction: YoloPredictResponse
    void_rates: List[VoidRateItem]


# ----------------------------
# SAM contracts
# ----------------------------
class SamSegmentStat(StrictBaseModel):
    id: int
    pixels: int
    bbox_xywh: Optional[BBoxXYWH] = None
    color_class: str
    mean_hsv: HSV3


class SamSegmentResponse(StrictBaseModel):
    image_png_base64: str
    segments: List[SamSegmentStat]


class SamRefineResponse(StrictBaseModel):
    bbox_xyxy: BBoxXYXY
    polygon: List[List[float]]
    score: float
    area_px: int


# ----------------------------
# Health check contracts
# ----------------------------
class HealthzResponse(StrictBaseModel):
    status: str
    sam_device: str
    yolo_device: Optional[str] = None
    sam_enabled: bool
    yolo_mode: str

