#!/usr/bin/env python3
"""
tests/test_contracts.py

Contract tests for Stage 1 endpoints.

These tests:
- Verify endpoints are reachable
- Validate JSON contracts (including strict schema via Pydantic models)
"""

import os
import pytest
import requests

from app.schemas import YoloPredictResponse, VoidRateResponse, SamSegmentResponse

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


def _server_available() -> bool:
    try:
        r = requests.get(BASE_URL + "/healthz", timeout=(2, 5))
        return r.ok
    except Exception:
        return False


def test_healthz():
    if not _server_available():
        pytest.skip("API server not running; start uvicorn to run contract tests.")

    r = requests.get(BASE_URL + "/healthz", timeout=(5, 30))
    r.raise_for_status()
    j = r.json()
    assert j["status"] == "ok"
    assert "sam_device" in j


def test_yolo_contract(test_image_path="test_images/board_01.jpg"):
    if not _server_available():
        pytest.skip("API server not running; start uvicorn to run contract tests.")
    if not os.path.exists(test_image_path):
        pytest.skip("test image not available in this environment.")

    with open(test_image_path, "rb") as f:
        files = {"file": ("board_01.jpg", f, "image/jpeg")}
        r = requests.post(BASE_URL + "/yolo/predict", files=files, timeout=(10, 600))
    r.raise_for_status()
    j = r.json()

    # Strict contract validation
    YoloPredictResponse(**j)

    # ensure polygons are JSON-safe lists (not numpy)
    for d in j.get("detections", [])[:3]:
        if d.get("polygon") is not None:
            assert isinstance(d["polygon"], list)
            assert len(d["polygon"]) > 0
            assert isinstance(d["polygon"][0], list)


def test_sam_contract(test_image_path="test_images/board_01.jpg"):
    if not _server_available():
        pytest.skip("API server not running; start uvicorn to run contract tests.")
    if not os.path.exists(test_image_path):
        pytest.skip("test image not available in this environment.")

    with open(test_image_path, "rb") as f:
        files = {"file": ("board_01.jpg", f, "image/jpeg")}
        r = requests.post(
            BASE_URL + "/sam/segment-overlay-with-stats",
            files=files,
            timeout=(10, 600),
        )
    r.raise_for_status()
    j = r.json()

    SamSegmentResponse(**j)
    assert isinstance(j["image_png_base64"], str)
    assert isinstance(j["segments"], list)


def test_void_rate_contract(test_image_path="test_images/board_01.jpg"):
    if not _server_available():
        pytest.skip("API server not running; start uvicorn to run contract tests.")
    if not os.path.exists(test_image_path):
        pytest.skip("test image not available in this environment.")

    with open(test_image_path, "rb") as f:
        files = {"file": ("board_01.jpg", f, "image/jpeg")}
        r = requests.post(BASE_URL + "/analyze/void-rate", files=files, timeout=(10, 600))
    r.raise_for_status()
    j = r.json()

    VoidRateResponse(**j)
    assert "prediction" in j and "void_rates" in j

