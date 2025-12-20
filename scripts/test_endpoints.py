#!/usr/bin/env python3
"""
scripts/test_endpoints.py

CLI smoke client to sanity-check the backend endpoints (NOT a pytest test module).

Endpoints:
- GET  /healthz
- POST /yolo/predict
- POST /sam/segment-overlay-with-stats
- POST /analyze/void-rate
- POST /feedback/roi-example

Usage:
    python scripts/test_endpoints.py path/to/image1.jpg [path/to/image2.png ...]

Environment variables:
    API_BASE_URL       (default: "http://127.0.0.1:8000")
    API_TIMEOUT        (default: "600.0" seconds read timeout)
    MAX_PRINT_DETS     (default: "5")  – max YOLO dets to print
    MAX_PRINT_SEGMENTS (default: "5")  – max SAM segments to print
    MAX_PRINT_CHIPS    (default: "5")  – max void-rate chip summaries to print
"""

# IMPORTANT:
# Prevent pytest from collecting this module as tests, even though the filename
# starts with "test_".
__test__ = False  # pytest will not collect tests from this module.

import os
import sys
import pathlib
from typing import Any, Dict

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("API_TIMEOUT", "600.0"))
TIMEOUT = (10.0, DEFAULT_TIMEOUT)  # (connect_timeout, read_timeout)

MAX_PRINT_DETS = int(os.getenv("MAX_PRINT_DETS", "5"))
MAX_PRINT_SEGMENTS = int(os.getenv("MAX_PRINT_SEGMENTS", "5"))
MAX_PRINT_CHIPS = int(os.getenv("MAX_PRINT_CHIPS", "5"))


def _post_file(endpoint: str, image_path: str) -> Dict[str, Any]:
    """POST an image file to an endpoint and return JSON."""
    url = BASE_URL + endpoint
    p = pathlib.Path(image_path)

    with p.open("rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=TIMEOUT)

    resp.raise_for_status()
    return resp.json()


def _get_json(endpoint: str) -> Dict[str, Any]:
    """GET JSON from an endpoint."""
    url = BASE_URL + endpoint
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def _smoke_yolo(image_path: str) -> None:
    """Smoke test /yolo/predict."""
    yolo = _post_file("/yolo/predict", image_path)
    dets = yolo.get("detections", [])
    tms = yolo.get("timing_ms")
    if tms is not None:
        print(f"[YOLO] timing_ms={float(tms):.2f}")
    print(f"[YOLO] detections: {len(dets)}")

    for d in dets[:MAX_PRINT_DETS]:
        score = float(d.get("score", 0.0) or 0.0)
        print(
            "  - id={id} cat={cat} score={score:.3f} bbox={bbox}".format(
                id=d.get("id"),
                cat=d.get("category"),
                score=score,
                bbox=d.get("bbox_xyxy"),
            )
        )


def _smoke_sam(image_path: str) -> None:
    """Smoke test /sam/segment-overlay-with-stats."""
    sam = _post_file("/sam/segment-overlay-with-stats", image_path)
    segments = sam.get("segments", [])
    print(f"[SAM] segments: {len(segments)}")

    for s in segments[:MAX_PRINT_SEGMENTS]:
        print(
            "  - id={id} pixels={px} color={cc} bbox={bbox}".format(
                id=s.get("id"),
                px=s.get("pixels"),
                cc=s.get("color_class"),
                bbox=s.get("bbox_xywh"),
            )
        )


def _smoke_void_rate(image_path: str) -> None:
    """Smoke test /analyze/void-rate."""
    vr = _post_file("/analyze/void-rate", image_path)
    vrates = vr.get("void_rates", [])
    print(f"[VOID-RATE] chips analyzed: {len(vrates)}")

    for r in vrates[:MAX_PRINT_CHIPS]:
        chip_id = r.get("chip_id")
        void_rate = float(r.get("void_rate", 0.0) or 0.0)
        hole_ids = r.get("hole_ids")
        print(
            "  - chip_id={cid} void_rate={vr:.4f} hole_ids={holes}".format(
                cid=chip_id,
                vr=void_rate,
                holes=hole_ids,
            )
        )


def _smoke_feedback(image_path: str) -> None:
    """Smoke test /feedback/roi-example using the full image as a dummy ROI."""
    url = BASE_URL + "/feedback/roi-example"
    p = pathlib.Path(image_path)

    print("[FEEDBACK] posting full image as ROI feedback...")
    with p.open("rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        data = {
            "label": "bubble",
            "notes": "smoke feedback from scripts/test_endpoints.py",
        }
        resp = requests.post(url, files=files, data=data, timeout=TIMEOUT)

    resp.raise_for_status()
    payload = resp.json()
    saved_dir = payload.get("saved_dir") or payload.get("status")
    print(f"[FEEDBACK] OK, feedback stored. Info: {saved_dir}")


def run_on_image(image_path: str) -> None:
    """Run the full smoke test battery on a single image."""
    print(f"\n=== Testing image: {image_path} ===")
    _smoke_yolo(image_path)
    _smoke_sam(image_path)
    _smoke_void_rate(image_path)
    _smoke_feedback(image_path)


def main(argv) -> None:
    """Entry point for CLI usage."""
    if not argv:
        print(
            "Usage: python scripts/test_endpoints.py "
            "path/to/image1.jpg [path/to/image2.png ...]"
        )
        sys.exit(1)

    health = _get_json("/healthz")
    print("[healthz]", health)

    for image_path in argv:
        if not os.path.exists(image_path):
            print(f"ERROR: {image_path} does not exist, skipping.")
            continue
        run_on_image(image_path)


if __name__ == "__main__":
    main(sys.argv[1:])

