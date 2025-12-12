#!/usr/bin/env python3
"""
scripts/test_endpoints.py

Small dev client to sanity-check the backend endpoints:

- GET /healthz
- POST /yolo/predict
- POST /sam/segment-overlay-with-stats
- POST /analyze/void-rate

Usage:
    python scripts/test_endpoints.py path/to/image.jpg
"""

import os
import sys
import pathlib
import requests
from typing import Any, Dict


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
# (connect_timeout, read_timeout)
DEFAULT_TIMEOUT = float(os.getenv("API_TIMEOUT", "600.0"))  # seconds
TIMEOUT = (10.0, DEFAULT_TIMEOUT)


def _post_file(endpoint: str, image_path: str) -> Dict[str, Any]:
    """POST an image file to an endpoint and return JSON."""
    url = BASE_URL.rstrip("/") + endpoint
    p = pathlib.Path(image_path)

    with p.open("rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=TIMEOUT)

    resp.raise_for_status()
    return resp.json()


def _get_json(endpoint: str) -> Dict[str, Any]:
    """GET JSON from an endpoint."""
    url = BASE_URL.rstrip("/") + endpoint
    resp = requests.get(url, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def test_image(image_path: str) -> None:
    print(f"\n=== Testing image: {image_path} ===")

    # 1) YOLO
    try:
        yolo = _post_file("/yolo/predict", image_path)
        dets = yolo.get("detections", [])
        print(f"[YOLO] detections: {len(dets)}")
        for d in dets[:5]:
            print(
                f"  - id={d.get('id')} cat={d.get('category')} "
                f"score={d.get('score'):.3f} bbox={d.get('bbox_xyxy')}"
            )
    except requests.exceptions.ReadTimeout:
        print("[YOLO] ERROR: Read timeout.")
        return
    except Exception as e:
        print(f"[YOLO] ERROR: {e}")
        return

    # 2) SAM overlay + stats
    try:
        sam = _post_file("/sam/segment-overlay-with-stats", image_path)
        segments = sam.get("segments", [])
        print(f"[SAM] segments: {len(segments)}")
    except requests.exceptions.ReadTimeout:
        print("[SAM] ERROR: Read timeout (SAM may be too heavy for the current timeout).")
        return
    except Exception as e:
        print(f"[SAM] ERROR: {e}")
        return

    # 3) Void-rate endpoint
    try:
        vr = _post_file("/analyze/void-rate", image_path)
        vrates = vr.get("void_rates", [])
        print(f"[VOID-RATE] chips analyzed: {len(vrates)}")
    except requests.exceptions.ReadTimeout:
        print("[VOID-RATE] ERROR: Read timeout.")
    except Exception as e:
        print(f"[VOID-RATE] ERROR: {e}")


def main(argv):
    if not argv:
        print("Usage: python scripts/test_endpoints.py path/to/image.jpg")
        sys.exit(1)

    image_path = argv[0]
    if not os.path.exists(image_path):
        print(f"ERROR: {image_path} does not exist.")
        sys.exit(1)

    # Health check first
    try:
        health = _get_json("/healthz")
        print("[healthz]", health)
    except Exception as e:
        print(f"[healthz] ERROR: {e}")
        sys.exit(1)

    test_image(image_path)


if __name__ == "__main__":
    main(sys.argv[1:])

