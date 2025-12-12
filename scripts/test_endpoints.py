#!/usr/bin/env python3
"""
Simple local test client for the FastAPI backend (Stage 1).

- Sends one or more images to:
    - /yolo/predict
    - /sam/segment-overlay-with-stats
    - /analyze/void-rate
- Prints a short summary for each.
"""

import os
import sys
import json
from typing import List

import requests


API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")


def _post_file(endpoint: str, image_path: str) -> dict:
    url = f"{API_BASE}{endpoint}"
    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(url, files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


def test_image(image_path: str) -> None:
    print(f"\n=== Testing image: {image_path} ===")

    # YOLO
    yolo = _post_file("/yolo/predict", image_path)
    detections = yolo.get("detections", [])
    print(f"[YOLO] detections: {len(detections)}")
    # Optional: show a few
    for det in detections[:5]:
        print(
            f"  - id={det['id']} cat={det['category']} "
            f"score={det['score']:.3f} bbox={det['bbox_xyxy']}"
        )

    # SAM
    sam = _post_file("/sam/segment-overlay-with-stats", image_path)
    segments = sam.get("segments", [])
    print(f"[SAM] segments: {len(segments)} "
          f"(overlay base64 length={len(sam.get('image_png_base64',''))})")

    # Void-rate
    void = _post_file("/analyze/void-rate", image_path)
    vrates = void.get("void_rates", [])
    print(f"[VOID] chips analyzed: {len(vrates)}")
    for v in vrates:
        print(
            f"  - chip_id={v['chip_id']} "
            f"void_rate={v['void_rate']:.4f} "
            f"void_area_px={v['void_area_px']} "
            f"component_area_px={v['component_area_px']}"
        )


def main(paths: List[str]) -> None:
    # Sanity: check healthz first
    health = requests.get(f"{API_BASE}/healthz", timeout=10).json()
    print(f"[healthz] {health}")

    for p in paths:
        if not os.path.isfile(p):
            print(f"Skipping non-file path: {p}")
            continue
        test_image(p)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_endpoints.py path/to/image1.jpg [image2.jpg ...]")
        sys.exit(1)
    main(sys.argv[1:])

