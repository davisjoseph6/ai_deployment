#!/usr/bin/env python3
"""
scripts/test_endpoints.py

Small dev client to sanity-check the backend endpoints:

- GET  /healthz
- POST /yolo/predict
- POST /sam/segment-overlay-with-stats
- POST /analyze/void-rate
- POST /feedback/roi-example   (optional; skipped if 404)

Usage:
    python scripts/test_endpoints.py path/to/image1.jpg [path/to/image2.png ...]

Environment variables:
    API_BASE_URL       (default: "http://127.0.0.1:8000")
    API_TIMEOUT        (default: "600.0" seconds read timeout)
    MAX_PRINT_DETS     (default: "5")  – max YOLO dets to print
    MAX_PRINT_SEGMENTS (default: "5")  – max SAM segments to print
    MAX_PRINT_CHIPS    (default: "5")  – max void-rate chip summaries to print
"""

import os
import sys
import pathlib
from typing import Any, Dict

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.getenv("API_TIMEOUT", "600.0"))  # seconds
# (connect_timeout, read_timeout)
TIMEOUT = (10.0, DEFAULT_TIMEOUT)

MAX_PRINT_DETS = int(os.getenv("MAX_PRINT_DETS", "5"))
MAX_PRINT_SEGMENTS = int(os.getenv("MAX_PRINT_SEGMENTS", "5"))
MAX_PRINT_CHIPS = int(os.getenv("MAX_PRINT_CHIPS", "5"))


def _post_file(endpoint: str, image_path: str) -> Dict[str, Any]:
    """POST an image file to an endpoint and return JSON."""
    url = BASE_URL + endpoint
    p = pathlib.Path(image_path)

    with p.open("rb") as f:
        # Use generic image/jpeg; backend just cares about bytes
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


def _test_yolo(image_path: str) -> None:
    """Test /yolo/predict on a single image."""
    try:
        yolo = _post_file("/yolo/predict", image_path)
        dets = yolo.get("detections", [])
        print(f"[YOLO] detections: {len(dets)}")
        for d in dets[:MAX_PRINT_DETS]:
            try:
                score = float(d.get("score", 0.0))
            except Exception:
                score = 0.0
            print(
                "  - id={id} cat={cat} score={score:.3f} bbox={bbox}".format(
                    id=d.get("id"),
                    cat=d.get("category"),
                    score=score,
                    bbox=d.get("bbox_xyxy"),
                )
            )
    except requests.exceptions.ReadTimeout:
        print("[YOLO] ERROR: Read timeout.")
        raise
    except Exception as e:
        print(f"[YOLO] ERROR: {e}")
        raise


def _test_sam(image_path: str) -> None:
    """Test /sam/segment-overlay-with-stats on a single image."""
    try:
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
    except requests.exceptions.ReadTimeout:
        print(
            "[SAM] ERROR: Read timeout "
            "(SAM may be too heavy for the current timeout)."
        )
        raise
    except Exception as e:
        print(f"[SAM] ERROR: {e}")
        raise


def _test_void_rate(image_path: str) -> None:
    """Test /analyze/void-rate on a single image."""
    try:
        vr = _post_file("/analyze/void-rate", image_path)
        vrates = vr.get("void_rates", [])
        print(f"[VOID-RATE] chips analyzed: {len(vrates)}")
        for r in vrates[:MAX_PRINT_CHIPS]:
            chip_id = r.get("chip_id")
            void_rate = r.get("void_rate")
            hole_ids = r.get("hole_ids")
            try:
                vr_f = float(void_rate) if void_rate is not None else 0.0
            except Exception:
                vr_f = 0.0
            print(
                "  - chip_id={cid} void_rate={vr:.4f} "
                "hole_ids={holes}".format(
                    cid=chip_id,
                    vr=vr_f,
                    holes=hole_ids,
                )
            )
    except requests.exceptions.ReadTimeout:
        print("[VOID-RATE] ERROR: Read timeout.")
        raise
    except Exception as e:
        print(f"[VOID-RATE] ERROR: {e}")
        raise


def _test_feedback(image_path: str) -> None:
    """
    Optional: test /feedback/roi-example with the FULL image as "ROI".

    This is just to verify that the endpoint accepts:
      - file: image
      - label: string
      - notes: string

    If the endpoint is not implemented and returns 404, we log & skip.
    """
    url = BASE_URL + "/feedback/roi-example"
    p = pathlib.Path(image_path)

    print("[FEEDBACK] posting full image as ROI feedback...")
    try:
        with p.open("rb") as f:
            files = {"file": (p.name, f, "image/jpeg")}
            data = {
                "label": "bubble",
                "notes": "test feedback from scripts/test_endpoints.py",
            }
            resp = requests.post(
                url,
                files=files,
                data=data,
                timeout=TIMEOUT,
            )
        if resp.status_code == 404:
            print(
                "[FEEDBACK] endpoint /feedback/roi-example not found (404) – "
                "skipping."
            )
            return
        resp.raise_for_status()
        payload = resp.json()
        saved_dir = payload.get("saved_dir") or payload.get("status")
        print(f"[FEEDBACK] OK, feedback stored. Info: {saved_dir}")
    except requests.exceptions.ReadTimeout:
        print("[FEEDBACK] ERROR: Read timeout.")
    except Exception as e:
        print(f"[FEEDBACK] ERROR: {e}")


def test_image(image_path: str) -> None:
    """Run the full test battery on a single image."""
    print(f"\n=== Testing image: {image_path} ===")

    # 1) YOLO
    try:
        _test_yolo(image_path)
    except Exception:
        # Already logged; stop this image test
        return

    # 2) SAM overlay + stats
    try:
        _test_sam(image_path)
    except Exception:
        return

    # 3) Void-rate endpoint
    try:
        _test_void_rate(image_path)
    except Exception:
        return

    # 4) Optional: feedback endpoint (won't crash if missing)
    _test_feedback(image_path)


def main(argv) -> None:
    if not argv:
        print(
            "Usage: python scripts/test_endpoints.py "
            "path/to/image1.jpg [path/to/image2.png ...]"
        )
        sys.exit(1)

    # Health check first
    try:
        health = _get_json("/healthz")
        print("[healthz]", health)
    except Exception as e:
        print(f"[healthz] ERROR: {e}")
        sys.exit(1)

    # Test each image path in turn
    for image_path in argv:
        if not os.path.exists(image_path):
            print(f"ERROR: {image_path} does not exist, skipping.")
            continue
        test_image(image_path)


if __name__ == "__main__":
    main(sys.argv[1:])

