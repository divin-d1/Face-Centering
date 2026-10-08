"""Shared camera-selection helpers.

The lab machine has two USB cameras (see `v4l2-ctl --list-devices`), so every
script that opens a webcam accepts a `--camera` argument instead of assuming
OpenCV index 0. A value can be a V4L2 device path (e.g. `/dev/video2`) or a
plain OpenCV index (e.g. `0`). The default is read from the `FACE_REC_CAMERA`
environment variable, falling back to `/dev/video2` (the external Wed Camera
capture node confirmed with `v4l2-ctl -d /dev/video2 --list-formats-ext`).
"""
from __future__ import annotations

import argparse
import os

import cv2

DEFAULT_CAMERA = os.environ.get("FACE_REC_CAMERA", "/dev/video2")


def add_camera_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=(
            "Camera device path (e.g. /dev/video2) or OpenCV index (e.g. 0). "
            f"Defaults to $FACE_REC_CAMERA or '{DEFAULT_CAMERA}'."
        ),
    )
    return parser


def open_camera(camera: str) -> cv2.VideoCapture:
    """Open a VideoCapture from a device path or a numeric index."""
    src = camera
    if isinstance(src, str) and src.lstrip("-").isdigit():
        src = int(src)
    # Let OpenCV choose the native backend (AVFoundation on macOS, V4L2 on
    # Linux). Forcing CAP_V4L2 prevents the project's camera from opening on Mac.
    return cv2.VideoCapture(src)
