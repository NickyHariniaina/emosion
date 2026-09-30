"""Face localization: YuNet DNN primary (accurate), Haar fallback (offline).

YuNet: 0.88 AP on WIDER Face, landmarks for alignment, ~1ms CPU.
Model auto-downloads once (~230KB) to ~/.cache/stress-vision/.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_PATH = Path.home() / ".cache" / "stress-vision" / "face_detection_yunet_2023mar.onnx"

_cascade = None
_yunet = None
_yunet_input_size = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _cascade
    if _cascade is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _cascade = cv2.CascadeClassifier(path)
    return _cascade


def _ensure_yunet() -> Path:
    YUNET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not YUNET_PATH.exists():
        print(f"Downloading YuNet face detector (~230KB) to {YUNET_PATH} ...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return YUNET_PATH


def _get_yunet(frame_shape):
    """YuNet needs input size at creation; recreate if frame size changed."""
    global _yunet, _yunet_input_size
    h, w = frame_shape[:2]
    if _yunet is None or _yunet_input_size != (w, h):
        _yunet = cv2.FaceDetectorYN_create(
            str(_ensure_yunet()), "", (w, h), 0.6, 0.3, 5000
        )
        _yunet_input_size = (w, h)
    return _yunet


def find_faces_yunet(frame_bgr) -> list[tuple[int, int, int, int]] | None:
    """Return boxes or None if YuNet unavailable (triggers fallback)."""
    detector = _get_yunet(frame_bgr.shape)
    detector.setInputSize((frame_bgr.shape[1], frame_bgr.shape[0]))
    _, faces = detector.detect(frame_bgr)
    if faces is None:
        return []
    boxes = []
    for f in faces:
        x, y, w, h = (int(v) for v in f[:4])
        # pad 10% for emotion context, clamp to frame
        px, py = int(w * 0.1), int(h * 0.1)
        x, y = max(0, x - px), max(0, y - py)
        w = min(frame_bgr.shape[1] - x, w + 2 * px)
        h = min(frame_bgr.shape[0] - y, h + 2 * py)
        boxes.append((x, y, w, h))
    return boxes


def find_faces_haar(frame_bgr, min_size: int = 60) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)
    )
    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]


def find_faces(frame_bgr, min_size: int = 60) -> list[tuple[int, int, int, int]]:
    """YuNet first, Haar fallback. Never raises."""
    import os

    if os.environ.get("FACE_BACKEND", "yunet").lower() == "haar":
        return find_faces_haar(frame_bgr, min_size)
    try:
        r = find_faces_yunet(frame_bgr)
        if r is not None:
            return r
    except Exception as e:  # noqa: BLE001
        print(f"[warn] YuNet failed ({e}), using Haar fallback")
    return find_faces_haar(frame_bgr, min_size)
