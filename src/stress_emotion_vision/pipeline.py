"""Emotion-checker pipeline: locate face -> classify emotion -> annotate.

Stress estimation kept as optional (--stress flag) but OFF by default.
This project is an emotion checker, not a stress checker.
"""

from __future__ import annotations

import cv2

from .detector import find_faces
from .draw import annotate
from .emotion import predict_emotions, top2


def _smooth(prev: dict | None, emotions: dict, alpha: float = 0.5) -> dict:
    """Temporal smoothing over emotion distributions (realtime stability)."""
    if prev is None:
        return dict(emotions)
    return {k: round(alpha * emotions.get(k, 0) + (1 - alpha) * prev.get(k, 0), 4)
            for k in emotions}


def process_frame(frame, prev_emotions: dict | None = None, smooth: bool = False):
    faces = find_faces(frame)
    results = []
    for box in faces:
        x, y, w, h = box
        crop = frame[max(0, y): y + h, max(0, x): x + w]
        if crop.size == 0:
            continue
        emotions, backend = predict_emotions(crop)
        if smooth:
            emotions = _smooth(prev_emotions, emotions)
            prev_emotions = emotions
        (emo, conf), (emo2, conf2) = top2(emotions)
        # only show 2nd if meaningful (>20%) and different
        second = emo2 if conf2 > 0.20 else ""
        annotate(frame, box, emo, conf, second, conf2 if second else 0.0)
        results.append({"box": box, "emotion": emo, "confidence": conf,
                        "second": second, "second_conf": conf2,
                        "emotions": emotions, "backend": backend})
    return frame, results


def process_image(in_path: str, out_path: str):
    img = cv2.imread(in_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {in_path}")
    frame, results = process_frame(img)
    cv2.imwrite(out_path, frame)
    return results
