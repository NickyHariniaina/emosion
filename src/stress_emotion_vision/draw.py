"""Drawing: emotion-first. Box color by emotion family."""

from __future__ import annotations

import cv2

# green = positive, red/orange = negative, gray = neutral
EMO_COLORS = {
    "happy": (0, 200, 0),
    "surprise": (0, 200, 200),
    "neutral": (170, 170, 170),
    "sad": (255, 150, 0),
    "fear": (0, 100, 255),
    "angry": (0, 0, 255),
    "disgust": (80, 0, 160),
}


def annotate(frame, box, emotion: str, conf: float, second: str = "", second_conf: float = 0.0):
    x, y, w, h = box
    color = EMO_COLORS.get(emotion, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    label1 = f"{emotion} {conf*100:.0f}%"
    label2 = f"{second} {second_conf*100:.0f}%" if second else ""
    bar_h = 44 if label2 else 26
    y0 = max(0, y - bar_h)
    cv2.rectangle(frame, (x, y0), (x + max(190, w), y), color, -1)
    cv2.putText(frame, label1, (x + 4, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if label2:
        cv2.putText(frame, label2, (x + 4, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    return frame
