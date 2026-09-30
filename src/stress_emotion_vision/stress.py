"""Stress estimation from facial emotion scores.

Vision-only stress is a *proxy*, not medical measurement.
Formula is deliberately simple + explainable for portfolio review.
"""

from __future__ import annotations

EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Weights tuned so fear/anger dominate; happy/neutral implicitly lower stress.
WEIGHTS = {
    "fear": 0.35,
    "angry": 0.30,
    "sad": 0.20,
    "disgust": 0.10,
    "surprise": 0.05,
    "happy": 0.0,
    "neutral": 0.0,
}


def estimate_stress(emotions: dict[str, float]) -> tuple[float, str]:
    """Return (score 0-100, level Low/Medium/High).

    Accepts scores as 0-1 probabilities or 0-100. Missing keys default to 0.
    """
    total = 0.0
    for emo, w in WEIGHTS.items():
        v = float(emotions.get(emo, 0.0))
        if v > 1.0:  # allow 0-100 inputs
            v /= 100.0
        v = max(0.0, min(1.0, v))
        total += v * w
    score = round(total * 100.0, 1)
    if score < 30:
        level = "Low"
    elif score < 65:
        level = "Medium"
    else:
        level = "High"
    return score, level


class StressSmoother:
    """Exponential moving average to avoid flicker in realtime loop."""

    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha
        self.value: float | None = None

    def update(self, score: float) -> float:
        if self.value is None:
            self.value = score
        else:
            self.value = self.alpha * score + (1 - self.alpha) * self.value
        return round(self.value, 1)
