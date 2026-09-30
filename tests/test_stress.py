from stress_emotion_vision.stress import StressSmoother, estimate_stress


def test_happy_is_low_stress():
    score, level = estimate_stress({"happy": 0.9, "neutral": 0.1})
    assert score < 30 and level == "Low"


def test_fear_is_high_stress():
    score, level = estimate_stress({"fear": 1.0, "angry": 1.0, "sad": 1.0})
    assert level == "High" and score > 65


def test_smoother_converges():
    s = StressSmoother(alpha=0.5)
    assert s.update(100) == 100
    v = s.update(0)
    assert 0 < v < 100
