import numpy as np

from stress_emotion_vision.detector import find_faces
from stress_emotion_vision.draw import annotate
from stress_emotion_vision.emotion import dominant
from stress_emotion_vision.pipeline import process_frame


def test_blank_image_no_crash():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out, results = process_frame(frame.copy())
    assert out.shape == frame.shape
    assert isinstance(results, list)


def test_annotate_draws_box():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotate(frame, (10, 50, 30, 30), "happy", 0.9, "neutral", 0.05)
    assert frame.sum() > 0


def test_detector_callable():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert isinstance(find_faces(frame), list)


def test_dominant_picks_max():
    emo, conf = dominant({"happy": 0.7, "sad": 0.2, "neutral": 0.1,
                          "angry": 0, "disgust": 0, "fear": 0, "surprise": 0})
    assert emo == "happy" and conf == 0.7
