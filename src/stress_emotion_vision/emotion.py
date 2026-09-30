"""Emotion classification — accuracy-first.

Priority (env EMOTION_BACKEND, default: hf):
1. hf   : dima806/facial_emotions_image_detection (ViT, 90.9% acc, 224px).
          Needs torch+transformers. Best accuracy. Auto-downloads once (~350MB).
2. onnx : FER+ tiny CNN via OpenCV DNN (light, no torch). Fallback.
3. fallback: neutral (offline) — clearly flagged.

All backends return 7 labels: angry, disgust, fear, happy, sad, surprise, neutral.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

HF_MODEL_ID = "dima806/facial_emotions_image_detection"
LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

ONNX_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
ONNX_PATH = Path.home() / ".cache" / "stress-vision" / "emotion-ferplus-8.onnx"
FERPLUS_MAP = {0: "neutral", 1: "happy", 2: "surprise", 3: "sad",
               4: "angry", 5: "disgust", 6: "fear", 7: "disgust"}

_onnx_net = None
_hf_pipe = None


def backend_name() -> str:
    return os.environ.get("EMOTION_BACKEND", "hf").lower()


# ---------- ONNX (light fallback) ----------

def _ensure_onnx() -> Path:
    ONNX_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not ONNX_PATH.exists():
        print(f"Downloading emotion model to {ONNX_PATH} ...")
        urllib.request.urlretrieve(ONNX_URL, ONNX_PATH)
    return ONNX_PATH


def predict_onnx(face_bgr) -> dict[str, float]:
    global _onnx_net
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64)).astype(np.float32)
    if _onnx_net is None:
        _onnx_net = cv2.dnn.readNetFromONNX(str(_ensure_onnx()))
    _onnx_net.setInput(gray.reshape(1, 1, 64, 64))
    scores = np.array(_onnx_net.forward()).flatten()
    e = np.exp(scores - scores.max())
    probs = e / e.sum()
    out = {k: 0.0 for k in LABELS}
    for i, p in enumerate(probs):
        out[FERPLUS_MAP[i]] += float(p)
    s = sum(out.values())
    return {k: round(v / s, 4) for k, v in out.items()}


# ---------- HF ViT (accurate primary) ----------

def _get_hf():
    global _hf_pipe
    if _hf_pipe is None:
        from transformers import pipeline

        # CPU, no top-k truncation — we want full distribution
        _hf_pipe = pipeline(
            "image-classification",
            model=HF_MODEL_ID,
            top_k=None,
            device=-1,
        )
    return _hf_pipe


def predict_hf(face_bgr) -> dict[str, float]:
    """ViT 224px. Upscale small crops, BGR->RGB, keep aspect via resize."""
    from PIL import Image

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    # upscale tiny webcam crops — ViT was trained on 224px
    h, w = rgb.shape[:2]
    if max(h, w) < 100:
        rgb = cv2.resize(rgb, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    img = Image.fromarray(rgb)
    preds = _get_hf()(img)
    out = {k: 0.0 for k in LABELS}
    for p in preds:
        label = str(p["label"]).lower().strip()
        # HF labels: "happy", "sad", ... sometimes "fear", "angry", "disgust", "surprise", "neutral"
        if label in out:
            out[label] = float(p["score"])
    s = sum(out.values()) or 1.0
    return {k: round(v / s, 4) for k, v in out.items()}


def predict_emotions(face_bgr) -> tuple[dict[str, float], str]:
    """Return (emotions dict, backend_used). Never raises."""
    b = backend_name()
    order = {"hf": ("hf", "onnx"), "onnx": ("onnx", "hf"),
             "deepface": ("deepface", "hf")}.get(b, ("hf", "onnx"))
    for name in order:
        try:
            if name == "hf":
                return predict_hf(face_bgr), "hf-vit-dima806"
            if name == "onnx":
                return predict_onnx(face_bgr), "onnx-ferplus"
            if name == "deepface":
                from deepface import DeepFace  # optional

                res = DeepFace.analyze(face_bgr, actions=["emotion"], enforce_detection=False)
                if isinstance(res, list):
                    res = res[0]
                emo = res["emotion"]
                total = sum(emo.values()) or 1.0
                return {k: round(float(emo.get(k, 0)) / total, 4) for k in LABELS}, "deepface"
        except Exception as e:  # noqa: BLE001
            print(f"[warn] emotion backend {name} failed ({e}), trying next")
    print("[warn] all emotion backends failed, using neutral fallback")
    return {k: (1.0 if k == "neutral" else 0.0) for k in LABELS}, "fallback"


def dominant(emotions: dict[str, float]) -> tuple[str, float]:
    k = max(emotions, key=lambda x: emotions[x])
    return k, float(emotions[k])


def top2(emotions: dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    s = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return (s[0][0], s[0][1]), (s[1][0], s[1][1])
