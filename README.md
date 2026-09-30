# Emotion Checker (accurate)

Locate faces in an image, webcam stream, or video file and classify **facial emotion** with confidence scores: angry, disgust, fear, happy, sad, surprise, neutral. Runs 100% locally — no API keys, no uploads.

Project root (full path):

```
/home/bocchi/workspace/personal/stress-emotion-vision
```

## How it works

```
/home/bocchi/workspace/personal/stress-emotion-vision/src/stress_emotion_vision/pipeline.py
  frame
  → detector.py   : YuNet DNN face detect (+10% padding) — Haar fallback
  → emotion.py    : ViT dima806/facial_emotions_image_detection (224px RGB) — ONNX FER+ fallback
  → draw.py       : box colored by emotion + "emotion NN%" label (+ 2nd guess if >20%)
  → realtime only : temporal smoothing over emotion distributions
```

* **Detector — YuNet** (`/home/bocchi/.cache/stress-vision/face_detection_yunet_2023mar.onnx`, ~230KB, auto-downloaded). WIDER Face AP ~0.88. Set `FACE_BACKEND=haar` to force the Haar-cascade fallback.
* **Emotion — ViT** (`dima806/facial_emotions_image_detection`, ~90.9% reported accuracy, cached by HuggingFace Hub under `/home/bocchi/.cache/huggingface/`). Set `EMOTION_BACKEND=onnx` to force the light FER+ CNN (`/home/bocchi/.cache/stress-vision/emotion-ferplus-8.onnx`, ~34MB, no torch needed).
* Measured on `/home/bocchi/workspace/personal/stress-emotion-vision/assets/demo/lena.jpg`: YuNet + ViT → `neutral 41% / happy 40% / sad 10%` (honest on an ambiguous smile). Webcam: stable detections across consecutive frames.
* NOTE: `src/stress_emotion_vision/stress.py` is legacy only. Nothing in the pipeline, CLI, or UI uses it. This is an emotion checker, not a stress checker.

## Project layout (full paths)

```
/home/bocchi/workspace/personal/stress-emotion-vision/
  pyproject.toml
  README.md
  .python-version                        (3.12)
  .venv/                                 (created by `uv sync`)
  assets/demo/lena.jpg                   (test input)
  assets/demo/lena_out.jpg               (annotated sample output)
  src/stress_emotion_vision/
    __init__.py
    pipeline.py                          (process_frame / process_image)
    detector.py                          (YuNet + Haar fallback)
    emotion.py                           (ViT + ONNX fallback)
    draw.py                              (emotion-colored boxes)
    image_cli.py                         (single image CLI)
    realtime.py                          (webcam / video CLI)
    stress.py                            (LEGACY, unused)
  tests/
    test_pipeline.py
    test_stress.py
```

Model / cache files (outside the repo):

```
/home/bocchi/.cache/stress-vision/face_detection_yunet_2023mar.onnx
/home/bocchi/.cache/stress-vision/emotion-ferplus-8.onnx
/home/bocchi/.cache/huggingface/                     (ViT weights + processor)
/dev/video0                                          (Integrated Camera — use this)
/dev/video1                                          (camera metadata node — do NOT use as source)
```

## 1. Setup (full path)

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv sync
```

This creates `/home/bocchi/workspace/personal/stress-emotion-vision/.venv/` with Python 3.12, `opencv-python==4.12.0.88` (4.x — 5.x removed `CascadeClassifier`), torch CPU, transformers, numpy, pillow. First model run downloads weights once (needs internet); afterwards fully offline.

Verify install:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run pytest -q
uv run ruff check src tests
```

## 2. Image mode — full use

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.image_cli \
  --in /home/bocchi/workspace/personal/stress-emotion-vision/assets/demo/lena.jpg \
  --out /tmp/emo_lena.jpg
```

Expected output:

```
Faces: 1 -> /tmp/emo_lena.jpg
  neutral 41.2% | [neutral:41% happy:40% sad:10%] | box (193, 162, 173, 246) | hf-vit-dima806
```

* `--in` — any readable image path (jpg/png). Use an absolute path.
* `--out` — annotated copy is written here (boxes + labels burned in). Parent dir must exist.
* Console prints per face: top emotion + confidence, top-3 distribution, box `(x, y, w, h)`, backend name.
* `Faces: 0` means no face found — move closer, face the camera, add front light.
* Try your own photo:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.image_cli \
  --in /home/bocchi/Pictures/myphoto.jpg \
  --out /tmp/myphoto_out.jpg
```

## 3. Webcam mode — full use (your camera)

Your camera: Integrated Camera at `/dev/video0`. (`/dev/video1` is its metadata node — ignore it.)

Basic (window shows live labels, press `q` to quit):

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.realtime --source 0 --show
```

If `/dev/video0` is busy or black, try `--source 1`:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.realtime --source 1 --show
```

Headless (no window — for Wayland/SSH) + save annotated video + CSV log:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.realtime \
  --source 0 --no-show \
  --out /tmp/annotated.mp4 \
  --log /tmp/session.csv
```

Then inspect:

```bash
head -n 10 /tmp/session.csv
```

CSV columns: `t,face,emotion,confidence,second,backend`.

All realtime flags (full list):

```
--source 0        camera index (/dev/video0) or full file path (see §4)
--show / --no-show
--out PATH        save annotated video (e.g. /tmp/annotated.mp4)
--log PATH        save per-face CSV log (e.g. /tmp/session.csv)
--every N         run YuNet+ViT every Nth frame (default 2; raise to 3-4 if slow)
--smooth / --no-smooth   temporal smoothing on/off (default on)
```

Performance tip on this machine (7GB RAM, CPU-only): `--every 2 --smooth` is the sweet spot. If laggy: `--every 4`.

## 4. Video-file mode — full use

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.realtime \
  --source /home/bocchi/Videos/clip.mp4 \
  --show
```

Headless transcode of a full file:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
uv run python -m stress_emotion_vision.realtime \
  --source /home/bocchi/Videos/clip.mp4 \
  --no-show \
  --out /tmp/clip_annotated.mp4 \
  --log /tmp/clip.csv
```

## 5. Backend overrides (full use)

Default = most accurate (`FACE_BACKEND=yunet`, `EMOTION_BACKEND=hf`). Override per run:

```bash
cd /home/bocchi/workspace/personal/stress-emotion-vision
# lightweight, no torch: Haar faces + FER+ ONNX emotions
FACE_BACKEND=haar EMOTION_BACKEND=onnx \
  uv run python -m stress_emotion_vision.realtime --source 0 --show

# YuNet faces + light ONNX emotions (good middle ground)
EMOTION_BACKEND=onnx \
  uv run python -m stress_emotion_vision.image_cli \
    --in /home/bocchi/workspace/personal/stress-emotion-vision/assets/demo/lena.jpg \
    --out /tmp/emo_lena_onnx.jpg
```

## 6. Reading the output

* Box color: green `happy`, orange `surprise`, gray `neutral`, blue-orange `sad`, red `angry`, purple `disgust`, deep-blue `fear`.
* Label line 1: `emotion NN%` = top prediction + confidence.
* Label line 2 (only if second guess >20%): e.g. `happy 40%` under `neutral 41%` = genuinely ambiguous face, not a bug.
* Webcam: trust the label that is stable for ~1 second with `--smooth` on, not single-frame flicker.

## 7. Tips for an accurate scan

1. Face the camera straight on, both eyes visible, ~50cm distance.
2. Light your face from the front (window/monitor); avoid strong backlight.
3. Start with a neutral background; then test happy / sad / surprised and compare the printed top-3.
4. Remove sunglasses/mask for the test — they drop every FER model.
5. If confidence is split (e.g. `neutral 41 / happy 40`), that IS the accurate answer for an ambiguous expression.

## 8. Troubleshooting (full paths)

| Symptom | Fix |
|---|---|
| `Cannot open source 0` | camera busy — close browser/meet; try `--source 1`; check `ls -l /dev/video*` |
| Black window | wrong node — use `--source 0`, not `1` |
| `imshow` / Qt error on Wayland | use `--no-show --out /tmp/annotated.mp4` then play the file |
| Slow / laggy | `--every 4 --no-smooth`, close other apps (torch CPU is RAM-hungry) |
| `Faces: 0` on photo | closer crop, frontal, better light; check file reads: `python3 -c "import cv2; print(cv2.imread('/your/photo.jpg') is not None)"` |
| First run downloads slowly | HF Hub rate limit — set `HF_TOKEN` env var, rerun; models land in `/home/bocchi/.cache/huggingface/` and `/home/bocchi/.cache/stress-vision/` |
| `CascadeClassifier` AttributeError | you installed opencv 5.x — repo pins `opencv-python==4.12.0.88` in `/home/bocchi/workspace/personal/stress-emotion-vision/pyproject.toml`; run `uv sync` again |
| Want a clean slate | `rm -rf /home/bocchi/.cache/stress-vision /home/bocchi/.cache/huggingface` then rerun (re-downloads) |

## 9. Limits

* FER datasets are biased by age, ethnicity, lighting, glasses. Accuracy drops off-distribution.
* Not for hiring, surveillance, or health decisions.
* Everything runs locally at `/home/bocchi/workspace/personal/stress-emotion-vision/.venv/` — frames never leave the machine.
