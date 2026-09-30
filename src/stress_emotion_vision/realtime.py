"""CLI: webcam / video emotion checker.

uv run python -m stress_emotion_vision.realtime --source 0 --show
uv run python -m stress_emotion_vision.realtime --source clip.mp4 --no-show --out annotated.mp4 --log session.csv
Press q to quit when --show. Set FACE_BACKEND=haar or EMOTION_BACKEND=onnx for lightweight mode.
"""

import argparse
import csv
import time

import cv2

from .draw import annotate
from .pipeline import process_frame


def parse_source(s: str):
    return int(s) if s.isdigit() else s


def main():
    ap = argparse.ArgumentParser(description="Realtime facial emotion checker")
    ap.add_argument("--source", default="0")
    ap.add_argument("--show", action="store_true", default=True)
    ap.add_argument("--no-show", dest="show", action="store_false")
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--every", type=int, default=2, help="run emotion every N frames (YuNet+ViT is heavier, default 2)")
    ap.add_argument("--smooth", action="store_true", default=True, help="temporal smoothing of emotion distribution")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false")
    a = ap.parse_args()

    cap = cv2.VideoCapture(parse_source(a.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {a.source} (try --source 1)")

    writer = None
    logf = open(a.log, "w", newline="") if a.log else None  # noqa: SIM115
    logw = csv.writer(logf) if logf else None
    if logw:
        logw.writerow(["t", "face", "emotion", "confidence", "second", "backend"])

    frame_i, last_results = 0, []
    prev_emotions: dict | None = None
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_i += 1
            if frame_i % a.every == 1:
                frame, last_results = process_frame(frame, prev_emotions if a.smooth else None, smooth=a.smooth)
                if last_results and a.smooth:
                    prev_emotions = last_results[0]["emotions"]
            else:
                for r in last_results:
                    annotate(frame, r["box"], r["emotion"], r["confidence"], r["second"], r["second_conf"])

            if writer is None and a.out:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            if writer:
                writer.write(frame)
            if logw:
                for i, r in enumerate(last_results):
                    logw.writerow([round(time.time() - t0, 2), i, r["emotion"],
                                   round(r["confidence"], 3), r["second"], r["backend"]])
            if a.show:
                cv2.imshow("emotion checker (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
        if logf:
            logf.close()
        cv2.destroyAllWindows()
    print(f"Done. frames={frame_i}" + (f" saved->{a.out}" if a.out else "") + (f" log->{a.log}" if a.log else ""))


if __name__ == "__main__":
    main()
