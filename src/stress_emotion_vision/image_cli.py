"""CLI: emotion checker for an image.

uv run python -m stress_emotion_vision.image_cli --in photo.jpg --out result.jpg
"""

import argparse

from .pipeline import process_image


def main():
    ap = argparse.ArgumentParser(description="Detect faces + emotion (accurate ViT model)")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    a = ap.parse_args()
    results = process_image(a.inp, a.out)
    print(f"Faces: {len(results)} -> {a.out}")
    for r in results:
        dist = " ".join(f"{k}:{v*100:.0f}%" for k, v in sorted(r["emotions"].items(), key=lambda x: -x[1])[:3])
        print(f"  {r['emotion']} {r['confidence']*100:.1f}% | [{dist}] | box {r['box']} | {r['backend']}")
    if not results:
        print("  (no faces found — try better lighting / frontal face, closer to camera)")


if __name__ == "__main__":
    main()
