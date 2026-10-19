"""Downloads the two pretrained ONNX models used for facial emotion detection.

Run once after cloning the project::

    python download_models.py

Both files land in ``models/`` and are then used entirely offline - the app
never contacts the network again once they are in place.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODELS = [
    {
        "name": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "min_bytes": 200_000,
        "description": "YuNet face detector (OpenCV Zoo)",
    },
    {
        "name": "emotion-ferplus-8.onnx",
        "url": "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
        "min_bytes": 30_000_000,
        "description": "FER+ facial emotion classifier (ONNX Model Zoo)",
    },
]


def report(count: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(count * block_size, total)
    percent = done * 100 // total
    bar = "#" * (percent // 3)
    sys.stdout.write(f"\r    [{bar:<33}] {percent:3d}%  ({done / 1e6:.1f}/{total / 1e6:.1f} MB)")
    sys.stdout.flush()


def download(model: dict) -> bool:
    destination = os.path.join(MODELS_DIR, model["name"])

    if os.path.exists(destination) and os.path.getsize(destination) >= model["min_bytes"]:
        print(f"  ✓ {model['name']} already present")
        return True

    print(f"  ↓ {model['description']}")
    temporary = destination + ".part"
    try:
        urllib.request.urlretrieve(model["url"], temporary, reporthook=report)
        print()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
        print(f"\n  ✗ Download failed: {exc}")
        if os.path.exists(temporary):
            os.remove(temporary)
        return False

    if os.path.getsize(temporary) < model["min_bytes"]:
        print(f"  ✗ File looks truncated ({os.path.getsize(temporary)} bytes)")
        os.remove(temporary)
        return False

    os.replace(temporary, destination)
    print(f"  ✓ Saved to models/{model['name']}")
    return True


def main() -> int:
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("Fetching pretrained models for the AI Emotion Detection System\n")

    if all(download(model) for model in MODELS):
        print("\nAll models are ready. Start the app with:  python app.py")
        return 0

    print("\nSome downloads failed. Check your connection and run this again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
