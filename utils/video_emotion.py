"""Facial emotion recognition that runs entirely on the local machine.

Two ONNX models are chained together:

1. **YuNet** (``face_detection_yunet_2023mar.onnx``) locates faces in the frame.
   It ships with the OpenCV Zoo, is ~230 KB and runs comfortably in real time.
2. **FER+** (``emotion-ferplus-8.onnx``) classifies the cropped face into eight
   emotion classes.  FER+ is the relabelled version of the FER2013 dataset where
   every image was scored by ten annotators, so its probabilities are noticeably
   better calibrated than the original FER2013 models.

Nothing here talks to the network: both models are read from ``models/`` and
inference happens through onnxruntime on the CPU (or Apple's CoreML provider
when it is available).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import cv2
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

FACE_MODEL_PATH = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
EMOTION_MODEL_PATH = os.path.join(MODELS_DIR, "emotion-ferplus-8.onnx")

# Output order of the FER+ model. Do not reorder - it is fixed by the network.
FERPLUS_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]

# The seven standard facial emotions this app reports. FER+ also emits
# "contempt", which is dropped: it is not one of the seven, it has no ground
# truth in FER2013, and once the prior correction below is applied it starts
# winning frames it has no business winning.
ACTIVE_LABELS = [label for label in FERPLUS_LABELS if label != "contempt"]
ACTIVE_INDICES = [FERPLUS_LABELS.index(label) for label in ACTIVE_LABELS]

# FER+ is badly skewed towards neutral and happiness. These are the model's own
# mean predicted probabilities over the 7178-image FER2013 test set - in other
# words, its built-in bias, measured rather than guessed. Order matches
# ACTIVE_LABELS.
MODEL_PRIOR = np.array(
    [0.3976, 0.2658, 0.1250, 0.1075, 0.0708, 0.0121, 0.0212], dtype=np.float64
)

# Strength of the prior correction (standard logit adjustment:
# p_adjusted ∝ p / prior**tau). Chosen by sweeping tau on one half of the
# FER2013 test set and confirming on the untouched other half. It raises mean
# per-class recall from 49.7% to 58.8% *and* plain accuracy from 56.8% to
# 60.5%, so it is not a precision-for-recall trade - see README.
PRIOR_STRENGTH = 1.1

# Padding applied to YuNet's box so the crop matches the framing FER+ was
# trained on. Measured by running YuNet over canonical FER2013 crops: its box
# covers 81% of the width but 97% of the height, so the horizontal padding has
# to be far larger than the vertical. A symmetric pad crops too loosely
# vertically and costs accuracy.
FACE_PAD_X = 0.115
FACE_PAD_Y = 0.018

# Friendly names used by the dashboard.
LABEL_DISPLAY = {
    "neutral": "Neutral",
    "happiness": "Happy",
    "surprise": "Surprise",
    "sadness": "Sad",
    "anger": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
}

EMOJI = {
    "Neutral": "😐",
    "Happy": "😄",
    "Surprise": "😲",
    "Sad": "😢",
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
}


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


class VideoEmotionDetector:
    """Detects the dominant facial emotion in a single BGR frame.

    The detector keeps a short history of recent predictions and averages them.
    Frame-by-frame predictions flicker a lot - a single blink can swing the
    top class - so the smoothed result is what the UI actually shows.

    The window is deliberately short (3 frames, about one second at the rate
    the dashboard polls). An earlier six-frame window averaged over over three
    seconds, which was long enough to erase brief expressions: a smile or a
    flash of surprise was averaged back into whatever the face rested on.
    """

    def __init__(self, smoothing_window: int = 3, score_threshold: float = 0.6):
        if not os.path.exists(FACE_MODEL_PATH):
            raise FileNotFoundError(
                f"Face detection model missing: {FACE_MODEL_PATH}\n"
                "Run 'python download_models.py' to fetch it."
            )
        if not os.path.exists(EMOTION_MODEL_PATH):
            raise FileNotFoundError(
                f"Emotion model missing: {EMOTION_MODEL_PATH}\n"
                "Run 'python download_models.py' to fetch it."
            )

        # YuNet is stateful (it remembers the input size), and the Flask dev
        # server is threaded, so every call has to hold this lock.
        self._lock = threading.Lock()

        self._face_detector = cv2.FaceDetectorYN.create(
            model=FACE_MODEL_PATH,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

        providers = ort.get_available_providers()
        # CoreML gives a decent speedup on Apple Silicon but silently falls back
        # to CPU on machines that do not have it.
        preferred = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in providers]
        options = ort.SessionOptions()
        options.log_severity_level = 3  # keep the console clean
        try:
            self._session = ort.InferenceSession(
                EMOTION_MODEL_PATH, sess_options=options, providers=preferred
            )
        except Exception:
            self._session = ort.InferenceSession(
                EMOTION_MODEL_PATH, sess_options=options, providers=["CPUExecutionProvider"]
            )

        self._input_name = self._session.get_inputs()[0].name
        self._window = max(1, smoothing_window)
        # One history per browser tab, so two tabs never blend their faces.
        self._histories: Dict[str, Deque[np.ndarray]] = {}

    # ------------------------------------------------------------------ faces

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, int]]:
        """Returns bounding boxes ``{x, y, w, h}`` for every face in the frame."""
        height, width = frame.shape[:2]
        with self._lock:
            self._face_detector.setInputSize((width, height))
            _, faces = self._face_detector.detect(frame)

        if faces is None:
            return []

        boxes = []
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            # YuNet can return boxes that hang off the edge of the frame.
            x, y = max(0, x), max(0, y)
            w, h = min(w, width - x), min(h, height - y)
            if w > 0 and h > 0:
                boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return boxes

    # --------------------------------------------------------------- emotions

    def _classify_face(self, gray_face: np.ndarray) -> np.ndarray:
        """Runs FER+ on a cropped grayscale face.

        Returns seven probabilities in :data:`ACTIVE_LABELS` order, already
        corrected for the model's neutral/happiness bias.
        """
        resized = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
        # FER+ expects raw 0-255 intensities, NOT values scaled to 0-1.
        # Dividing by 255 collapses every prediction to "neutral" - verified.
        tensor = resized.astype(np.float32).reshape(1, 1, 64, 64)
        logits = self._session.run(None, {self._input_name: tensor})[0]

        probs = _softmax(logits[0])[ACTIVE_INDICES]
        probs = probs / probs.sum()
        return self._apply_prior_correction(probs)

    @staticmethod
    def _apply_prior_correction(probs: np.ndarray) -> np.ndarray:
        """Divides out FER+'s built-in class bias (logit adjustment).

        Without this the model answers "neutral" or "happy" for about 70% of
        all frames and essentially never says "fear" or "disgust", regardless
        of what the face is doing.
        """
        if PRIOR_STRENGTH <= 0:
            return probs
        adjusted = probs / np.power(MODEL_PRIOR, PRIOR_STRENGTH)
        return adjusted / adjusted.sum()

    def predict(self, frame: np.ndarray, smooth: bool = True, sid: str = "default") -> Dict[str, Any]:
        """Analyses one BGR frame and returns the dominant emotion.

        The largest face is used when several people are in shot, since that is
        almost always the person sitting in front of the webcam.
        """
        boxes = self.detect_faces(frame)
        if not boxes:
            # Let the history decay so a stale emotion does not linger forever.
            self.reset(sid)
            return {
                "face_detected": False,
                "emotion": None,
                "confidence": 0.0,
                "scores": {},
                "faces": [],
                "message": "No face detected",
            }

        primary = max(boxes, key=lambda b: b["w"] * b["h"])

        # Reshape YuNet's box into the framing FER+ expects (see FACE_PAD_*).
        pad_x = int(primary["w"] * FACE_PAD_X)
        pad_y = int(primary["h"] * FACE_PAD_Y)
        x1 = max(0, primary["x"] - pad_x)
        y1 = max(0, primary["y"] - pad_y)
        x2 = min(frame.shape[1], primary["x"] + primary["w"] + pad_x)
        y2 = min(frame.shape[0], primary["y"] + primary["h"] + pad_y)

        face_bgr = frame[y1:y2, x1:x2]
        if face_bgr.size == 0:
            return {
                "face_detected": False,
                "emotion": None,
                "confidence": 0.0,
                "scores": {},
                "faces": [],
                "message": "Face crop was empty",
            }

        # No histogram equalisation here on purpose. It sounds like it should
        # help with bad lighting, but FER+ was trained on un-equalised FER2013
        # crops: measured over the FER2013 test set it costs 4 points of mean
        # per-class recall and cuts "surprise" recall from 75% to 50%.
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)

        probs = self._classify_face(gray)

        if smooth:
            with self._lock:
                if sid not in self._histories and len(self._histories) >= 16:
                    self._histories.pop(next(iter(self._histories)))  # oldest tab
                history = self._histories.setdefault(sid, deque(maxlen=self._window))
                history.append(probs)
                probs = np.mean(np.stack(history, axis=0), axis=0)

        top_index = int(np.argmax(probs))
        display = LABEL_DISPLAY[ACTIVE_LABELS[top_index]]

        scores = {
            LABEL_DISPLAY[label]: round(float(probs[i]) * 100.0, 2)
            for i, label in enumerate(ACTIVE_LABELS)
        }

        return {
            "face_detected": True,
            "emotion": display,
            "emoji": EMOJI.get(display, "🙂"),
            "confidence": round(float(probs[top_index]) * 100.0, 2),
            "scores": scores,
            "faces": boxes,
            "primary_face": primary,
            "message": "ok",
        }

    def reset(self, sid: str = "default") -> None:
        """Clears one tab's smoothing history."""
        with self._lock:
            self._histories.pop(sid, None)


_detector: Optional[VideoEmotionDetector] = None
_detector_lock = threading.Lock()


def get_detector() -> VideoEmotionDetector:
    """Lazily builds one shared detector - loading the ONNX models is slow."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = VideoEmotionDetector()
    return _detector
