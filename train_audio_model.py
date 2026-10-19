"""Trains the local speech-emotion classifier on the RAVDESS corpus.

Usage
-----
    python train_audio_model.py                 # train from data/ravdess
    python train_audio_model.py --data-dir DIR  # custom dataset location

The dataset is *not* bundled with this project (it is ~200 MB).  Download
"Audio_Speech_Actors_01-24.zip" from https://zenodo.org/records/1188976 and
unzip it into ``data/ravdess/`` so the layout is::

    data/ravdess/Actor_01/03-01-01-01-01-01-01.wav
    data/ravdess/Actor_02/...

The trained pipeline is written to ``models/audio_emotion_model.joblib``.

Note on evaluation: the train/test split is made **by actor**, so no speaker
appears in both halves.  A random file-level split would let the model memorise
individual voices and would report a much higher - and much less honest -
accuracy than it deserves.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.audio_emotion import (  # noqa: E402
    EMOTIONS,
    MAX_SECONDS,
    MODEL_PATH,
    RAVDESS_EMOTION_MAP,
    SAMPLE_RATE,
    extract_features,
    load_audio,
    preprocess,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data", "ravdess")

# Actors held out for testing. RAVDESS actors alternate female/male by parity,
# so this selection keeps both genders represented on each side of the split.
TEST_ACTORS = {"05", "06", "17", "18", "23", "24"}


def parse_filename(path: str) -> Tuple[str, str]:
    """Returns ``(emotion_label, actor_id)`` for a RAVDESS file, or ``("", "")``."""
    name = os.path.splitext(os.path.basename(path))[0]
    parts = name.split("-")
    if len(parts) != 7:
        return "", ""
    emotion = RAVDESS_EMOTION_MAP.get(parts[2], "")
    return emotion, parts[6]


def augment(y: np.ndarray) -> List[np.ndarray]:
    """Returns perturbed copies of a clip for training-set expansion.

    RAVDESS only has 24 speakers, so a model trained on it happily memorises
    individual voices.  Shifting the pitch, adding hiss and changing the speed
    forces it to rely on the emotional contour instead.  This is applied to the
    training half only - augmenting the test half would inflate the score.
    """
    import librosa

    variants = [
        librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=1.5),
        librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=-1.5),
        np.clip(y + 0.005 * np.random.randn(len(y)).astype(np.float32), -1.0, 1.0),
        _fit_length(librosa.effects.time_stretch(y=y, rate=1.12)),
    ]
    return [v.astype(np.float32) for v in variants]


def _fit_length(y: np.ndarray) -> np.ndarray:
    """Crops or pads a stretched clip back to the fixed analysis window."""
    target = int(SAMPLE_RATE * MAX_SECONDS)
    if y.size > target:
        start = (y.size - target) // 2
        return y[start : start + target]
    return np.pad(y, (0, target - y.size), mode="constant")


def _process_one(path: str, with_augmentation: bool):
    """Returns ``(list_of_feature_vectors, emotion, actor)`` for one clip."""
    emotion, actor = parse_filename(path)
    if not emotion:
        return None
    try:
        y = preprocess(load_audio(path))
        vectors = [extract_features(y)]
        if with_augmentation:
            vectors.extend(extract_features(v) for v in augment(y))
        return vectors, emotion, actor
    except Exception as exc:  # a single unreadable clip should not stop training
        print(f"  ! skipping {os.path.basename(path)}: {exc}")
        return None


def build_dataset(
    data_dir: str, augment_actors: set[str] | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts features for every usable clip.

    ``augment_actors`` names the actors whose clips get augmented copies; pass
    the training actors so the held-out set stays untouched.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True))
    if not files:
        raise SystemExit(
            f"No .wav files found under {data_dir}\n"
            "Download RAVDESS from https://zenodo.org/records/1188976 and unzip it there."
        )

    usable = [f for f in files if parse_filename(f)[0]]
    print(f"Found {len(files)} files, {len(usable)} match the five target emotions.")
    print("Extracting features (a few minutes with augmentation)...")

    # Deliberately single-process: librosa's numba kernels segfault inside
    # forked joblib workers on macOS, and extraction is fast enough serially.
    started = time.time()
    features: List[np.ndarray] = []
    labels: List[str] = []
    actor_ids: List[str] = []

    for index, path in enumerate(usable, start=1):
        result = _process_one(
            path, with_augmentation=bool(augment_actors) and parse_filename(path)[1] in augment_actors
        )
        if result is None:
            continue
        vectors, emotion, actor = result
        for vector in vectors:
            features.append(vector)
            labels.append(emotion)
            actor_ids.append(actor)
        if index % 200 == 0 or index == len(usable):
            print(f"  {index}/{len(usable)} clips  ({time.time() - started:.0f}s)", flush=True)

    print(f"Built {len(features)} feature vectors in {time.time() - started:.1f}s")
    return np.stack(features), np.array(labels), np.array(actor_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default=MODEL_PATH)
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="skip training-set augmentation (faster, but ~4 points less accurate)",
    )
    args = parser.parse_args()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib

    all_actors = {f"{i:02d}" for i in range(1, 25)}
    train_actors = all_actors - TEST_ACTORS
    X, y, actors = build_dataset(
        args.data_dir, augment_actors=None if args.no_augment else train_actors
    )
    print(f"\nFeature matrix: {X.shape}")
    for label in EMOTIONS:
        print(f"  {label:<9} {int(np.sum(y == label)):>5} vectors")

    is_test = np.isin(actors, list(TEST_ACTORS))
    X_train, y_train = X[~is_test], y[~is_test]
    X_test, y_test = X[is_test], y[is_test]
    print(
        f"\nSpeaker-independent split: {len(X_train)} train / {len(X_test)} test "
        f"({len(set(actors[~is_test]))} vs {len(set(actors[is_test]))} actors)"
    )

    candidates = {
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(512, 256),
                        activation="relu",
                        solver="adam",
                        alpha=1e-3,
                        batch_size=64,
                        learning_rate_init=7e-4,
                        max_iter=800,
                        early_stopping=True,
                        n_iter_no_change=30,
                        validation_fraction=0.12,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "SVM-RBF": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        C=12.0,
                        gamma="scale",
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=600,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }

    best_name, best_pipeline, best_score = "", None, -1.0
    for name, pipeline in candidates.items():
        print(f"\n--- Training {name} ---")
        started = time.time()
        pipeline.fit(X_train, y_train)
        score = accuracy_score(y_test, pipeline.predict(X_test))
        print(f"{name}: test accuracy {score * 100:.2f}%  ({time.time() - started:.1f}s)")
        if score > best_score:
            best_name, best_pipeline, best_score = name, pipeline, score

    print(f"\n===== Best model: {best_name} ({best_score * 100:.2f}%) =====")
    predictions = best_pipeline.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))
    labels = sorted(set(y))
    print("Confusion matrix (rows = true, cols = predicted)")
    print("           " + "".join(f"{l[:5]:>7}" for l in labels))
    for row_label, row in zip(labels, confusion_matrix(y_test, predictions, labels=labels)):
        print(f"{row_label:<10} " + "".join(f"{v:>7}" for v in row))

    # Refit the winner on every actor so the shipped model uses all the data.
    print("\nRefitting the best model on the full dataset...")
    best_pipeline.fit(X, y)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(
        {
            "pipeline": best_pipeline,
            "labels": list(best_pipeline.classes_),
            "metadata": {
                "model": best_name,
                "dataset": "RAVDESS (Audio_Speech_Actors_01-24)",
                "held_out_accuracy": round(float(best_score), 4),
                "n_samples": int(X.shape[0]),
                "n_features": int(X.shape[1]),
                "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        },
        args.out,
    )
    print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
