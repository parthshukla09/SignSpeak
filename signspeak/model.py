"""Model loading, input preparation and prediction smoothing."""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "SignSpeak_finetuned.keras"
CLASS_NAMES_PATH = PROJECT_ROOT / "class_names.json"
INPUT_SIZE = 224


def load_class_names(path: Path | str = CLASS_NAMES_PATH) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = json.load(f)
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"{path} must contain a JSON list of class names")
    return names


def load_keras_model(path: Path | str = MODEL_PATH):
    """Load a SignSpeak .keras model.

    The original model contains a Lambda layer that calls MobileNetV2's preprocess_input by name,
    so it must be supplied here. Models produced by train.py use a Rescaling layer instead and
    simply ignore this argument.
    """
    import tensorflow as tf

    return tf.keras.models.load_model(
        path,
        custom_objects={"preprocess_input": tf.keras.applications.mobilenet_v2.preprocess_input},
    )


def to_model_input(image_bgr: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """BGR uint8 image -> RGB float32 (size, size, 3) in 0..255.

    Pixel scaling to [-1, 1] happens inside the model, so it must NOT be done here.
    """
    import cv2

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


class LetterClassifier:
    """Wraps the Keras model.

    flip_average=True averages each prediction with the prediction for the mirror image.
    The RealSign training photos mix mirrored and unmirrored captures, so the model handles both
    orientations about equally well (test accuracy 90.7% as stored, 91.1% flipped, 91.3%
    averaged). Averaging makes left- and right-handed signing, and mirrored or unmirrored
    cameras, equally reliable, for about 1.3x the compute.
    """

    def __init__(self, model_path: Path | str = MODEL_PATH,
                 class_names_path: Path | str = CLASS_NAMES_PATH, flip_average: bool = False):
        self.flip_average = flip_average
        self.class_names = load_class_names(class_names_path)
        self.model = load_keras_model(model_path)
        n_out = self.model.output_shape[-1]
        if n_out != len(self.class_names):
            raise ValueError(f"model has {n_out} outputs but {class_names_path} lists "
                             f"{len(self.class_names)} classes")
        # models from train.py --img-size N expect N x N input
        self.input_size = self.model.input_shape[1] or INPUT_SIZE

    def predict(self, crop_bgr: np.ndarray, flip_average: bool | None = None) -> np.ndarray:
        """Class probabilities for one cropped image."""
        return self.predict_batch([crop_bgr], flip_average=flip_average)[0]

    def predict_batch(self, crops_bgr: list[np.ndarray], batch_size: int = 32,
                      flip_average: bool | None = None) -> np.ndarray:
        """`flip_average` overrides the instance setting for this call (None = use the setting)."""
        batch = np.stack([to_model_input(c, self.input_size) for c in crops_bgr])
        if not (self.flip_average if flip_average is None else flip_average):
            # predict() measured ~2x faster than model(x) for single images on CPU
            return self.model.predict(batch, verbose=0, batch_size=batch_size)
        both = np.concatenate([batch, batch[:, :, ::-1]])  # originals, then mirror images
        probs = self.model.predict(both, verbose=0, batch_size=batch_size)
        return (probs[:len(batch)] + probs[len(batch):]) / 2

    def warm_up(self, runs: int = 2) -> None:
        """Run dummy predictions so the first real frame is not delayed by graph building
        (the first predict() takes 1-5 s)."""
        blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(runs):
            self.predict(blank)


class PredictionSmoother:
    """Averages class probabilities over the last `window` frames to steady the output.
    window=1 means no smoothing."""

    def __init__(self, window: int = 7):
        self._history: deque[np.ndarray] = deque(maxlen=max(1, window))
        self.average: np.ndarray | None = None  # latest smoothed probabilities

    def add(self, probs: np.ndarray) -> tuple[int, float]:
        self._history.append(np.asarray(probs, dtype=np.float32))
        self.average = np.mean(self._history, axis=0)
        idx = int(np.argmax(self.average))
        return idx, float(self.average[idx])

    def top(self, k: int = 3) -> list[tuple[int, float]]:
        """The k most likely classes of the latest smoothed prediction, as (index, probability)."""
        if self.average is None:
            return []
        order = np.argsort(self.average)[::-1][:k]
        return [(int(i), float(self.average[i])) for i in order]

    def reset(self) -> None:
        self._history.clear()
        self.average = None

    def __len__(self) -> int:
        return len(self._history)
