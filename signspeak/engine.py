"""The per-session recognition pipeline shared by the web app and webcam.py.

    frame -> hands (MediaPipe) -> crop -> classify -> smooth -> compose text

RecognitionEngine.process() takes one camera frame (BGR, unmirrored) and returns a FrameResult:
plain data that the web server serialises to JSON and webcam.py draws with OpenCV. Coordinates
in the result are normalised to 0..1 of the frame and are NOT mirrored (a mirrored display flips
x itself).
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

from signspeak.composer import ComposerConfig, TextComposer
from signspeak.hands import sign_crop
from signspeak.model import PredictionSmoother

# One TensorFlow model may be shared by several sessions (e.g. two browser tabs); serialise calls.
_PREDICT_LOCK = threading.Lock()


@dataclass
class EngineConfig:
    threshold: float = 0.60     # minimum smoothed confidence for a letter to count
    smoothing: int = 7          # frames averaged
    hold: float = 1.0           # seconds a letter must be held to be typed
    release: float = 0.3        # seconds a sign must be released before the same letter repeats
    auto_space: float = 2.0     # seconds without hands before a space; 0 disables
    flip_average: bool = True   # also classify the mirror image and average

    LIMITS = {"threshold": (0.0, 1.0), "smoothing": (1, 30), "hold": (0.0, 5.0),
              "release": (0.0, 3.0), "auto_space": (0.0, 10.0)}

    def updated(self, values: dict) -> EngineConfig:
        """A copy with `values` applied. Unknown keys are ignored; numbers are clamped to LIMITS;
        invalid values (wrong type, NaN) are ignored."""
        cfg = EngineConfig(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        for key, value in values.items():
            if key == "flip_average" and isinstance(value, bool):
                cfg.flip_average = value
            elif key in self.LIMITS and isinstance(value, (int, float)) and not isinstance(value, bool):
                if not math.isfinite(value):
                    continue
                lo, hi = self.LIMITS[key]
                v = min(max(value, lo), hi)
                setattr(cfg, key, int(round(v)) if key == "smoothing" else float(v))
        return cfg


@dataclass
class FrameResult:
    hands: list[list[tuple[float, float]]] = field(default_factory=list)  # 21 points per hand
    box: tuple[float, float, float, float] | None = None  # crop box x1, y1, x2, y2
    best: str | None = None          # most likely letter (even if not confident)
    confidence: float = 0.0          # smoothed probability of `best`
    letter: str | None = None        # `best` if confidence >= threshold, else None
    top: list[tuple[str, float]] = field(default_factory=list)
    typed: str | None = None         # character typed on this frame (letter or " ")
    text: str = ""
    progress: float = 0.0            # hold progress 0..1 for the current letter
    waiting_release: bool = False    # the letter shown was just typed; relax to repeat it
    crop: np.ndarray | None = None   # the crop the model classified (not serialised)

    def to_json(self) -> dict:
        r = lambda v: round(float(v), 4)  # noqa: E731
        return {
            "hands": [[[r(x), r(y)] for x, y in hand] for hand in self.hands],
            "box": [r(v) for v in self.box] if self.box else None,
            "best": self.best,
            "confidence": r(self.confidence),
            "letter": self.letter,
            "top": [[name, r(p)] for name, p in self.top],
            "typed": self.typed,
            "text": self.text,
            "progress": r(self.progress),
            "waiting_release": self.waiting_release,
        }


class RecognitionEngine:
    """One user's live session. Not thread-safe: call process() from one thread at a time."""

    def __init__(self, classifier, tracker, config: EngineConfig | None = None):
        self.classifier = classifier
        self.tracker = tracker
        self.config = config or EngineConfig()
        self.smoother = PredictionSmoother(self.config.smoothing)
        self.composer = TextComposer(self._composer_config())

    def _composer_config(self) -> ComposerConfig:
        c = self.config
        return ComposerConfig(hold_seconds=c.hold, release_seconds=c.release, auto_space_seconds=c.auto_space)

    def configure(self, values: dict) -> None:
        old = self.config
        self.config = old.updated(values)
        if self.config.smoothing != old.smoothing:
            self.smoother = PredictionSmoother(self.config.smoothing)
        self.composer.config = self._composer_config()

    def set_classifier(self, classifier) -> None:
        if len(classifier.class_names) == 0:
            raise ValueError("classifier has no classes")
        self.classifier = classifier
        self.smoother.reset()

    def process(self, frame_bgr: np.ndarray, now: float, compose: bool = True) -> FrameResult:
        """Run the pipeline on one frame. With compose=False the text is left untouched
        (used while practising or recording)."""
        h, w = frame_bgr.shape[:2]
        detections = self.tracker.detect(frame_bgr)
        res = FrameResult()
        res.hands = [[(float(lm.x), float(lm.y)) for lm in d.landmarks.landmark] for d in detections]
        if detections:
            crop, box = sign_crop(frame_bgr, detections)
            with _PREDICT_LOCK:
                probs = self.classifier.predict(crop, flip_average=self.config.flip_average)
            idx, conf = self.smoother.add(probs)
            names = self.classifier.class_names
            res.crop = crop
            res.box = (box.x1 / w, box.y1 / h, box.x2 / w, box.y2 / h)
            res.best, res.confidence = names[idx], conf
            res.top = [(names[i], p) for i, p in self.smoother.top(3)]
            if conf >= self.config.threshold:
                res.letter = res.best
        else:
            self.smoother.reset()

        if compose:
            res.typed = self.composer.update(res.letter, hand_visible=bool(detections), now=now)
            res.progress = self.composer.hold_progress(now)
            res.waiting_release = (self.composer.waiting_for_release and res.letter is not None
                                   and res.letter == self.composer.last_committed)
        res.text = self.composer.text
        return res

    # keyboard / button editing ------------------------------------------------
    def command(self, action: str) -> str | None:
        """Apply an editing action. Returns the finished word if a space was added (to speak)."""
        if action == "space":
            return self.composer.last_word() if self.composer.add_space() else None
        if action == "backspace":
            self.composer.backspace()
        elif action == "clear":
            self.composer.clear()
        else:
            raise ValueError(f"unknown command {action!r}")
        return None
