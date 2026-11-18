"""Hand detection (MediaPipe) and the crop geometry shared by the app and the evaluation script.

Most ISL alphabet signs use BOTH hands, so the crop covers every detected hand that belongs to
the sign, not just the largest one.

The crop is deliberately NOT square. The training images are tight, non-square crops around the
hands that were stretched to 224x224, so the model expects that distortion. The app reproduces
it: it frames the hands with the margins measured on the dataset, then stretches the crop to
224x224 (in model.to_model_input).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Space kept around the hands' bounding box, as fractions of the box's width (left, right) and
# height (top, bottom). Calibrated on the RealSign Validation split by comparing MediaPipe's hand
# box with each training image (median margins: left 0.42, top 0.27, right 0.24, bottom 0.31),
# then made symmetric so left- and right-handed signers are treated alike. On the Testing split
# placed in a larger frame, every margin setting between 0.25 and 0.40 scored 84.5-85.5%, against
# 80.6% for a square crop that preserves the aspect ratio. What matters is the stretched,
# dataset-like framing, not the exact margins.
CONTEXT_MARGINS = (0.33, 0.29, 0.33, 0.29)  # left, top, right, bottom

# A second hand is treated as part of the same sign when the gap between the two hand boxes
# is at most this many times the larger hand's longer side (ISL two-handed letters touch or overlap).
DEFAULT_MAX_GAP_RATIO = 1.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in pixel coordinates; x2/y2 are exclusive."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return max(0, self.w) * max(0, self.h)

    def union(self, other: Box) -> Box:
        return Box(min(self.x1, other.x1), min(self.y1, other.y1),
                   max(self.x2, other.x2), max(self.y2, other.y2))

    def gap_to(self, other: Box) -> float:
        """Shortest distance between the two boxes (0 if they touch or overlap)."""
        dx = max(0, max(self.x1, other.x1) - min(self.x2, other.x2))
        dy = max(0, max(self.y1, other.y1) - min(self.y2, other.y2))
        return float(np.hypot(dx, dy))


def landmarks_to_box(points_xy: list[tuple[float, float]], frame_w: int, frame_h: int) -> Box:
    """Bounding box of normalized (0..1) landmark coordinates, clamped to the frame.

    MediaPipe can report landmarks slightly outside the image; clamping keeps every box
    (and therefore every crop) at least partly inside the frame.
    """
    xs = [min(max(x, 0.0), 1.0) * frame_w for x, _ in points_xy]
    ys = [min(max(y, 0.0), 1.0) * frame_h for _, y in points_xy]
    x1, y1 = int(min(xs)), int(min(ys))
    x2, y2 = int(np.ceil(max(xs))), int(np.ceil(max(ys)))
    # guarantee a non-empty box even for degenerate landmark sets
    return Box(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


def select_sign_boxes(boxes: list[Box], max_gap_ratio: float = DEFAULT_MAX_GAP_RATIO) -> list[Box]:
    """The largest hand plus any other hand close enough to be part of the same sign.

    This keeps two-handed letters together while ignoring, say, someone else's hand
    elsewhere in the frame.
    """
    if not boxes:
        return []
    main = max(boxes, key=lambda b: b.area)
    limit = max_gap_ratio * max(main.w, main.h)
    return [main] + [b for b in boxes if b is not main and main.gap_to(b) <= limit]


def union_box(boxes: list[Box]) -> Box:
    if not boxes:
        raise ValueError("union_box needs at least one box")
    out = boxes[0]
    for b in boxes[1:]:
        out = out.union(b)
    return out


def context_box(box: Box, frame_w: int, frame_h: int,
                margins: tuple[float, float, float, float] = CONTEXT_MARGINS) -> Box:
    """Expand the hands' box by `margins` (left, top, right, bottom; fractions of its width/height)
    and clip it to the frame, like a person cropping a training image out of a video frame.

    The result always lies inside the frame and always contains the (clamped) hand box.
    """
    left, top, right, bottom = margins
    x1 = max(0, box.x1 - round(left * box.w))
    y1 = max(0, box.y1 - round(top * box.h))
    x2 = min(frame_w, box.x2 + round(right * box.w))
    y2 = min(frame_h, box.y2 + round(bottom * box.h))
    # a hand box from landmarks_to_box is already inside the frame; guard anyway
    x1, y1 = min(x1, frame_w - 1), min(y1, frame_h - 1)
    return Box(x1, y1, max(x2, x1 + 1), max(y2, y1 + 1))


def crop_box(frame: np.ndarray, box: Box) -> np.ndarray:
    """Cut `box` out of `frame`. Any part of the box outside the frame is filled with black,
    so the result always has shape (box.h, box.w, channels)."""
    h, w = frame.shape[:2]
    out = np.zeros((box.h, box.w) + frame.shape[2:], dtype=frame.dtype)
    sx1, sy1 = max(0, box.x1), max(0, box.y1)
    sx2, sy2 = min(w, box.x2), min(h, box.y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - box.y1:sy2 - box.y1, sx1 - box.x1:sx2 - box.x1] = frame[sy1:sy2, sx1:sx2]
    return out


# Pairs of MediaPipe hand-landmark indices joined when drawing a hand (same as mp.solutions.hands).
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),  # little finger and palm
)


@dataclass
class Detection:
    landmarks: object  # mediapipe NormalizedLandmarkList, used for drawing
    box: Box


class HandTracker:
    """Thin wrapper around MediaPipe Hands that returns pixel boxes for each detected hand."""

    def __init__(self, static_image_mode: bool = False, max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        import mediapipe as mp  # imported lazily so the geometry helpers work without it

        self._mp_hands = mp.solutions.hands
        self._draw = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        import cv2

        h, w = frame_bgr.shape[:2]
        results = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        detections = []
        for hand in results.multi_hand_landmarks or []:
            pts = [(lm.x, lm.y) for lm in hand.landmark]
            detections.append(Detection(hand, landmarks_to_box(pts, w, h)))
        return detections

    def draw(self, frame_bgr: np.ndarray, detection: Detection) -> None:
        self._draw.draw_landmarks(frame_bgr, detection.landmarks, self._mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        self._hands.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def sign_crop(frame_bgr: np.ndarray, detections: list[Detection],
              margins: tuple[float, float, float, float] = CONTEXT_MARGINS
              ) -> tuple[np.ndarray, Box] | None:
    """Crop the sign (all hands that belong to it) out of a frame. Returns (crop, box) or None.

    The crop keeps its natural (usually non-square) shape; it is stretched to the model's square
    input later, exactly as the training images were.
    """
    if not detections:
        return None
    h, w = frame_bgr.shape[:2]
    boxes = select_sign_boxes([d.box for d in detections])
    box = context_box(union_box(boxes), w, h, margins)
    return crop_box(frame_bgr, box), box
