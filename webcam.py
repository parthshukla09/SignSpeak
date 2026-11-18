"""SignSpeak webcam app: fingerspell Indian Sign Language (ISL) letters, build words, hear them.

Run from anywhere:  python webcam.py            (see --help for options)

How to type
  * Hold a letter sign steady until the bar fills (1 s by default) -> the letter is typed.
  * To type the same letter twice (e.g. the LL in HELLO), drop or relax your hands briefly,
    then sign it again.
  * Lower your hands out of view for 2 s -> a space is added and the finished word is spoken.
  * The view is mirrored like a selfie camera. The "model sees" panel shows exactly what is
    classified, with the top 3 guesses.

Keys
  SPACE  add a space (finishes and speaks the word)    BACKSPACE  delete last character
  C      clear all text                                 ENTER      speak all text
  Q/Esc  quit (closing the window also quits)
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from signspeak.camera import FrameReader, open_camera
from signspeak.engine import EngineConfig, RecognitionEngine
from signspeak.hands import HAND_CONNECTIONS, HandTracker
from signspeak.model import CLASS_NAMES_PATH, MODEL_PATH, LetterClassifier
from signspeak.speech import Speaker

WINDOW = "SignSpeak"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
GREY = (190, 190, 190)
AMBER = (0, 200, 255)
TOP_BAR = 112
BOTTOM_BAR = 78

KEY_ESC, KEY_ENTER, KEY_BACKSPACE, KEY_SPACE = 27, 13, 8, 32


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-time ISL fingerspelling to text and speech.")
    p.add_argument("--camera", type=int, default=0, help="camera index (default 0)")
    p.add_argument("--model", default=str(MODEL_PATH), help="path to the .keras model")
    p.add_argument("--classes", default=str(CLASS_NAMES_PATH), help="path to class_names.json")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="minimum smoothed confidence for a letter to count (default 0.60)")
    p.add_argument("--smoothing", type=int, default=7,
                   help="number of frames whose predictions are averaged; 1 = none (default 7)")
    p.add_argument("--hold", type=float, default=1.0,
                   help="seconds a letter must be held to be typed (default 1.0)")
    p.add_argument("--release", type=float, default=0.3,
                   help="seconds a sign must be released before the same letter can repeat (default 0.3)")
    p.add_argument("--auto-space", type=float, default=2.0,
                   help="seconds without hands before a space is added; 0 disables (default 2.0)")
    p.add_argument("--no-speech", action="store_true", help="disable text-to-speech")
    p.add_argument("--speak-letters", action="store_true", help="also speak each letter as it is typed")
    p.add_argument("--no-mirror", action="store_true",
                   help="show the camera image unmirrored (the default is a mirror view, like a selfie camera)")
    p.add_argument("--no-flip-average", action="store_true",
                   help="classify only the camera image, not also its mirror image (a bit faster)")
    p.add_argument("--no-preview", action="store_true",
                   help="hide the 'model sees' preview and top-3 guesses")
    args = p.parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        p.error("--threshold must be between 0 and 1")
    if args.smoothing < 1:
        p.error("--smoothing must be at least 1 (1 = no smoothing)")
    for name in ("hold", "release", "auto_space"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be a non-negative number of seconds")
    return args


# ---------------------------------------------------------------------- drawing helpers
def darken(frame: np.ndarray, y1: int, y2: int, keep: float = 0.35) -> None:
    """Darken a horizontal band in place (cheaper than blending a full-frame overlay)."""
    band = frame[max(0, y1):max(0, y2)]
    band[:] = (band * keep).astype(np.uint8)


def fit_text(text: str, max_width: int, scale: float, thickness: int, keep: str = "end") -> str:
    """Trim text (from the start or the end) until it fits in max_width pixels."""
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_width:
        return text
    ell = "..."
    while text and cv2.getTextSize(ell + text if keep == "end" else text + ell,
                                   FONT, scale, thickness)[0][0] > max_width:
        text = text[1:] if keep == "end" else text[:-1]
    return ell + text if keep == "end" else text + ell


def put(frame, text, org, scale, color, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def draw_hands(frame, res) -> None:
    """Hand skeletons and the crop box, from the engine's normalised (unmirrored) coordinates."""
    h, w = frame.shape[:2]
    for hand in res.hands:
        pts = [(int(x * w), int(y * h)) for x, y in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], WHITE, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1, cv2.LINE_AA)
    if res.box is not None:
        x1, y1, x2, y2 = res.box
        cv2.rectangle(frame, (int(x1 * w), int(y1 * h)), (int(x2 * w) - 1, int(y2 * h) - 1), GREEN, 2)


PREVIEW = 120  # side of the "model sees" thumbnail in pixels


def draw_preview(frame, model_input_bgr, top3) -> None:
    """Top-right thumbnail of exactly what the model classifies, with its top guesses."""
    h, w = frame.shape[:2]
    lines = len(top3)
    panel_h = PREVIEW + 22 + 18 * lines
    x1, y1 = w - PREVIEW - 12, TOP_BAR + 8
    if x1 < 20 or y1 + panel_h > h - BOTTOM_BAR - 4:
        return  # frame too small for the panel
    darken(frame[:, x1 - 4:x1 + PREVIEW + 4], y1 - 4, y1 + panel_h)
    frame[y1 + 18:y1 + 18 + PREVIEW, x1:x1 + PREVIEW] = cv2.resize(model_input_bgr, (PREVIEW, PREVIEW))
    put(frame, "model sees", (x1, y1 + 13), 0.42, GREY)
    for i, (name, p) in enumerate(top3):
        put(frame, f"{name}  {p * 100:3.0f}%", (x1, y1 + 18 + PREVIEW + 16 + 18 * i), 0.5,
            WHITE if i == 0 else GREY, 1)


def draw_ui(frame, *, status, status_color, progress, waiting_release, text, fps, speech,
            preview=None, top3=None):
    h, w = frame.shape[:2]
    darken(frame, 0, TOP_BAR)
    darken(frame, h - BOTTOM_BAR, h)

    title = "SignSpeak - ISL"
    put(frame, title, (20, 30), 0.7, WHITE, 2)
    title_w = cv2.getTextSize(title, FONT, 0.7, 2)[0][0]
    for right in (f"{fps:4.1f} FPS | speech {speech}", f"{fps:.0f} FPS"):
        rw = cv2.getTextSize(right, FONT, 0.5, 1)[0][0]
        if w - rw - 20 >= 30 + title_w:  # only if it does not collide with the title
            put(frame, right, (w - rw - 20, 28), 0.5, GREY)
            break

    scale = 1.3
    while scale > 0.7 and cv2.getTextSize(status, FONT, scale, 3)[0][0] > w - 40:
        scale -= 0.1
    put(frame, fit_text(status, w - 40, scale, 3, keep="start"), (20, 80), scale, status_color, 3)
    bar_w = max(0, min(320, w - 40))
    cv2.rectangle(frame, (20, 94), (20 + bar_w, 102), (80, 80, 80), -1)
    if progress > 0:
        cv2.rectangle(frame, (20, 94), (20 + int(bar_w * progress), 102), GREEN, -1)
    hint_room = w - (30 + bar_w) - 10
    if waiting_release and hint_room >= 60:
        put(frame, fit_text("typed - relax hands to repeat", hint_room, 0.45, 1, keep="start"),
            (30 + bar_w, 103), 0.45, AMBER)

    if preview is not None:
        draw_preview(frame, preview, top3 or [])

    shown = fit_text(text + "_", w - 40, 0.9, 2, keep="end")
    put(frame, shown, (20, h - 42), 0.9, WHITE, 2)
    help_text = "SPACE space | BKSP delete | C clear | ENTER speak | Q quit"
    scale = 0.45
    while scale > 0.3 and cv2.getTextSize(help_text, FONT, scale, 1)[0][0] > w - 40:
        scale -= 0.02
    put(frame, help_text, (20, h - 14), scale, GREY)


# ---------------------------------------------------------------------- main loop
def main(argv=None) -> int:
    args = parse_args(argv)

    print("Loading model...", flush=True)
    classifier = LetterClassifier(args.model, args.classes)
    classifier.warm_up()  # the first predict() takes seconds; do it before the camera starts
    config = EngineConfig(threshold=args.threshold, smoothing=args.smoothing, hold=args.hold,
                          release=args.release, auto_space=args.auto_space,
                          flip_average=not args.no_flip_average)
    speaker = Speaker(enabled=not args.no_speech)
    if not args.no_speech and not speaker.available:
        if speaker.starting:
            print(f"Text-to-speech: {speaker.error}.")
        else:
            print(f"Text-to-speech unavailable ({speaker.error}); continuing without speech.")

    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not get video from camera {args.camera}. Is it connected, switched on, and "
              "not used by another app? (If another app just closed it, wait a moment and retry.)")
        speaker.close()
        return 1
    reader = FrameReader(cap)

    tracker = HandTracker(max_num_hands=2)
    engine = RecognitionEngine(classifier, tracker, config)
    composer = engine.composer
    fps, last_t = 0.0, time.perf_counter()
    window_shown = False
    print("Running. Press Q or Esc in the video window to quit.", flush=True)

    try:
        while True:
            frame = reader.read()
            if frame is None:
                print("Camera stopped delivering frames.")
                break

            now = time.perf_counter()
            dt = now - last_t
            last_t = now
            fps = (0.9 * fps + 0.1 / dt) if fps else 1.0 / max(dt, 1e-6)

            res = engine.process(frame, now)  # crops BEFORE anything is drawn on the frame
            preview = top3 = None
            if res.best is None:
                status, status_color = "Show your hands", GREY
            elif res.letter:
                status, status_color = f"{res.best}  {res.confidence * 100:.0f}%", GREEN
            else:
                status, status_color = f"Uncertain ({res.best} {res.confidence * 100:.0f}%)", AMBER
            if res.crop is not None and not args.no_preview:
                # the exact stretched image the model classifies
                preview = cv2.resize(res.crop, (classifier.input_size, classifier.input_size))
                if not args.no_mirror:
                    preview = cv2.flip(preview, 1)  # match the mirrored view on screen
                top3 = res.top
            draw_hands(frame, res)

            if not args.no_mirror:
                # mirror only what is shown (after the hand overlay, before the text);
                # the model already received the unmirrored crop
                frame = cv2.flip(frame, 1)

            if res.typed == " ":
                speaker.say(composer.last_word())
            elif res.typed and args.speak_letters:
                speaker.say(res.typed)

            speech = "on" if speaker.available else ("starting" if speaker.starting else "off")
            draw_ui(frame, status=status, status_color=status_color, progress=res.progress,
                    waiting_release=res.waiting_release, text=res.text, fps=fps, speech=speech,
                    preview=preview, top3=top3)
            cv2.imshow(WINDOW, frame)
            window_shown = True

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key == KEY_SPACE:
                word = engine.command("space")
                if word:
                    speaker.say(word)
            elif key == KEY_BACKSPACE:
                engine.command("backspace")
            elif key in (ord("c"), ord("C")):
                engine.command("clear")
            elif key == KEY_ENTER:
                speaker.say(composer.text)

            if window_shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # window closed with the X button
    finally:
        cv2.destroyAllWindows()  # close the window first; speaker.close() may wait up to 2 s
        cap.release()
        tracker.close()
        speaker.close()

    if composer.text.strip():
        print(f"Final text: {composer.text.strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
