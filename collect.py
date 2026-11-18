"""Record examples of YOUR signs, so the model can be fine-tuned to your hands (see finetune.py).

  python collect.py --letters ALL --per-take 20   # recommended: the whole alphabet
  python collect.py --letters CLOV                # only some letters
  python collect.py                               # D P N M T K

Record ALL letters if you can. If only some letters are recorded, the fine-tuned model can learn
"an image from this camera is one of the recorded letters", and other letters signed in front of
your camera then get read as them (e.g. F->D). Letters that already have enough takes are
skipped, so you can run this again to fill in the rest.

For each letter you record a few short takes (3 by default). During a take, hold the sign and
move a little (slightly closer/further, left/right, tilt) so the examples vary. Relax your hands
between takes. The window shows the dataset's reference photo of each letter, so you can compare
your sign with the one the model learned.

Each saved image is exactly the crop the app classifies. They go to data/user/<LETTER>/
(ignored by git). Re-running adds new takes, it never overwrites.

Keys: SPACE start the next take | N skip this letter | BACKSPACE delete the last take
      Q / Esc quit (everything recorded so far is kept)
"""
from __future__ import annotations

import argparse
import sys
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from signspeak.camera import FrameReader, open_camera
from signspeak.hands import HandTracker, sign_crop
from signspeak.model import CLASS_NAMES_PATH, PROJECT_ROOT, load_class_names
from signspeak.userdata import next_take_number

WINDOW = "SignSpeak - record your signs"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN, WHITE, GREY, AMBER, RED = (0, 255, 0), (255, 255, 255), (190, 190, 190), (0, 200, 255), (60, 60, 255)
KEY_ESC, KEY_BACKSPACE, KEY_SPACE = 27, 8, 32
COUNTDOWN_S = 1.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--letters", default="DPNMTK", help="letters to record, or ALL (default DPNMTK)")
    p.add_argument("--takes", type=int, default=3, help="takes per letter (default 3; the last is held out)")
    p.add_argument("--per-take", type=int, default=30, help="images saved per take (default 30)")
    p.add_argument("--interval", type=float, default=0.12, help="seconds between saved images (default 0.12)")
    p.add_argument("--out", default=str(PROJECT_ROOT / "data" / "user"), help="where to save (default data/user)")
    p.add_argument("--reference", default=str(PROJECT_ROOT / "data" / "RealSign" / "Letters"),
                   help="folder with the dataset's reference photo per letter (optional)")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--classes", default=str(CLASS_NAMES_PATH))
    p.add_argument("--redo", action="store_true",
                   help="also record letters that already have --takes takes (default: skip them)")
    args = p.parse_args(argv)
    names = load_class_names(args.classes)
    letters = names if args.letters.upper() == "ALL" else [c for c in args.letters.upper() if not c.isspace()]
    unknown = [c for c in letters if c not in names]
    if unknown:
        p.error(f"unknown letters {unknown}; the model knows {''.join(names)}")
    if not letters:
        p.error("no letters given")
    if args.takes < 1 or args.per_take < 1 or args.interval < 0:
        p.error("--takes and --per-take must be at least 1, --interval must be >= 0")
    args.letter_list = list(dict.fromkeys(letters))  # keep order, drop duplicates
    return args


def put(frame, text, org, scale, color, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


@lru_cache(maxsize=None)
def load_reference(folder: Path, letter: str, size: int = 160) -> np.ndarray | None:
    for ext in (".jpg", ".jpeg", ".png"):
        img = cv2.imread(str(folder / f"{letter}{ext}"))
        if img is not None:
            h, w = img.shape[:2]
            s = size / max(h, w)
            return cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    return None


def main(argv=None) -> int:
    args = parse_args(argv)
    out_root = Path(args.out)
    ref_dir = Path(args.reference)
    letters = args.letter_list
    if not args.redo:
        done = [c for c in letters if next_take_number(out_root / c) - 1 >= args.takes]
        letters = [c for c in letters if c not in done]
        if done:
            print(f"Already recorded (skipping, use --redo to add more): {''.join(done)}")
        if not letters:
            print("Nothing left to record.")
            return 0
    # one entry per finished take: (letter, files, letter index, takes done before it), for BACKSPACE
    session: list[tuple[str, list[Path], int, int]] = []
    take_no = 0

    cap = open_camera(args.camera)
    if cap is None:
        print(f"Could not get video from camera {args.camera}. Is another app using it?")
        return 1
    reader = FrameReader(cap)
    tracker = HandTracker(max_num_hands=2)
    li, takes_done = 0, 0
    mode, mode_since, saved, last_save = "wait", 0.0, [], 0.0
    message = ""
    print(f"Recording {''.join(letters)}: {args.takes} takes x {args.per_take} images -> {out_root}")

    try:
        while li < len(letters):
            frame = reader.read()
            if frame is None:
                print("Camera stopped delivering frames.")
                break
            now = time.perf_counter()
            letter = letters[li]
            detections = tracker.detect(frame)
            result = sign_crop(frame, detections)

            if mode == "countdown" and now - mode_since >= COUNTDOWN_S:
                mode, saved, last_save = "record", [], 0.0
            if mode == "record" and result is not None and now - last_save >= args.interval:
                d = out_root / letter
                d.mkdir(parents=True, exist_ok=True)
                if not saved:
                    take_no = next_take_number(d)
                path = d / f"take{take_no:02d}_{len(saved) + 1:03d}.jpg"
                if cv2.imwrite(str(path), result[0], [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    saved.append(path)
                    last_save = now
                if len(saved) >= args.per_take:
                    session.append((letter, saved, li, takes_done))
                    takes_done += 1
                    message = f"Saved take {takes_done} of {letter} ({len(saved)} images)"
                    mode = "wait"
                    if takes_done >= args.takes:
                        li, takes_done = li + 1, 0

            # ---------------- draw (mirrored view, like the app)
            for det in detections:
                tracker.draw(frame, det)
            if result is not None:
                b = result[1]
                cv2.rectangle(frame, (b.x1, b.y1), (b.x2 - 1, b.y2 - 1),
                              RED if mode == "record" else GREEN, 2)
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            band = frame[:120]
            band[:] = (band * 0.35).astype(np.uint8)
            if li < len(letters):
                put(frame, f"Sign: {letters[li]}", (20, 50), 1.5, WHITE, 3)
                put(frame, f"take {takes_done + 1}/{args.takes}   letter {li + 1}/{len(letters)}",
                    (20, 80), 0.55, GREY)
                if mode == "wait":
                    status, color = "Press SPACE when your sign is ready", AMBER
                elif mode == "countdown":
                    status, color = f"Get ready... {COUNTDOWN_S - (now - mode_since):.1f}", AMBER
                else:
                    status = f"Recording {len(saved)}/{args.per_take} - hold it, move a little"
                    color = RED if result is not None else AMBER
                    if result is None:
                        status = "Recording paused - show both hands"
                    cv2.rectangle(frame, (20, 100), (20 + int(300 * len(saved) / args.per_take), 108), RED, -1)
                put(frame, status, (20, 98 if mode != "record" else 95), 0.55, color)
                ref = load_reference(ref_dir, letters[li])
                if ref is not None and ref.shape[0] + 150 < h and ref.shape[1] + 20 < w:
                    rh, rw = ref.shape[:2]
                    frame[130:130 + rh, w - rw - 10:w - 10] = ref
                    put(frame, "reference", (w - rw - 10, 125), 0.45, GREY)
            if message:
                put(frame, message, (20, h - 40), 0.55, GREEN)
            put(frame, "SPACE record | N skip letter | BKSP delete last take | Q quit",
                (20, h - 14), 0.45, GREY)
            cv2.imshow(WINDOW, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), KEY_ESC):
                break
            if key == KEY_SPACE and mode == "wait":
                mode, mode_since, message = "countdown", now, ""
            elif key in (ord("n"), ord("N")) and mode == "wait":
                li, takes_done, message = li + 1, 0, f"Skipped {letter}"
            elif key == KEY_BACKSPACE and mode == "wait" and session:
                gone_letter, files, li, takes_done = session.pop()  # go back to before that take
                for f in files:
                    f.unlink(missing_ok=True)
                message = f"Deleted the last take of {gone_letter} - record it again"
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
        cap.release()
        tracker.close()

    if mode == "record" and saved:  # quit mid-take: drop the incomplete take
        for f in saved:
            f.unlink(missing_ok=True)
    total = sum(len(entry[1]) for entry in session)
    print(f"Recorded {len(session)} takes, {total} images in {out_root}")
    if session:
        print("Next: python finetune.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
