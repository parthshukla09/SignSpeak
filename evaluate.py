"""Measure a SignSpeak model's accuracy on a dataset split (default: RealSign Testing).

Examples
  python evaluate.py                                  # whole test split, images as stored
  python evaluate.py --crop hands                     # same, but through the webcam app's hand crop
  python evaluate.py --split Validation --limit-per-class 50
  python evaluate.py --model runs/my_run/SignSpeak.keras --classes runs/my_run/class_names.json

--crop none   feeds each dataset image as-is (the dataset images are already cropped around
              the hands), which is how the model was trained.
--crop hands  simulates the webcam app: each image is placed in a larger frame (its edges
              extended), MediaPipe finds the hands, and the app's exact crop is applied
              (images where no hand is found fall back to the whole image). This is the
              better estimate of live accuracy, and it checks that the app's preprocessing
              agrees with the training data.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from signspeak.dataset import list_split
from signspeak.model import CLASS_NAMES_PATH, MODEL_PATH, PROJECT_ROOT, LetterClassifier


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(PROJECT_ROOT / "data" / "RealSign"),
                   help="dataset root containing Training/Validation/Testing")
    p.add_argument("--split", default="Testing", help="split folder to evaluate (default Testing)")
    p.add_argument("--model", default=str(MODEL_PATH))
    p.add_argument("--classes", default=str(CLASS_NAMES_PATH))
    p.add_argument("--crop", choices=["none", "hands"], default="none")
    p.add_argument("--flip-average", action="store_true",
                   help="average each prediction with its mirror image (the webcam app's default)")
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="evaluate at most N evenly spaced images per class (0 = all)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--confusion-csv", help="write the confusion matrix to this CSV file")
    p.add_argument("--json", help="write a summary (accuracy, per-class accuracy) to this JSON file")
    args = p.parse_args(argv)
    if args.limit_per_class < 0:
        p.error("--limit-per-class must be 0 (all images) or more")
    if args.batch_size < 1:
        p.error("--batch-size must be at least 1")
    return args


def simulated_frame(img: np.ndarray, extend: float = 0.6) -> np.ndarray:
    """Place a tightly cropped dataset image in a larger 'camera frame' by extending its edges,
    so MediaPipe sees whole hands and the app's crop has surroundings to take margins from."""
    p = int(extend * max(img.shape[:2]))
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REPLICATE)


def main(argv=None) -> int:
    args = parse_args(argv)
    split_dir = Path(args.data) / args.split
    if not split_dir.is_dir():
        sys.exit(f"Dataset split not found: {split_dir}\n"
                 "Download it first - see 'Dataset' in README.md.")

    clf = LetterClassifier(args.model, args.classes, flip_average=args.flip_average)
    names = clf.class_names
    items, skipped = list_split(split_dir, names, args.limit_per_class)
    print(f"Evaluating {len(items)} images from {split_dir} (crop={args.crop})", flush=True)
    for p in skipped:
        print(f"  skipping {p}: not an image file")

    tracker = None
    if args.crop == "hands":
        from signspeak.hands import HandTracker, sign_crop
        tracker = HandTracker(static_image_mode=True, max_num_hands=2)

    y_true, y_pred, unreadable, no_hand = [], [], 0, 0
    batch, batch_labels = [], []
    t0 = time.perf_counter()

    def flush():
        if batch:
            y_pred.extend(clf.predict_batch(batch, args.batch_size).argmax(axis=1).tolist())
            y_true.extend(batch_labels)
            batch.clear()
            batch_labels.clear()

    for i, (path, label) in enumerate(items, 1):
        img = cv2.imread(str(path))
        if img is None:
            unreadable += 1
            continue
        if tracker is not None:
            frame = simulated_frame(img)
            result = sign_crop(frame, tracker.detect(frame))
            if result is None:
                no_hand += 1
            else:
                img = result[0]
        batch.append(img)
        batch_labels.append(label)
        if len(batch) >= args.batch_size:
            flush()
        if i % 500 == 0:
            print(f"  {i}/{len(items)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    flush()
    if tracker is not None:
        tracker.close()

    if not y_true:
        sys.exit(f"No readable images found in {split_dir}")
    y_true, y_pred = np.array(y_true, dtype=int), np.array(y_pred, dtype=int)
    n = len(names)
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    acc = float((y_true == y_pred).mean())
    # None for classes without images (keeps the JSON valid and the sort well-defined)
    per_class = {names[c]: (float(cm[c, c] / cm[c].sum()) if cm[c].sum() else None)
                 for c in range(n)}

    print(f"\nAccuracy: {acc * 100:.2f}%  ({int((y_true == y_pred).sum())}/{len(y_true)})"
          f"  in {time.perf_counter() - t0:.0f}s")
    if unreadable:
        print(f"Unreadable images skipped: {unreadable}")
    if tracker is not None:
        print(f"No hand detected (whole image used): {no_hand}/{len(y_true)}")

    print("\nPer-class accuracy (worst first):")
    for name, a in sorted(per_class.items(), key=lambda kv: (kv[1] is None, kv[1] or 0.0)):
        print(f"  {name}: " + ("no images" if a is None else f"{a * 100:5.1f}%"))

    off = [(cm[i, j], names[i], names[j]) for i in range(n) for j in range(n) if i != j and cm[i, j]]
    if off:
        print("\nMost common mistakes (true -> predicted):")
        for count, t, p in sorted(off, reverse=True)[:10]:
            print(f"  {t} -> {p}: {count}")

    if args.confusion_csv:
        with open(args.confusion_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["true\\pred"] + names)
            for i in range(n):
                w.writerow([names[i]] + cm[i].tolist())
        print(f"\nConfusion matrix written to {args.confusion_csv}")
    if args.json:
        Path(args.json).write_text(json.dumps({
            "split": str(split_dir), "crop": args.crop, "images": int(len(y_true)),
            "accuracy": acc, "per_class": per_class,
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
