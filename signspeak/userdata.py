"""Your own recorded signs (from collect.py), used by finetune.py.

Layout: <root>/<LETTER>/take<NN>_<NNN>.jpg. A "take" is one continuous recording of a letter.
Frames within a take are near-duplicates, so train/validation are split by take, never by frame:
the last take of each letter is held out for validation (when the letter has two or more takes).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from signspeak.dataset import is_image_file

TAKE_RE = re.compile(r"^take(\d+)_\d+$")


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    letter: str
    take: int


def take_of(path: Path) -> int | None:
    m = TAKE_RE.match(path.stem)
    return int(m.group(1)) if m else None


def next_take_number(letter_dir: Path) -> int:
    takes = [t for p in letter_dir.glob("take*_*.jpg") if (t := take_of(p)) is not None]
    return max(takes, default=0) + 1


def list_samples(root: Path | str, class_names: list[str]) -> list[Sample]:
    root = Path(root)
    samples = []
    for label, letter in enumerate(class_names):
        d = root / letter
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.jpg")):
            take = take_of(p)
            if take is not None and is_image_file(p):
                samples.append(Sample(p, label, letter, take))
    return samples


def summary(root: Path | str, class_names: list[str]) -> dict[str, dict[str, int]]:
    """{letter: {"takes": n, "images": m}} for every class (zeros for unrecorded letters)."""
    out = {c: {"takes": 0, "images": 0} for c in class_names}
    takes: dict[str, set[int]] = {}
    for s in list_samples(root, class_names):
        out[s.letter]["images"] += 1
        takes.setdefault(s.letter, set()).add(s.take)
    for letter, t in takes.items():
        out[letter]["takes"] = len(t)
    return out


def delete_last_take(root: Path | str, letter: str) -> int:
    """Delete the highest-numbered take of `letter`. Returns how many files were removed."""
    d = Path(root) / letter
    last = next_take_number(d) - 1
    if last < 1:
        return 0
    files = [p for p in d.glob(f"take{last:02d}_*.jpg") if take_of(p) == last]
    for p in files:
        p.unlink(missing_ok=True)
    return len(files)


class TakeRecorder:
    """Saves one take: up to `per_take` crops, at most one every `interval` seconds."""

    def __init__(self, root: Path | str, letter: str, per_take: int = 20, interval: float = 0.12):
        self.dir = Path(root) / letter
        self.letter = letter
        self.per_take = per_take
        self.interval = interval
        self.take = 0
        self.files: list[Path] = []
        self._last = float("-inf")

    @property
    def done(self) -> bool:
        return len(self.files) >= self.per_take

    def offer(self, crop_bgr, now: float) -> bool:
        """Save the crop if the interval has passed. Returns True if it was saved."""
        import cv2

        if self.done or now - self._last < self.interval:
            return False
        if not self.files:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.take = next_take_number(self.dir)
        path = self.dir / f"take{self.take:02d}_{len(self.files) + 1:03d}.jpg"
        if not cv2.imwrite(str(path), crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            return False
        self.files.append(path)
        self._last = now
        return True

    def cancel(self) -> None:
        """Delete everything saved by this (unfinished) take."""
        for p in self.files:
            p.unlink(missing_ok=True)
        self.files.clear()


def split_by_take(samples: list[Sample]) -> tuple[list[Sample], list[Sample]]:
    """(train, validation): the last take of each letter with 2+ takes goes to validation."""
    last_take: dict[str, int] = {}
    takes_per_letter: dict[str, set[int]] = {}
    for s in samples:
        takes_per_letter.setdefault(s.letter, set()).add(s.take)
    for letter, takes in takes_per_letter.items():
        if len(takes) >= 2:
            last_take[letter] = max(takes)
    train = [s for s in samples if last_take.get(s.letter) != s.take]
    val = [s for s in samples if last_take.get(s.letter) == s.take]
    return train, val
