"""Listing dataset images, shared by train.py and evaluate.py.

The RealSign archive contains a couple of files named *.jpg that are not images (e.g.
Testing/E/301.jpg is a Python script), so files are checked by their header, not just their name.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_MAGIC = (b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"BM")


def is_image_file(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(8)
    except OSError:
        return False
    return head.startswith(_MAGIC)


def list_split(split_dir: Path, class_names: list[str],
               limit_per_class: int = 0) -> tuple[list[tuple[Path, int]], list[Path]]:
    """Return ([(image_path, class_index), ...], [skipped non-image files]).

    With limit_per_class > 0, keeps at most that many evenly spaced files per class
    (deterministic, so repeated runs use the same images).
    """
    split_dir = Path(split_dir)
    missing = [c for c in class_names if not (split_dir / c).is_dir()]
    if missing:
        sys.exit(f"{split_dir} is missing class folders: {missing}")
    items, skipped = [], []
    for ci, cname in enumerate(class_names):
        files = []
        for p in sorted((split_dir / cname).iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            (files if is_image_file(p) else skipped).append(p)
        if limit_per_class > 0 and len(files) > limit_per_class:
            idx = np.linspace(0, len(files) - 1, limit_per_class).round().astype(int)
            files = [files[i] for i in idx]
        items += [(f, ci) for f in files]
    return items, skipped
