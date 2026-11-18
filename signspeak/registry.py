"""The models the app can use: the original one and anything produced in runs/.

Model ids are "original" or "run:<folder name>". Clients only ever pass ids, never paths.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

from signspeak.model import CLASS_NAMES_PATH, MODEL_PATH, PROJECT_ROOT


@dataclass(frozen=True)
class ModelInfo:
    id: str
    label: str
    kind: str            # "original" | "personalised" | "trained"
    path: Path
    classes: Path
    summary: str | None
    modified: float

    def to_json(self) -> dict:
        return {"id": self.id, "label": self.label, "kind": self.kind, "summary": self.summary}


def _run_info(run: Path) -> ModelInfo | None:
    model, classes, metrics = run / "SignSpeak.keras", run / "class_names.json", run / "metrics.json"
    if not (model.is_file() and classes.is_file()):
        return None
    kind, summary = "trained", None
    try:
        m = json.loads(metrics.read_text(encoding="utf-8")) if metrics.is_file() else {}
    except (OSError, ValueError):
        m = {}
    if "after" in m:  # written by finetune.py
        kind = "personalised"
        acc = (m["after"].get("your_heldout_take") or {}).get("accuracy")
        n = len(m.get("recordings", {}))
        summary = (f"{n} letters recorded" + (f", held-out {acc * 100:.1f}%" if acc is not None else ""))
    elif "test_accuracy" in m:  # written by train.py
        summary = f"test accuracy {m['test_accuracy'] * 100:.1f}%"
    label = ("Personalised" if kind == "personalised" else "Trained") + f" · {run.name}"
    return ModelInfo(f"run:{run.name}", label, kind, model, classes, summary, model.stat().st_mtime)


def list_models(root: Path = PROJECT_ROOT) -> list[ModelInfo]:
    out = []
    original = root / MODEL_PATH.relative_to(PROJECT_ROOT)
    if original.is_file():
        out.append(ModelInfo("original", "Original (RealSign dataset)", "original", original,
                             root / CLASS_NAMES_PATH.relative_to(PROJECT_ROOT),
                             "4 signers, 26 letters", original.stat().st_mtime))
    runs = root / "runs"
    if runs.is_dir():
        found = [i for d in runs.iterdir() if d.is_dir() and (i := _run_info(d))]
        out += sorted(found, key=lambda i: i.modified, reverse=True)
    return out


def default_model_id(models: list[ModelInfo]) -> str | None:
    """The newest personalised model if there is one, else the original."""
    personal = [m for m in models if m.kind == "personalised"]
    if personal:
        return max(personal, key=lambda m: m.modified).id
    return next((m.id for m in models if m.kind == "original"), models[0].id if models else None)


class ModelCache:
    """Loads models on first use and keeps them (each is ~25 MB in memory)."""

    def __init__(self, loader=None):
        from signspeak.model import LetterClassifier

        self._loader = loader or (lambda info: LetterClassifier(info.path, info.classes, flip_average=True))
        self._models: dict[str, object] = {}
        self._lock = threading.Lock()

    def get(self, info: ModelInfo):
        with self._lock:
            if info.id not in self._models:
                clf = self._loader(info)
                if hasattr(clf, "warm_up"):
                    clf.warm_up()
                self._models[info.id] = clf
            return self._models[info.id]

    def forget(self, model_id: str) -> None:
        with self._lock:
            self._models.pop(model_id, None)
