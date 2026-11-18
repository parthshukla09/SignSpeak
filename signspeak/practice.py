"""Learn mode: the app shows a target letter; holding that sign for `hold` seconds counts as correct.

Pure logic (no camera, no model) so it can be unit-tested. Feed it the engine's confident letter
every frame.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class LetterStats:
    attempts: int = 0     # targets shown (finished or skipped)
    correct: int = 0
    best_seconds: float | None = None  # fastest time from showing the target to success

    def to_json(self) -> dict:
        return {"attempts": self.attempts, "correct": self.correct,
                "best_seconds": None if self.best_seconds is None else round(self.best_seconds, 2)}


@dataclass
class PracticeSession:
    letters: list[str]
    order: str = "random"          # "random" or "sequential"
    hold: float = 0.8
    seed: int | None = None
    target: str | None = None
    stats: dict[str, LetterStats] = field(default_factory=dict)
    streak: int = 0
    _rng: random.Random = field(init=False, repr=False)
    _index: int = field(default=-1, init=False, repr=False)
    _shown_at: float | None = field(default=None, init=False, repr=False)
    _match_since: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if not self.letters:
            raise ValueError("practice needs at least one letter")
        if self.order not in ("random", "sequential"):
            raise ValueError("order must be 'random' or 'sequential'")
        self._rng = random.Random(self.seed)
        self.stats = {c: LetterStats() for c in self.letters}
        self._advance()

    def _advance(self) -> None:
        if self.order == "sequential":
            self._index = (self._index + 1) % len(self.letters)
            self.target = self.letters[self._index]
        else:
            choices = [c for c in self.letters if c != self.target] or self.letters
            self.target = self._rng.choice(choices)
        self._shown_at = None
        self._match_since = None

    def update(self, letter: str | None, now: float) -> str | None:
        """Feed one frame. Returns "correct" on the frame the target is completed, else None."""
        if self._shown_at is None:
            self._shown_at = now
        if letter != self.target:
            self._match_since = None
            return None
        if self._match_since is None:
            self._match_since = now
        if now - self._match_since < self.hold:
            return None
        s = self.stats[self.target]
        s.attempts += 1
        s.correct += 1
        elapsed = now - self._shown_at
        s.best_seconds = elapsed if s.best_seconds is None else min(s.best_seconds, elapsed)
        self.streak += 1
        self._advance()
        return "correct"

    def progress(self, now: float) -> float:
        if self._match_since is None or self.hold <= 0:
            return 0.0
        return min(1.0, (now - self._match_since) / self.hold)

    def skip(self) -> None:
        self.stats[self.target].attempts += 1
        self.streak = 0
        self._advance()

    def to_json(self, now: float) -> dict:
        return {"target": self.target, "progress": round(self.progress(now), 3), "streak": self.streak,
                "stats": {c: s.to_json() for c, s in self.stats.items()}}
