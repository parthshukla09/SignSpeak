"""Turns a noisy stream of per-frame letter predictions into typed text.

Rules
-----
* Hold to confirm: a letter is typed only after it has been the confident prediction
  continuously for `hold_seconds` AND for at least `min_frames` frames (so a stalled frame
  rate cannot type a letter from one or two frames). Brief mispredictions while the hands
  move between signs therefore never get typed.
* Release before repeating: after a letter is typed, nothing more is typed until the sign is
  released (hands leave the frame, the prediction becomes uncertain, or a different letter
  appears) for at least `release_seconds`. Signing the same letter again after a release
  types it again - that is how double letters such as the "LL" in HELLO are entered.
* Auto space: when no hand has been visible for `auto_space_seconds` after a letter, a space
  is typed, which finishes the word. It fires once per word. Typing a space, deleting a
  space, or clearing the text cancels it until another letter is signed. Deleting a letter
  does not, so the corrected word still gets its space. Set it to 0 to disable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ComposerConfig:
    hold_seconds: float = 1.0
    release_seconds: float = 0.3
    auto_space_seconds: float = 2.0
    min_frames: int = 3
    max_chars: int = 500


class TextComposer:
    def __init__(self, config: ComposerConfig | None = None):
        self.config = config or ComposerConfig()
        self.text = ""
        self._candidate: str | None = None
        self._candidate_since = 0.0
        self._candidate_frames = 0
        self._armed = True
        self._last_committed: str | None = None
        self._release_since: float | None = None
        self._last_hand_seen: float | None = None
        self._space_pending = False  # a letter was typed since the last space / edit

    # ------------------------------------------------------------------ per-frame update
    def update(self, letter: str | None, hand_visible: bool, now: float) -> str | None:
        """Feed one frame. `letter` is the confident prediction, or None if there is none.

        Returns the character typed on this frame (a letter or " "), or None.
        """
        cfg = self.config
        typed = None

        if hand_visible:
            self._last_hand_seen = now
        elif (cfg.auto_space_seconds > 0 and self._space_pending
              and self._last_hand_seen is not None
              and now - self._last_hand_seen >= cfg.auto_space_seconds):
            self._space_pending = False
            if not self.text.endswith(" ") and self._append(" "):
                typed = " "

        if letter != self._candidate:
            self._candidate = letter
            self._candidate_since = now
            self._candidate_frames = 0
        self._candidate_frames += 1

        if not self._armed:
            if letter != self._last_committed:
                if self._release_since is None:
                    self._release_since = now
                if now - self._release_since >= cfg.release_seconds:
                    self._armed = True
                    self._release_since = None
            else:
                self._release_since = None

        if (self._armed and letter is not None
                and now - self._candidate_since >= cfg.hold_seconds
                and self._candidate_frames >= cfg.min_frames):
            if self._append(letter):
                typed = letter
                self._space_pending = True
            self._last_committed = letter
            self._armed = False
            self._release_since = None

        return typed

    def hold_progress(self, now: float) -> float:
        """0..1 progress towards typing the current candidate letter (for the UI)."""
        if not self._armed or self._candidate is None or self.config.hold_seconds <= 0:
            return 0.0
        progress = min(1.0, (now - self._candidate_since) / self.config.hold_seconds)
        if self._candidate_frames < self.config.min_frames:
            progress = min(progress, 0.99)  # never show "full" on a frame that cannot type
        return progress

    @property
    def candidate(self) -> str | None:
        return self._candidate

    @property
    def waiting_for_release(self) -> bool:
        return not self._armed

    @property
    def last_committed(self) -> str | None:
        """The letter typed most recently by signing (None before the first one)."""
        return self._last_committed

    # ------------------------------------------------------------------ editing (keyboard)
    # Deleting a space (or emptying the text) cancels a pending auto-space, so BACKSPACE over an
    # auto-added space is not undone on the next frame while the hands are still out of view.
    # Deleting a letter keeps it pending, so the corrected word still gets its space.
    def add_space(self) -> bool:
        """Finish the current word. Returns True if a space was added."""
        self._space_pending = False
        if self.text and not self.text.endswith(" "):
            return self._append(" ")
        return False

    def backspace(self) -> None:
        removed = self.text[-1:]
        self.text = self.text[:-1]
        if removed == " " or not self.text:
            self._space_pending = False

    def clear(self) -> None:
        self._space_pending = False
        self.text = ""

    def last_word(self) -> str:
        """The most recently finished (or current) word."""
        words = self.text.split()
        return words[-1] if words else ""

    # ------------------------------------------------------------------ internals
    def _append(self, ch: str) -> bool:
        if len(self.text) >= self.config.max_chars:
            return False
        self.text += ch
        return True
