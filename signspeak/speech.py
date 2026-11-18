"""Offline text-to-speech on a background thread, so speaking never stalls the camera loop.

Windows uses the built-in SAPI voice directly (fast and reliable from a worker thread).
Other platforms use pyttsx3 with a fresh engine per utterance; reusing one pyttsx3 engine
across utterances from a worker thread can hang on its second runAndWait() call.
"""
from __future__ import annotations

import queue
import sys
import threading


class Speaker:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.available = False
        self.error: str | None = None
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None
        if enabled:
            self._thread = threading.Thread(target=self._run, name="speaker", daemon=True)
            self._thread.start()
            if not self._ready.wait(timeout=5.0):
                self.error = "voice engine still starting; speech turns on when it is ready"

    @property
    def starting(self) -> bool:
        """True while the voice engine is still initialising in the background."""
        return self._thread is not None and self._thread.is_alive() and not self._ready.is_set()

    def say(self, text: str) -> None:
        """Queue text to be spoken. Text queued while the engine is still starting is kept."""
        text = text.strip()
        if text and self._thread is not None and self._thread.is_alive():
            self._queue.put(text)

    def close(self, timeout: float = 2.0) -> None:
        if self._thread is not None and self._thread.is_alive():
            # drop anything still waiting so shutdown is quick
            try:
                while True:
                    self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put(None)
            self._thread.join(timeout=timeout)

    # ------------------------------------------------------------------ worker thread
    def _run(self) -> None:
        try:
            speak = self._make_backend()
            self.error = None
            self.available = True
        except Exception as exc:  # no voice installed, missing package, ...
            self.error = f"{type(exc).__name__}: {exc}"
            if sys.platform.startswith("linux"):
                self.error += " - install a speech engine: sudo apt install espeak-ng"
            elif sys.platform == "darwin":
                self.error += " - check that pyobjc is installed: pip install pyobjc"
            return
        finally:
            self._ready.set()

        for text in iter(self._queue.get, None):
            try:
                speak(text)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"

    def _make_backend(self):
        if sys.platform == "win32":
            import pythoncom
            import win32com.client

            pythoncom.CoInitialize()
            return win32com.client.Dispatch("SAPI.SpVoice").Speak

        import pyttsx3

        pyttsx3.init().stop()  # fail fast here if no TTS driver is available

        def speak(text: str) -> None:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
            engine.stop()

        return speak
