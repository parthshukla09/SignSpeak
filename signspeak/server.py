"""SignSpeak web server (FastAPI). Start it with `python app.py`. The protocol is in web/PROTOCOL.md.

The browser owns the camera and sends JPEG frames over a WebSocket; each connection gets its own
RecognitionEngine (hand tracker, smoothing, text) while the TensorFlow models are shared.

Security: the server is meant to listen on 127.0.0.1 only. It also rejects requests whose Host or
Origin is not this local server, so a web page from another site open in the same browser cannot
use it (e.g. to delete recordings or read camera results) - including via DNS rebinding.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from signspeak import userdata
from signspeak.engine import EngineConfig, RecognitionEngine
from signspeak.model import PROJECT_ROOT
from signspeak.practice import PracticeSession
from signspeak.registry import ModelCache, ModelInfo, default_model_id, list_models

MAX_FRAME_BYTES = 3_000_000
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]", "::1"}
LOG_NOISE = ("W0000", "I0000", "oneDNN", "WARNING:tensorflow", "WARNING: All log messages",
             "INFO: Created TensorFlow")


# ---------------------------------------------------------------------- fine-tuning job
class FinetuneJob:
    """Runs finetune.py in a subprocess (keeps TensorFlow training out of the server process)."""

    def __init__(self, root: Path, python: str = sys.executable):
        self.root, self.python = root, python
        self.state = "idle"
        self.run: str | None = None
        self.log: deque[str] = deque(maxlen=200)
        self.started: float | None = None
        self.finished: float | None = None
        self.missing: list[str] = []
        self.result: dict | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        with self._lock:
            return {"state": self.state, "run": self.run,
                    "model_id": f"run:{self.run}" if self.state == "done" and self.run else None,
                    "log": list(self.log), "started": self.started, "finished": self.finished,
                    "missing_letters": list(self.missing), "result": self.result}

    def start(self, missing_letters: list[str]) -> None:
        with self._lock:
            if self.state == "running":
                raise RuntimeError("a fine-tuning job is already running")
            self.state, self.result, self.finished = "running", None, None
            self.run = "web-" + time.strftime("%Y%m%d-%H%M%S")
            self.missing = list(missing_letters)
            self.log.clear()
            self.started = time.time()
            env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="2", PYTHONUNBUFFERED="1")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            self._proc = subprocess.Popen(
                [self.python, "-u", str(self.root / "finetune.py"), "--name", self.run],
                cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", creationflags=flags)
        threading.Thread(target=self._pump, name="finetune-log", daemon=True).start()

    def _pump(self) -> None:
        proc = self._proc
        assert proc is not None and proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line and not any(n in line for n in LOG_NOISE):
                with self._lock:
                    self.log.append(line)
        code = proc.wait()
        metrics = self.root / "runs" / (self.run or "") / "metrics.json"
        with self._lock:
            self.finished = time.time()
            if code == 0 and metrics.is_file():
                m = json.loads(metrics.read_text(encoding="utf-8"))
                self.result = {"before": m.get("before"), "after": m.get("after")}
                self.state = "done"
            else:
                self.log.append(f"fine-tuning failed (exit code {code})")
                self.state = "failed"

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()


# ---------------------------------------------------------------------- one browser connection
class Session:
    def __init__(self, root: Path, tracker):
        self.root = root
        self.tracker = tracker
        self.engine: RecognitionEngine | None = None
        self.model: ModelInfo | None = None
        self.config = EngineConfig()
        self.mode = "translate"
        self.practice: PracticeSession | None = None
        self.recorder: userdata.TakeRecorder | None = None
        self.last_record_state: dict | None = None

    @property
    def classes(self) -> list[str]:
        return list(self.engine.classifier.class_names) if self.engine else []

    def use_classifier(self, info: ModelInfo, classifier) -> None:
        if self.engine is None:
            self.engine = RecognitionEngine(classifier, self.tracker, self.config)
        else:
            self.engine.set_classifier(classifier)
        self.model = info

    def ready_message(self) -> dict:
        return {"type": "ready", "model": self.model.id, "model_label": self.model.label,
                "config": self.config_json(), "text": self.engine.composer.text, "classes": self.classes}

    def config_json(self) -> dict:
        c = self.engine.config if self.engine else self.config
        return {k: getattr(c, k) for k in c.__dataclass_fields__}

    def recordings(self) -> dict:
        return userdata.summary(self.root / "data" / "user", self.classes)

    def record_state(self) -> dict | None:
        r = self.recorder
        if r is None:
            return self.last_record_state
        return {"letter": r.letter, "active": not r.done, "saved": len(r.files), "needed": r.per_take,
                "take": r.take or userdata.next_take_number(r.dir), "done": r.done}

    def cancel_recording(self) -> bool:
        if self.recorder is not None and not self.recorder.done:
            self.recorder.cancel()
            self.recorder = None
            return True
        self.recorder = None
        return False

    def process(self, frame: np.ndarray) -> dict:
        t0 = time.perf_counter()
        now = time.monotonic()
        res = self.engine.process(frame, now, compose=(self.mode == "translate"))
        out = {"type": "result", "mode": self.mode, **res.to_json(),
               "word": self.engine.composer.last_word() if res.typed == " " else None,
               "learn": None, "record": None}
        if self.mode == "learn" and self.practice is not None:
            event = self.practice.update(res.letter, now)
            out["learn"] = {**self.practice.to_json(now), "event": event}
        if self.mode == "record" and self.recorder is not None:
            if res.crop is not None:
                self.recorder.offer(res.crop, now)
            out["record"] = self.record_state()
            if self.recorder.done:
                out["recordings"] = self.recordings()
                self.last_record_state = out["record"]
                self.recorder = None
        elif self.mode == "record":
            out["record"] = self.last_record_state
        out["ms"] = round((time.perf_counter() - t0) * 1000)
        return out

    def close(self) -> None:
        self.cancel_recording()
        if self.tracker is not None:
            self.tracker.close()


def _empty_result(mode: str) -> dict:
    return {"type": "result", "mode": mode, "hands": [], "box": None, "best": None, "confidence": 0.0,
            "letter": None, "top": [], "typed": None, "word": None, "text": "", "progress": 0.0,
            "waiting_release": False, "ms": 0, "learn": None, "record": None, "skipped": True}


# ---------------------------------------------------------------------- the app
def create_app(root: Path = PROJECT_ROOT, port: int = 8000, model_cache: ModelCache | None = None,
               tracker_factory=None, finetune_job: FinetuneJob | None = None,
               extra_hosts: tuple[str, ...] = (), preload: bool = True) -> FastAPI:
    """`extra_hosts` adds accepted Host names. `preload` starts loading the default model at
    start-up."""
    root = Path(root)
    web = root / "web"
    cache = model_cache or ModelCache()
    job = finetune_job or FinetuneJob(root)
    if tracker_factory is None:
        def tracker_factory():
            from signspeak.hands import HandTracker
            return HandTracker(static_image_mode=False, max_num_hands=2)

    def preload_default_model() -> None:
        """Load the default model in the background at start-up (importing TensorFlow and
        loading the model takes ~10 s), so it is usually ready by the time the browser connects."""
        t0 = time.perf_counter()
        try:
            ms = list_models(root)
            info = next((m for m in ms if m.id == default_model_id(ms)), None)
            if info is not None:
                cache.get(info)
                print(f"Model ready: {info.label} ({time.perf_counter() - t0:.0f} s)", flush=True)
        except Exception as exc:  # the browser will get the error when it asks for the model
            print(f"Could not preload the model: {exc}", flush=True)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if preload:
            threading.Thread(target=preload_default_model, name="preload", daemon=True).start()
        yield
        job.stop()  # don't leave a training subprocess running after the server stops

    app = FastAPI(title="SignSpeak", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    allowed_hosts = {f"{h}:{port}" for h in LOCAL_HOSTS} | LOCAL_HOSTS | set(extra_hosts)

    def host_ok(host: str | None) -> bool:
        return (host or "").lower() in allowed_hosts

    def origin_ok(origin: str | None) -> bool:
        if not origin:  # same-origin GETs and non-browser clients send no Origin
            return True
        parts = urlsplit(origin)
        return parts.scheme in ("http", "https") and host_ok(parts.netloc)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not host_ok(request.headers.get("host")) or not origin_ok(request.headers.get("origin")):
            return JSONResponse({"detail": "forbidden"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def classes() -> list[str]:
        from signspeak.model import load_class_names
        return load_class_names(root / "class_names.json")

    def models() -> list[ModelInfo]:
        return list_models(root)

    def finetune_data_available() -> bool:
        """finetune.py needs either the dataset or the bundled dataset features."""
        return ((root / "data" / "RealSign" / "Training").is_dir()
                or (root / "models" / "replay_features.npz").is_file())

    # ------------------------------------------------------------------ HTTP
    @app.get("/")
    def index():
        return FileResponse(web / "index.html")

    if web.is_dir():
        app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/api/state")
    def state():
        ms = models()
        d = EngineConfig()
        return {
            "classes": classes(),
            "models": [m.to_json() for m in ms],
            "default_model": default_model_id(ms),
            "config_defaults": {k: getattr(d, k) for k in d.__dataclass_fields__},
            "config_limits": {k: list(v) for k, v in EngineConfig.LIMITS.items()},
            "recordings": userdata.summary(root / "data" / "user", classes()),
            "dataset_available": (root / "data" / "RealSign" / "Training").is_dir(),
            "finetune_available": finetune_data_available(),
            "finetune": job.status(),
        }

    @app.post("/api/recordings/{letter}/delete-last")
    def delete_last(letter: str):
        names = classes()
        if letter not in names:
            raise HTTPException(404, f"unknown letter {letter!r}")
        removed = userdata.delete_last_take(root / "data" / "user", letter)
        return {"removed": removed, "recordings": userdata.summary(root / "data" / "user", names)}

    @app.post("/api/finetune", status_code=202)
    def start_finetune():
        names = classes()
        rec = userdata.summary(root / "data" / "user", names)
        if not any(v["takes"] for v in rec.values()):
            raise HTTPException(400, "Nothing recorded yet. Record your signs first.")
        if not finetune_data_available():
            raise HTTPException(400, "Fine-tuning needs models/replay_features.npz (included in the "
                                     "repository) or the RealSign dataset in data/RealSign, and neither "
                                     "was found. See 'Dataset' in README.md.")
        try:
            job.start([c for c in names if rec[c]["takes"] == 0])
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"finetune": job.status()}

    @app.get("/api/finetune")
    def finetune_status():
        return {"finetune": job.status()}

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        if not host_ok(ws.headers.get("host")) or not origin_ok(ws.headers.get("origin")):
            await ws.close(code=1008)
            return
        await ws.accept()
        session = Session(root, None)

        async def send_error(message: str):
            await ws.send_json({"type": "error", "message": message})

        async def load_model(model_id: str | None) -> None:
            ms = models()
            model_id = model_id or default_model_id(ms)
            info = next((m for m in ms if m.id == model_id), None)
            if info is None:
                await send_error(f"Unknown model {model_id!r}; using the default.")
                info = next((m for m in ms if m.id == default_model_id(ms)), None)
                if info is None:
                    await send_error("No model available.")
                    return
            await ws.send_json({"type": "loading", "model": info.id})
            try:
                if session.tracker is None:
                    session.tracker = await asyncio.to_thread(tracker_factory)
                clf = await asyncio.to_thread(cache.get, info)
                session.use_classifier(info, clf)
            except Exception as exc:  # corrupt model file, out of memory, ...
                await send_error(f"Could not load model {info.label}: {exc}")
                return
            await ws.send_json(session.ready_message())

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data is not None:
                    if session.engine is None:
                        await ws.send_json(_empty_result(session.mode))
                        continue
                    frame = None
                    if len(data) <= MAX_FRAME_BYTES:
                        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        await send_error("Could not decode the camera frame.")
                        await ws.send_json(_empty_result(session.mode))
                        continue
                    await ws.send_json(await asyncio.to_thread(session.process, frame))
                    continue
                try:
                    m = json.loads(msg.get("text") or "")
                    if not isinstance(m, dict):
                        raise ValueError
                except ValueError:
                    await send_error("Messages must be JSON objects.")
                    continue
                await handle(m, session, ws, load_model, send_error)
        except WebSocketDisconnect:
            pass
        finally:
            await asyncio.to_thread(session.close)

    async def handle(m: dict, session: Session, ws: WebSocket, load_model, send_error) -> None:
        kind = m.get("type")
        ws_send = ws.send_json
        if kind == "hello":
            if isinstance(m.get("config"), dict):
                session.config = session.config.updated(m["config"])
            await load_model(m.get("model") if isinstance(m.get("model"), str) else None)
            return
        if kind == "model":
            await load_model(m.get("model") if isinstance(m.get("model"), str) else None)
            return
        if kind == "config":
            if isinstance(m.get("config"), dict):
                session.config = session.config.updated(m["config"])
                if session.engine is not None:
                    session.engine.configure(m["config"])
            await ws_send({"type": "config", "config": session.config_json()})
            return
        if session.engine is None:
            await send_error("Not ready yet: the model is still loading.")
            return
        if kind == "mode":
            mode = m.get("mode")
            if mode not in ("translate", "learn", "record"):
                await send_error(f"Unknown mode {mode!r}.")
                return
            if session.mode == "record" and mode != "record":
                session.cancel_recording()
            session.mode = mode
            return
        if kind == "command":
            try:
                word = session.engine.command(str(m.get("action")))
            except ValueError as exc:
                await send_error(str(exc))
                return
            await ws_send({"type": "text", "text": session.engine.composer.text, "word": word})
            return
        if kind == "learn_start":
            names = session.classes
            letters = m.get("letters") or names
            if not isinstance(letters, list) or not all(isinstance(c, str) and c in names for c in letters):
                await send_error("learn_start: letters must be a list of known letters.")
                return
            order = m.get("order", "random")
            hold = m.get("hold", 0.8)
            if order not in ("random", "sequential"):
                order = "random"
            hold = float(hold) if isinstance(hold, (int, float)) and not isinstance(hold, bool) else 0.8
            hold = min(max(hold, 0.2), 3.0) if hold == hold else 0.8
            session.practice = PracticeSession(list(dict.fromkeys(letters)), order=order, hold=hold)
            session.mode = "learn"
            await ws_send({"type": "learn", "learn": {**session.practice.to_json(time.monotonic()), "event": None}})
            return
        if kind == "learn_skip":
            if session.practice is None:
                await send_error("No practice session; send learn_start first.")
                return
            session.practice.skip()
            await ws_send({"type": "learn", "learn": {**session.practice.to_json(time.monotonic()), "event": None}})
            return
        if kind == "record_start":
            letter = m.get("letter")
            if letter not in session.classes:
                await send_error(f"record_start: unknown letter {letter!r}.")
                return
            per_take = m.get("per_take", 20)
            per_take = int(per_take) if isinstance(per_take, (int, float)) and not isinstance(per_take, bool) and per_take == per_take else 20
            session.cancel_recording()
            session.last_record_state = None
            session.recorder = userdata.TakeRecorder(root / "data" / "user", letter,
                                                     per_take=min(max(per_take, 5), 60))
            session.mode = "record"
            await ws_send({"type": "record", "record": session.record_state(),
                                    "recordings": session.recordings()})
            return
        if kind == "record_cancel":
            cancelled = session.cancel_recording()
            session.last_record_state = None
            await ws_send({"type": "record", "record": {"active": False, "cancelled": cancelled},
                                    "recordings": session.recordings()})
            return
        await send_error(f"Unknown message type {kind!r}.")

    return app
