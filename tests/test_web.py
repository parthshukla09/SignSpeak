"""Engine, practice, model registry and web server tests (fake tracker/classifier: no camera, no TF)."""
import contextlib
import json
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from signspeak.engine import EngineConfig, RecognitionEngine
from signspeak.hands import Detection, landmarks_to_box
from signspeak.practice import PracticeSession
from signspeak.registry import ModelCache, default_model_id, list_models

LETTERS = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


class _Pt:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Hand:
    def __init__(self, pts):
        self.landmark = [_Pt(x, y) for x, y in pts]


class FakeTracker:
    """A bright frame contains one hand in the middle; a dark frame contains none."""

    def __init__(self):
        self.closed = False

    def detect(self, frame):
        if frame.mean() < 50:
            return []
        h, w = frame.shape[:2]
        pts = [(0.40 + 0.01 * i, 0.35 + 0.015 * i) for i in range(21)]
        return [Detection(_Hand(pts), landmarks_to_box(pts, w, h))]

    def close(self):
        self.closed = True


class FakeClassifier:
    class_names = LETTERS
    input_size = 224

    def __init__(self, letter="A", confidence=0.9):
        self.letter, self.confidence = letter, confidence

    def predict(self, crop, flip_average=None):
        p = np.full(26, (1 - self.confidence) / 25, np.float32)
        p[LETTERS.index(self.letter)] = self.confidence
        return p

    def warm_up(self):
        pass


BRIGHT = np.full((480, 640, 3), 200, np.uint8)
DARK = np.zeros((480, 640, 3), np.uint8)


def jpeg(frame):
    return cv2.imencode(".jpg", frame)[1].tobytes()


class EngineTests(unittest.TestCase):
    def test_types_after_hold_and_reports_geometry(self):
        e = RecognitionEngine(FakeClassifier("D"), FakeTracker(), EngineConfig(hold=0.0))
        results = [e.process(BRIGHT, now=t * 0.1) for t in range(3)]
        self.assertEqual([r.typed for r in results], [None, None, "D"])
        r = results[-1]
        self.assertEqual((r.best, r.letter, r.text), ("D", "D", "D"))
        self.assertEqual(len(r.hands), 1)
        self.assertEqual(len(r.hands[0]), 21)
        x1, y1, x2, y2 = r.box
        self.assertTrue(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1)
        self.assertEqual(r.top[0][0], "D")
        j = r.to_json()
        json.dumps(j)  # serialisable
        self.assertNotIn("crop", j)

    def test_no_hands_and_low_confidence(self):
        e = RecognitionEngine(FakeClassifier("A", confidence=0.4), FakeTracker())
        r = e.process(BRIGHT, now=0.0)
        self.assertEqual((r.best, r.letter), ("A", None))  # below the 0.6 threshold
        r = e.process(DARK, now=0.1)
        self.assertEqual((r.hands, r.box, r.best), ([], None, None))

    def test_compose_false_leaves_text_alone(self):
        e = RecognitionEngine(FakeClassifier("B"), FakeTracker(), EngineConfig(hold=0.0))
        for t in range(5):
            e.process(BRIGHT, now=t * 0.1, compose=False)
        self.assertEqual(e.composer.text, "")

    def test_commands_and_config_clamping(self):
        e = RecognitionEngine(FakeClassifier("H"), FakeTracker(), EngineConfig(hold=0.0))
        for t in range(3):
            e.process(BRIGHT, now=t * 0.1)
        self.assertEqual(e.command("space"), "H")
        e.command("backspace")
        e.command("clear")
        self.assertEqual(e.composer.text, "")
        with self.assertRaises(ValueError):
            e.command("explode")
        e.configure({"threshold": 7, "smoothing": 0, "hold": float("nan"), "bogus": 1, "flip_average": "yes"})
        self.assertEqual((e.config.threshold, e.config.smoothing, e.config.hold, e.config.flip_average),
                         (1.0, 1, 0.0, True))


class PracticeTests(unittest.TestCase):
    def test_correct_after_hold_then_next_target(self):
        p = PracticeSession(list("ABC"), order="sequential", hold=0.5)
        self.assertEqual(p.target, "A")
        self.assertIsNone(p.update("A", 0.0))
        self.assertIsNone(p.update("B", 0.2))  # wrong letter resets the hold
        self.assertIsNone(p.update("A", 0.3))
        self.assertEqual(p.update("A", 0.85), "correct")
        self.assertEqual(p.target, "B")
        self.assertEqual(p.stats["A"].correct, 1)
        self.assertEqual(p.streak, 1)

    def test_skip_and_random_never_repeats_immediately(self):
        p = PracticeSession(list("AB"), order="random", seed=1)
        targets = []
        for _ in range(10):
            targets.append(p.target)
            p.skip()
        self.assertTrue(all(a != b for a, b in zip(targets, targets[1:])))
        self.assertEqual(p.streak, 0)
        with self.assertRaises(ValueError):
            PracticeSession([])


class RegistryTests(unittest.TestCase):
    def test_lists_and_picks_newest_personalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "models" / "SignSpeak_finetuned.keras").write_bytes(b"x")
            (root / "class_names.json").write_text(json.dumps(LETTERS))
            for name, metrics in [("smoke", {"test_accuracy": 0.24}),
                                  ("finetune-a", {"after": {"your_heldout_take": {"accuracy": 0.9}}, "recordings": {"A": 1}}),
                                  ("broken", None)]:
                d = root / "runs" / name
                d.mkdir(parents=True)
                (d / "SignSpeak.keras").write_bytes(b"x")
                if metrics is not None:
                    (d / "class_names.json").write_text(json.dumps(LETTERS))
                    (d / "metrics.json").write_text(json.dumps(metrics))
            models = list_models(root)
            kinds = {m.id: m.kind for m in models}
            self.assertEqual(kinds, {"original": "original", "run:smoke": "trained", "run:finetune-a": "personalised"})
            self.assertEqual(models[0].id, "original")
            self.assertEqual(default_model_id(models), "run:finetune-a")
            self.assertIn("90.0%", next(m for m in models if m.id == "run:finetune-a").summary)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class WS:
    """Small JSON helper around a websockets sync connection."""

    def __init__(self, conn):
        self.conn = conn

    def send_json(self, obj):
        self.conn.send(json.dumps(obj))

    def send_bytes(self, data):
        self.conn.send(data)

    def receive_json(self):
        return json.loads(self.conn.recv(timeout=10))


class ServerTests(unittest.TestCase):
    """Runs the real server (uvicorn) on a free local port with a fake tracker and classifier."""

    def setUp(self):
        import uvicorn

        from signspeak.server import create_app

        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "models").mkdir()
        (root / "models" / "SignSpeak_finetuned.keras").write_bytes(b"x")
        (root / "class_names.json").write_text(json.dumps(LETTERS))
        (root / "web").mkdir()
        (root / "web" / "index.html").write_text("<!doctype html><title>t</title>")
        self.root = root
        self.trackers = []

        def tracker_factory():
            t = FakeTracker()
            self.trackers.append(t)
            return t

        cache = ModelCache(loader=lambda info: FakeClassifier("D"))
        self.port = free_port()
        app = create_app(root, port=self.port, model_cache=cache, tracker_factory=tracker_factory)
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)
        self.thread.start()
        deadline = time.time() + 10
        while not self.server.started and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.server.started, "server did not start")

    def tearDown(self):
        self.server.should_exit = True
        self.thread.join(timeout=10)
        self.tmp.cleanup()

    def http(self, method, path, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", method=method,
                                     headers=headers or {}, data=b"{}" if method == "POST" else None)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            body = e.read()
            return e.code, json.loads(body) if body.startswith(b"{") else body

    @contextlib.contextmanager
    def ws(self, **kwargs):
        from websockets.sync.client import connect

        with connect(f"ws://127.0.0.1:{self.port}/ws", open_timeout=10, **kwargs) as conn:
            yield WS(conn)

    def hello(self, ws, **config):
        ws.send_json({"type": "hello", "model": None, "config": config})
        self.assertEqual(ws.receive_json()["type"], "loading")
        ready = ws.receive_json()
        self.assertEqual(ready["type"], "ready")
        return ready

    def test_index_state_and_static(self):
        status, _ = self.http("GET", "/api/state")
        self.assertEqual(status, 200)
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/", timeout=10) as r:
            self.assertEqual(r.status, 200)
        _, s = self.http("GET", "/api/state")
        self.assertEqual(s["classes"], LETTERS)
        self.assertEqual(s["default_model"], "original")
        self.assertEqual(s["recordings"]["A"], {"takes": 0, "images": 0})
        self.assertFalse(s["dataset_available"])
        self.assertFalse(s["finetune_available"])
        (self.root / "models" / "replay_features.npz").write_bytes(b"x")  # the bundled features
        _, s = self.http("GET", "/api/state")
        self.assertTrue(s["finetune_available"])
        self.assertFalse(s["dataset_available"])
        self.assertEqual(s["finetune"]["state"], "idle")
        self.assertEqual(s["config_limits"]["smoothing"], [1, 30])

    def test_rejects_foreign_origin_and_host(self):
        status, _ = self.http("POST", "/api/recordings/A/delete-last", {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.http("GET", "/api/state", {"Host": "attacker.example"})
        self.assertEqual(status, 403)
        status, _ = self.http("POST", "/api/recordings/A/delete-last",
                              {"Origin": f"http://127.0.0.1:{self.port}"})  # the app's own page
        self.assertEqual(status, 200)
        with self.assertRaises(Exception):
            with self.ws(origin="https://evil.example") as ws:
                ws.receive_json()

    def test_translate_flow(self):
        with self.ws(origin=f"http://127.0.0.1:{self.port}") as ws:
            ready = self.hello(ws, hold=0.0)
            self.assertEqual(ready["model"], "original")
            self.assertEqual(ready["config"]["hold"], 0.0)
            results = []
            for _ in range(3):
                ws.send_bytes(jpeg(BRIGHT))
                results.append(ws.receive_json())
            self.assertEqual(results[-1]["typed"], "D")
            self.assertEqual(results[-1]["text"], "D")
            self.assertEqual(results[-1]["mode"], "translate")
            self.assertEqual(len(results[-1]["hands"][0]), 21)
            ws.send_json({"type": "command", "action": "space"})
            self.assertEqual(ws.receive_json(), {"type": "text", "text": "D ", "word": "D"})
            ws.send_bytes(b"not a jpeg")
            self.assertEqual(ws.receive_json()["type"], "error")
            self.assertTrue(ws.receive_json()["skipped"])
            ws.send_json({"type": "config", "config": {"threshold": 0.95}})
            self.assertEqual(ws.receive_json()["config"]["threshold"], 0.95)
            ws.send_bytes(jpeg(BRIGHT))
            self.assertIsNone(ws.receive_json()["letter"])  # 0.9 < 0.95 now
            ws.send_json({"type": "nonsense"})
            self.assertEqual(ws.receive_json()["type"], "error")
        deadline = time.time() + 5
        while not self.trackers[0].closed and time.time() < deadline:
            time.sleep(0.02)
        self.assertTrue(self.trackers[0].closed)

    def test_frames_before_ready_get_an_empty_result(self):
        with self.ws() as ws:
            ws.send_bytes(jpeg(BRIGHT))
            self.assertTrue(ws.receive_json()["skipped"])

    def test_learn_flow(self):
        with self.ws() as ws:
            self.hello(ws)
            ws.send_json({"type": "learn_start", "letters": ["D"], "order": "sequential", "hold": 0.2})
            learn = ws.receive_json()["learn"]
            self.assertEqual(learn["target"], "D")
            event, r = None, None
            deadline = time.time() + 5
            while event != "correct" and time.time() < deadline:
                ws.send_bytes(jpeg(BRIGHT))
                r = ws.receive_json()
                self.assertEqual(r["mode"], "learn")
                self.assertEqual(r["text"], "")  # no typing while learning
                event = r["learn"]["event"]
            self.assertEqual(event, "correct")
            self.assertEqual(r["learn"]["stats"]["D"]["correct"], 1)
            ws.send_json({"type": "learn_start", "letters": ["?"]})
            self.assertEqual(ws.receive_json()["type"], "error")

    def test_record_flow_and_delete_last(self):
        with self.ws() as ws:
            self.hello(ws)
            ws.send_json({"type": "record_start", "letter": "K", "per_take": 5})
            self.assertEqual(ws.receive_json()["record"]["needed"], 5)
            done, r = None, None
            deadline = time.time() + 10
            while not done and time.time() < deadline:
                ws.send_bytes(jpeg(BRIGHT))
                r = ws.receive_json()
                done = r["record"] and r["record"]["done"]
            self.assertEqual(r["recordings"]["K"], {"takes": 1, "images": 5})
            # a second take, cancelled half way, leaves nothing behind
            ws.send_json({"type": "record_start", "letter": "K", "per_take": 50})
            ws.receive_json()
            ws.send_bytes(jpeg(BRIGHT))
            ws.receive_json()
            ws.send_json({"type": "record_cancel"})
            self.assertEqual(ws.receive_json()["recordings"]["K"], {"takes": 1, "images": 5})
        status, body = self.http("POST", "/api/recordings/K/delete-last")
        self.assertEqual((status, body["removed"], body["recordings"]["K"]["takes"]), (200, 5, 0))
        status, _ = self.http("POST", "/api/recordings/nope/delete-last")
        self.assertEqual(status, 404)

    def test_finetune_needs_recordings_and_dataset(self):
        status, body = self.http("POST", "/api/finetune")
        self.assertEqual(status, 400)
        self.assertIn("Record", body["detail"])
        d = self.root / "data" / "user" / "A"
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "take01_001.jpg"), BRIGHT)
        status, body = self.http("POST", "/api/finetune")
        self.assertEqual(status, 400)
        self.assertIn("dataset", body["detail"])


if __name__ == "__main__":
    unittest.main()
