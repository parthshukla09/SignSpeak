import contextlib
import io
import unittest

import numpy as np

import webcam
from signspeak.model import PredictionSmoother


class ArgumentTests(unittest.TestCase):
    def test_defaults(self):
        a = webcam.parse_args([])
        self.assertEqual((a.hold, a.release, a.auto_space, a.smoothing), (1.0, 0.3, 2.0, 7))

    def test_invalid_values_are_rejected_up_front(self):
        for argv in (["--smoothing", "0"], ["--threshold", "1.5"], ["--hold", "-1"],
                     ["--auto-space", "-2"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    webcam.parse_args(argv)

    def test_smoother_window_one_means_no_smoothing(self):
        s = PredictionSmoother(window=0)  # clamped to 1 instead of crashing
        p = np.eye(3)[2]
        self.assertEqual(s.add(p), (2, 1.0))

    def test_smoother_top_k(self):
        s = PredictionSmoother(window=2)
        self.assertEqual(s.top(), [])
        s.add(np.array([0.1, 0.6, 0.3]))
        s.add(np.array([0.1, 0.4, 0.5]))
        self.assertEqual([i for i, _ in s.top(3)], [1, 2, 0])
        s.reset()
        self.assertEqual(s.top(), [])

    def test_mirror_flags(self):
        a = webcam.parse_args([])
        self.assertFalse(a.no_mirror or a.no_flip_average or a.no_preview)
        a = webcam.parse_args(["--no-mirror", "--no-flip-average", "--no-preview"])
        self.assertTrue(a.no_mirror and a.no_flip_average and a.no_preview)


class DrawingTests(unittest.TestCase):
    def test_draw_ui_handles_any_frame_size_and_long_text(self):
        for w, h in [(1280, 720), (640, 480), (320, 240), (160, 120), (50, 30)]:
            for text in ("", "HELLO WORLD " * 40):
                with self.subTest(size=(w, h), chars=len(text)):
                    frame = np.full((h, w, 3), 200, np.uint8)
                    webcam.draw_ui(frame, status="Uncertain (W 45%)", status_color=webcam.AMBER,
                                   progress=0.5, waiting_release=True, text=text,
                                   fps=14.2, speech="on",
                                   preview=np.zeros((224, 224, 3), np.uint8),
                                   top3=[("W", 0.45), ("V", 0.3), ("U", 0.1)])

    def test_fit_text_trims_from_the_requested_side(self):
        long = "ABCDEFGHIJ" * 20
        end = webcam.fit_text(long, 200, 0.9, 2, keep="end")
        start = webcam.fit_text(long, 200, 0.9, 2, keep="start")
        self.assertTrue(end.startswith("...") and long.endswith(end[3:]))
        self.assertTrue(start.endswith("...") and long.startswith(start[:-3]))


if __name__ == "__main__":
    unittest.main()
