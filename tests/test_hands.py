import random
import unittest

import numpy as np

from signspeak.hands import (CONTEXT_MARGINS, Box, context_box, crop_box, landmarks_to_box,
                             select_sign_boxes, union_box)


class BoxTests(unittest.TestCase):
    def test_gap_is_zero_when_touching_or_overlapping(self):
        self.assertEqual(Box(0, 0, 10, 10).gap_to(Box(10, 0, 20, 10)), 0)
        self.assertEqual(Box(0, 0, 10, 10).gap_to(Box(5, 5, 20, 20)), 0)

    def test_gap_is_euclidean_between_corners(self):
        self.assertAlmostEqual(Box(0, 0, 10, 10).gap_to(Box(13, 14, 20, 20)), 5.0)

    def test_union(self):
        self.assertEqual(union_box([Box(0, 5, 10, 10), Box(3, 0, 20, 8)]), Box(0, 0, 20, 10))


class LandmarkBoxTests(unittest.TestCase):
    def test_clamps_out_of_frame_landmarks(self):
        box = landmarks_to_box([(-0.2, 0.5), (1.3, 0.6)], 640, 480)
        self.assertEqual((box.x1, box.x2), (0, 640))

    def test_never_empty(self):
        box = landmarks_to_box([(0.5, 0.5)] * 21, 640, 480)
        self.assertGreater(box.w, 0)
        self.assertGreater(box.h, 0)


class SelectSignBoxesTests(unittest.TestCase):
    def test_keeps_both_hands_of_a_two_handed_sign(self):
        left, right = Box(100, 100, 200, 220), Box(190, 110, 300, 230)
        self.assertEqual(set(select_sign_boxes([left, right])), {left, right})

    def test_drops_a_far_away_hand(self):
        signer, stranger = Box(100, 100, 200, 200), Box(500, 100, 560, 160)
        self.assertEqual(select_sign_boxes([signer, stranger]), [signer])

    def test_largest_hand_comes_first(self):
        small, big = Box(0, 0, 50, 50), Box(40, 0, 200, 200)
        self.assertEqual(select_sign_boxes([small, big])[0], big)

    def test_empty(self):
        self.assertEqual(select_sign_boxes([]), [])


class ContextBoxTests(unittest.TestCase):
    def test_margins_are_applied_per_side(self):
        b = context_box(Box(200, 100, 300, 200), 640, 480, margins=(0.5, 0.1, 0.2, 0.3))
        self.assertEqual(b, Box(150, 90, 320, 230))

    def test_default_margins_keep_natural_shape(self):
        tall = context_box(Box(300, 100, 340, 300), 640, 480)  # a tall sign stays tall
        self.assertGreater(tall.h, tall.w)
        left, top, right, bottom = CONTEXT_MARGINS
        self.assertEqual(tall.x1, 300 - round(left * 40))

    def test_clipped_to_frame(self):
        b = context_box(Box(0, 0, 80, 80), 640, 480)
        self.assertEqual((b.x1, b.y1), (0, 0))
        b = context_box(Box(560, 400, 640, 480), 640, 480)
        self.assertEqual((b.x2, b.y2), (640, 480))

    def test_random_boxes_stay_inside_frame_and_contain_the_hands(self):
        rng = random.Random(0)
        for _ in range(2000):
            fw, fh = rng.randint(50, 1280), rng.randint(50, 960)
            x1, y1 = rng.randint(0, fw - 1), rng.randint(0, fh - 1)
            box = Box(x1, y1, rng.randint(x1 + 1, fw), rng.randint(y1 + 1, fh))
            c = context_box(box, fw, fh)
            self.assertTrue(0 <= c.x1 < c.x2 <= fw and 0 <= c.y1 < c.y2 <= fh)
            self.assertTrue(c.x1 <= box.x1 and c.y1 <= box.y1 and c.x2 >= box.x2 and c.y2 >= box.y2)


class CropBoxTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.arange(48 * 64 * 3, dtype=np.uint32).reshape(48, 64, 3).astype(np.uint8)

    def test_inside_frame_matches_slice(self):
        out = crop_box(self.frame, Box(10, 5, 30, 25))
        np.testing.assert_array_equal(out, self.frame[5:25, 10:30])

    def test_returns_a_copy(self):
        out = crop_box(self.frame, Box(10, 5, 30, 25))
        self.assertFalse(np.shares_memory(out, self.frame))  # drawing on the frame can't leak in

    def test_out_of_frame_parts_are_black(self):
        out = crop_box(self.frame, Box(-10, -10, 20, 20))
        self.assertEqual(out.shape, (30, 30, 3))
        self.assertTrue((out[:10] == 0).all())
        np.testing.assert_array_equal(out[10:, 10:], self.frame[0:20, 0:20])

    def test_fully_outside_is_black_not_wrapped(self):
        out = crop_box(self.frame, Box(-50, -50, -20, -20))
        self.assertEqual(out.shape, (30, 30, 3))
        self.assertFalse(out.any())


if __name__ == "__main__":
    unittest.main()
