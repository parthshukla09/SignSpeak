import unittest

from signspeak.composer import ComposerConfig, TextComposer

FPS = 10
DT = 1.0 / FPS


class Feeder:
    """Feeds frames to a composer at a fixed frame rate."""

    def __init__(self, composer: TextComposer):
        self.c = composer
        self.t = 0.0

    def frames(self, letter, seconds, hand=True):
        typed = []
        for _ in range(round(seconds * FPS)):
            self.t += DT
            out = self.c.update(letter, hand_visible=hand, now=self.t)
            if out:
                typed.append(out)
        return typed


def make(**kw):
    cfg = ComposerConfig(**{"hold_seconds": 1.0, "release_seconds": 0.3, "auto_space_seconds": 2.0, **kw})
    c = TextComposer(cfg)
    return c, Feeder(c)


class HoldToConfirmTests(unittest.TestCase):
    def test_letter_typed_only_after_hold(self):
        c, f = make()
        f.frames("A", 0.8)
        self.assertEqual(c.text, "")
        f.frames("A", 0.4)
        self.assertEqual(c.text, "A")

    def test_brief_misprediction_is_not_typed(self):
        c, f = make()
        f.frames("A", 1.2)
        f.frames("B", 0.3)  # transition flicker
        f.frames("C", 1.2)
        self.assertEqual(c.text, "AC")

    def test_flicker_resets_the_hold_timer(self):
        c, f = make()
        f.frames("A", 0.6)
        f.frames(None, 0.1)
        f.frames("A", 0.6)
        self.assertEqual(c.text, "")

    def test_holding_longer_types_only_once(self):
        c, f = make()
        f.frames("A", 5.0)
        self.assertEqual(c.text, "A")

    def test_stalled_frames_cannot_type_from_two_frames(self):
        # regression: a 1-5 s stall (e.g. the first model call) made the 2nd frame exceed the hold time
        c = TextComposer(ComposerConfig())
        c.update("X", True, now=0.0)
        self.assertIsNone(c.update("X", True, now=1.3))
        self.assertEqual(c.update("X", True, now=1.35), "X")  # 3rd frame of evidence

    def test_progress(self):
        c, f = make()
        f.frames("A", 0.5)  # 5 frames: the hold is timed from the first one, so 0.4 s elapsed
        self.assertAlmostEqual(c.hold_progress(f.t), 0.4, places=5)
        f.frames("A", 0.7)
        self.assertEqual(c.text, "A")
        self.assertEqual(c.hold_progress(f.t), 0.0)  # typed, waiting for release
        self.assertTrue(c.waiting_for_release)


class DoubleLetterTests(unittest.TestCase):
    def test_release_then_resign_types_double_letter(self):
        c, f = make()
        f.frames("L", 1.2)
        f.frames(None, 0.5, hand=False)  # drop hands briefly
        f.frames("L", 1.2)
        self.assertEqual(c.text, "LL")

    def test_uncertain_release_also_works(self):
        c, f = make()
        f.frames("L", 1.2)
        f.frames(None, 0.5, hand=True)  # relax hands: prediction becomes uncertain
        f.frames("L", 1.2)
        self.assertEqual(c.text, "LL")

    def test_too_short_release_does_not_repeat(self):
        c, f = make()
        f.frames("L", 1.2)
        f.frames(None, 0.1)
        f.frames("L", 3.0)
        self.assertEqual(c.text, "L")

    def test_hello(self):
        c, f = make()
        for letter in "HEL":
            f.frames(letter, 1.2)
        f.frames(None, 0.5)
        for letter in "LO":
            f.frames(letter, 1.2)
        self.assertEqual(c.text, "HELLO")


class SpacingAndEditingTests(unittest.TestCase):
    def test_auto_space_after_hands_leave(self):
        c, f = make()
        f.frames("H", 1.2)
        f.frames("I", 1.2)
        typed = f.frames(None, 2.5, hand=False)
        self.assertEqual(c.text, "HI ")
        self.assertEqual(typed, [" "])
        self.assertEqual(c.last_word(), "HI")
        f.frames(None, 5.0, hand=False)
        self.assertEqual(c.text, "HI ")  # only one space

    def test_no_auto_space_while_hands_visible_but_uncertain(self):
        c, f = make()
        f.frames("H", 1.2)
        f.frames(None, 5.0, hand=True)
        self.assertEqual(c.text, "H")

    def test_auto_space_disabled(self):
        c, f = make(auto_space_seconds=0)
        f.frames("H", 1.2)
        f.frames(None, 5.0, hand=False)
        self.assertEqual(c.text, "H")

    def test_no_space_on_empty_text(self):
        c, f = make()
        f.frames(None, 5.0, hand=False)
        self.assertEqual(c.text, "")
        self.assertFalse(c.add_space())

    def test_manual_space_backspace_clear(self):
        c, f = make()
        f.frames("A", 1.2)
        self.assertTrue(c.add_space())
        self.assertFalse(c.add_space())  # no double spaces
        self.assertEqual(c.text, "A ")
        c.backspace()
        c.backspace()
        c.backspace()  # extra backspace on empty text is harmless
        self.assertEqual(c.text, "")
        f.frames(None, 0.5, hand=False)
        f.frames("B", 1.2)
        c.clear()
        self.assertEqual(c.text, "")

    def test_backspace_while_holding_does_not_retype(self):
        c, f = make()
        f.frames("A", 1.2)
        c.backspace()
        f.frames("A", 2.0)
        self.assertEqual(c.text, "")

    def test_backspace_over_auto_space_sticks_while_hands_away(self):
        # regression: the auto-space used to come straight back (and be spoken again) every frame
        c, f = make()
        f.frames("H", 1.2)
        f.frames("I", 1.2)
        self.assertEqual(f.frames(None, 2.5, hand=False), [" "])
        c.backspace()
        self.assertEqual(f.frames(None, 1.0, hand=False), [])
        self.assertEqual(c.text, "HI")
        c.backspace()
        f.frames(None, 1.0, hand=False)
        self.assertEqual(c.text, "H")

    def test_deleting_a_letter_keeps_the_auto_space(self):
        # regression: backspacing a wrong letter used to cancel the word's space, gluing words
        c, f = make()
        for letter in "HIX":
            f.frames(letter, 1.2)
        c.backspace()
        self.assertEqual(f.frames(None, 2.5, hand=False), [" "])
        self.assertEqual(c.text, "HI ")

    def test_backspacing_everything_leaves_no_stray_space(self):
        c, f = make()
        f.frames("A", 1.2)
        c.backspace()
        f.frames(None, 3.0, hand=False)
        self.assertEqual(c.text, "")

    def test_progress_never_full_without_enough_frames(self):
        c = TextComposer(ComposerConfig(hold_seconds=1.0))
        c.update("A", True, now=0.0)
        self.assertIsNone(c.update("A", True, now=1.0))
        self.assertLess(c.hold_progress(1.0), 1.0)

    def test_auto_space_rearms_after_next_letter(self):
        c, f = make()
        f.frames("A", 1.2)
        f.frames(None, 2.5, hand=False)
        c.backspace()  # cancel this word's space
        f.frames("B", 1.2)
        f.frames(None, 2.5, hand=False)
        self.assertEqual(c.text, "AB ")

    def test_max_chars(self):
        c, f = make(max_chars=2)
        for letter in "ABC":
            f.frames(letter, 1.2)
        self.assertEqual(c.text, "AB")


if __name__ == "__main__":
    unittest.main()
