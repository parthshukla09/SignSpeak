import tempfile
import unittest
from pathlib import Path

import collect
from signspeak.userdata import list_samples, next_take_number, split_by_take, take_of

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
NAMES = [chr(c) for c in range(ord("A"), ord("Z") + 1)]


class UserDataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, letter, take, count):
        d = self.root / letter
        d.mkdir(exist_ok=True)
        for i in range(count):
            (d / f"take{take:02d}_{i + 1:03d}.jpg").write_bytes(JPEG)

    def test_take_numbers(self):
        self.assertEqual(take_of(Path("take07_012.jpg")), 7)
        self.assertIsNone(take_of(Path("notes.jpg")))
        self.assertEqual(next_take_number(self.root / "D"), 1)
        self.make("D", 1, 2)
        self.make("D", 3, 2)
        self.assertEqual(next_take_number(self.root / "D"), 4)

    def test_last_take_is_held_out_per_letter(self):
        self.make("D", 1, 5)
        self.make("D", 2, 5)
        self.make("D", 3, 4)
        self.make("P", 1, 6)  # a single take stays in training
        (self.root / "P" / "stray.jpg").write_bytes(JPEG)  # not a take file: ignored
        samples = list_samples(self.root, NAMES)
        self.assertEqual(len(samples), 20)
        train, val = split_by_take(samples)
        self.assertEqual({(s.letter, s.take) for s in val}, {("D", 3)})
        self.assertEqual(len(val), 4)
        self.assertEqual(len(train), 16)
        self.assertEqual({s.label for s in samples}, {NAMES.index("D"), NAMES.index("P")})

    def test_collect_letter_parsing(self):
        self.assertEqual(collect.parse_args([]).letter_list, list("DPNMTK"))
        self.assertEqual(collect.parse_args(["--letters", "d p d"]).letter_list, ["D", "P"])
        self.assertEqual(len(collect.parse_args(["--letters", "all"]).letter_list), 26)
        for bad in (["--letters", "1"], ["--takes", "0"]):
            with self.subTest(argv=bad), self.assertRaises(SystemExit):
                import contextlib, io  # noqa: E401
                with contextlib.redirect_stderr(io.StringIO()):
                    collect.parse_args(bad)


if __name__ == "__main__":
    unittest.main()
