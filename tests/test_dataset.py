import tempfile
import unittest
from pathlib import Path

from signspeak.dataset import is_image_file, list_split

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16


class DatasetListingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        for cls in "AB":
            (self.root / cls).mkdir()
            for i in range(10):
                (self.root / cls / f"{i}.jpg").write_bytes(JPEG)
        (self.root / "A" / "fake.jpg").write_bytes(b"import os\n")  # like Testing/E/301.jpg
        (self.root / "A" / "notes.txt").write_text("ignored")

    def tearDown(self):
        self.tmp.cleanup()

    def test_non_images_named_jpg_are_skipped(self):
        self.assertFalse(is_image_file(self.root / "A" / "fake.jpg"))
        self.assertTrue(is_image_file(self.root / "A" / "0.jpg"))
        items, skipped = list_split(self.root, ["A", "B"])
        self.assertEqual(len(items), 20)
        self.assertEqual([p.name for p in skipped], ["fake.jpg"])
        self.assertEqual({label for _, label in items}, {0, 1})

    def test_limit_is_deterministic_and_spread_out(self):
        items, _ = list_split(self.root, ["A", "B"], limit_per_class=3)
        a = [p.name for p, label in items if label == 0]
        self.assertEqual(a, ["0.jpg", "4.jpg", "9.jpg"])  # sorted names: 0,1,...,9 -> ends + middle
        self.assertEqual(items, list_split(self.root, ["A", "B"], limit_per_class=3)[0])

    def test_missing_class_folder_exits(self):
        with self.assertRaises(SystemExit):
            list_split(self.root, ["A", "B", "C"])


if __name__ == "__main__":
    unittest.main()
