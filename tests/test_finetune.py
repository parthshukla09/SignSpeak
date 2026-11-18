import unittest

import numpy as np


class MaskedDenseTests(unittest.TestCase):
    def test_only_recorded_letters_change(self):
        import tensorflow as tf

        import finetune

        rng = np.random.default_rng(0)
        w0 = rng.normal(size=(8, 5)).astype(np.float32)
        b0 = rng.normal(size=5).astype(np.float32)
        layer = finetune.masked_dense(tf, w0, b0, trainable_classes=[1, 3])
        inp = tf.keras.Input((8,))
        model = tf.keras.Model(inp, layer(inp))
        model.compile(tf.keras.optimizers.Adam(0.05), "sparse_categorical_crossentropy")
        x = rng.normal(size=(64, 8)).astype(np.float32)
        y = rng.integers(0, 5, size=64)
        model.fit(x, y, epochs=5, verbose=0)

        w, b = layer.final_weights()
        frozen = [0, 2, 4]
        np.testing.assert_array_equal(w[:, frozen], w0[:, frozen])
        np.testing.assert_array_equal(b[frozen], b0[frozen])
        self.assertFalse(np.allclose(w[:, [1, 3]], w0[:, [1, 3]]))  # the recorded ones did train
        probs = model.predict(x[:4], verbose=0)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)


class BundleTests(unittest.TestCase):
    def test_fingerprint_ignores_only_the_last_layer(self):
        import tensorflow as tf

        import finetune

        inp = tf.keras.Input((6,))
        out = tf.keras.layers.Dense(3, activation="softmax")(tf.keras.layers.Dense(4, activation="relu")(inp))
        model = tf.keras.Model(inp, out)
        fp = finetune.extractor_fingerprint(model)
        w, b = model.layers[-1].get_weights()
        model.layers[-1].set_weights([w + 1, b + 1])  # what fine-tuning changes
        self.assertEqual(finetune.extractor_fingerprint(model), fp)
        w, b = model.layers[-2].get_weights()
        model.layers[-2].set_weights([w + 1, b])  # a different feature extractor
        self.assertNotEqual(finetune.extractor_fingerprint(model), fp)

    def test_save_and_load_round_trip_and_mismatches(self):
        import tempfile
        from pathlib import Path

        import finetune

        rng = np.random.default_rng(0)
        names = list("ABC")
        replay = (rng.random((6, 8), np.float32), rng.random((6, 8), np.float32), np.array([0, 0, 1, 1, 2, 2]))
        test = (rng.random((3, 8), np.float32), rng.random((3, 8), np.float32), np.array([0, 1, 2]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "b.npz"
            finetune.save_bundle(path, "fp1", names, replay, test, 2, 1)
            (r, t), why = finetune.load_bundle(path, "fp1", names, 2, 1)
            self.assertIsNone(why)
            np.testing.assert_allclose(r[0], replay[0], atol=1e-3)  # stored as float16
            np.testing.assert_array_equal(t[2], test[2])
            self.assertEqual(r[0].dtype, np.float32)
            self.assertIn("different model", finetune.load_bundle(path, "fp2", names, 2, 1)[1])
            self.assertIn("different classes", finetune.load_bundle(path, "fp1", list("ABD"), 2, 1)[1])
            self.assertIn("per letter", finetune.load_bundle(path, "fp1", names, 5, 1)[1])
            self.assertIsNone(finetune.load_bundle(path, "fp1", names, 2, 0)[1])  # no test sample is fine
            self.assertIn("no bundled", finetune.load_bundle(Path(tmp) / "none.npz", "fp1", names, 2, 1)[1])

    def test_shipped_bundle_matches_the_shipped_model(self):
        from pathlib import Path

        import finetune
        from signspeak.model import MODEL_PATH, load_class_names, load_keras_model

        if not finetune.BUNDLE_PATH.is_file():
            self.skipTest("models/replay_features.npz not present")
        model = load_keras_model(MODEL_PATH)
        names = load_class_names(Path(MODEL_PATH).parent.parent / "class_names.json")
        bundle, why = finetune.load_bundle(finetune.BUNDLE_PATH, finetune.extractor_fingerprint(model), names, 100, 30)
        self.assertIsNone(why)
        (fr, _, yr), (ft, _, yt) = bundle
        self.assertEqual((len(yr), len(yt)), (2600, 780))
        self.assertEqual(sorted(set(yr.tolist())), list(range(26)))


if __name__ == "__main__":
    unittest.main()
