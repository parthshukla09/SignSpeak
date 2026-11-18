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


if __name__ == "__main__":
    unittest.main()
