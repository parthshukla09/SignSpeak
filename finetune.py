"""Fine-tune SignSpeak to YOUR signs, recorded with collect.py.

Only the model's final layer is retrained, and only the weights of the letters you recorded may
change. Every other letter keeps exactly its original weights, so it cannot get worse by being
pulled towards a different letter. (Retraining the whole final layer made I drop from 100% to 73%
on the dataset even though I was never recorded.) A sample of the original RealSign training
images is mixed in ("replay"), so the recorded letters do not start claiming images of other
letters. The rest of the network is kept, so its features are computed once and cached, and
training takes a minute or two on a laptop CPU.

The last take of each letter is held out. The script prints accuracy on it before and after
fine-tuning, plus accuracy on a sample of the RealSign test images, to check that nothing else
got worse. The result goes to runs/<name>/; models/ is never changed.

  python finetune.py
  python webcam.py --model runs/<name>/SignSpeak.keras --classes runs/<name>/class_names.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from signspeak.dataset import list_split
from signspeak.model import (CLASS_NAMES_PATH, MODEL_PATH, PROJECT_ROOT, load_class_names,
                             load_keras_model, to_model_input)
from signspeak.userdata import list_samples, split_by_take


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=str(MODEL_PATH), help="model to start from")
    p.add_argument("--classes", default=str(CLASS_NAMES_PATH))
    p.add_argument("--user-data", default=str(PROJECT_ROOT / "data" / "user"), help="recordings from collect.py")
    p.add_argument("--dataset", default=str(PROJECT_ROOT / "data" / "RealSign"), help="RealSign dataset root")
    p.add_argument("--replay-per-class", type=int, default=100,
                   help="RealSign training images per letter mixed in (default 100)")
    p.add_argument("--test-per-class", type=int, default=30,
                   help="RealSign test images per letter used to check for regressions (default 30)")
    p.add_argument("--no-cache", action="store_true",
                   help="recompute the dataset features instead of reusing runs/cache/")
    p.add_argument("--cache-only", action="store_true",
                   help="only compute and cache the dataset features, then exit")
    p.add_argument("--user-weight", type=float, default=1.0,
                   help="how much your recordings of a letter count relative to the dataset's (default 1)")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--name", default="finetune-" + time.strftime("%Y%m%d-%H%M%S"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.replay_per_class < 1 or args.test_per_class < 0 or args.epochs < 1 or args.user_weight <= 0:
        p.error("--replay-per-class and --epochs must be >= 1, --test-per-class >= 0, --user-weight > 0")
    return args


def masked_dense(tf, w0: np.ndarray, b0: np.ndarray, trainable_classes: list[int]):
    """A softmax Dense layer that starts at (w0, b0) where only the given classes' weights change."""

    class MaskedDense(tf.keras.layers.Layer):
        def __init__(self):
            super().__init__()
            mask = np.zeros(len(b0), np.float32)
            mask[trainable_classes] = 1.0
            self.w0, self.b0, self.mask = tf.constant(w0), tf.constant(b0), tf.constant(mask)
            self.dw = self.add_weight(shape=w0.shape, initializer="zeros", name="dw")
            self.db = self.add_weight(shape=b0.shape, initializer="zeros", name="db")

        def call(self, x):
            return tf.nn.softmax(tf.matmul(x, self.w0 + self.dw * self.mask)
                                 + self.b0 + self.db * self.mask)

        def final_weights(self) -> list[np.ndarray]:
            return [(self.w0 + self.dw * self.mask).numpy(), (self.b0 + self.db * self.mask).numpy()]

    return MaskedDense()


class FeatureCache:
    """Features (the input to the final Dense layer) for images and their mirror images."""

    def __init__(self, tf, model):
        self.extractor = tf.keras.Model(model.inputs, model.layers[-1].input)
        self.size = model.input_shape[1]

    def __call__(self, paths: list[Path], batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
        feats, flipped = [], []
        for i in range(0, len(paths), 256):  # bounded memory
            imgs = []
            for p in paths[i:i + 256]:
                img = cv2.imread(str(p))
                if img is None:
                    raise RuntimeError(f"cannot read {p}")
                imgs.append(to_model_input(img, self.size))
            batch = np.stack(imgs)
            feats.append(self.extractor.predict(batch, verbose=0, batch_size=batch_size))
            flipped.append(self.extractor.predict(batch[:, :, ::-1], verbose=0, batch_size=batch_size))
        return np.concatenate(feats), np.concatenate(flipped)


def dataset_features(feats: FeatureCache, split_dir: Path, names: list[str], per_class: int,
                     base: Path, use_cache: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Features of a RealSign split sample, cached in runs/cache/ (they only depend on the base
    model and the sample, not on your recordings, so re-runs after recording more are fast)."""
    st = base.stat()
    key = hashlib.sha1(f"{base.resolve()}|{st.st_size}|{st.st_mtime_ns}|{split_dir.resolve()}|"
                       f"{per_class}|{','.join(names)}".encode()).hexdigest()[:16]
    cache = PROJECT_ROOT / "runs" / "cache" / f"{split_dir.name}_{per_class}_{key}.npz"
    if use_cache and cache.exists():
        z = np.load(cache)
        return z["f"], z["f_flip"], z["y"]
    items, _ = list_split(split_dir, names, per_class)
    f, f_flip = feats([p for p, _ in items])
    y = np.array([label for _, label in items], dtype=int)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, f=f, f_flip=f_flip, y=y)
    return f, f_flip, y


def accuracy_report(probs_fn, f, f_flip, y, names, title) -> dict:
    """Flip-averaged accuracy (as the webcam app predicts), per letter, and top confusions."""
    if len(y) == 0:
        return {}
    pred = ((probs_fn(f) + probs_fn(f_flip)) / 2).argmax(1)
    ok = pred == y
    per = {names[c]: float(ok[y == c].mean()) for c in sorted(set(y.tolist()))}
    mistakes = Counter((names[t], names[p]) for t, p in zip(y, pred) if t != p).most_common(5)
    print(f"  {title}: {ok.mean() * 100:5.1f}%   "
          + "  ".join(f"{k} {v * 100:.0f}%" for k, v in per.items() if len(per) <= 12)
          + ("" if not mistakes else "   mistakes: " + ", ".join(f"{t}->{p} x{n}" for (t, p), n in mistakes)))
    return {"accuracy": float(ok.mean()), "per_letter": per,
            "mistakes": [[t, p, n] for (t, p), n in mistakes]}


def main(argv=None) -> int:
    args = parse_args(argv)
    names = load_class_names(args.classes)
    n = len(names)
    samples = list_samples(args.user_data, names)
    if not samples and not args.cache_only:
        sys.exit(f"No recordings found in {args.user_data}. Record some first: python collect.py")
    train_u, val_u = split_by_take(samples)
    takes = Counter((s.letter, s.take) for s in samples)
    per_letter = Counter(s.letter for s in samples)
    if samples:
        print("Your recordings: " + ", ".join(
            f"{k} {v} imgs/{len({t for (l, t) in takes if l == k})} takes"
            for k, v in sorted(per_letter.items())))
        if not train_u:
            sys.exit("Every recording is in a held-out take. Record at least 2 takes per letter.")
        if not val_u:
            print("  (no letter has 2+ takes, so there is no held-out take to measure improvement on)")
        missing = [c for c in names if c not in per_letter]
        if missing:
            print(f"  WARNING: {len(missing)} letters are not recorded ({''.join(missing)}). With your "
                  "camera they may now be read as recorded letters,\n  because every image from your "
                  "setting was one of those. Record them too:  python collect.py --letters ALL")

    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    model = load_keras_model(args.base)
    dense = model.layers[-1]
    if not isinstance(dense, tf.keras.layers.Dense) or dense.units != n:
        sys.exit(f"{args.base}: expected the last layer to be Dense({n})")
    w0, b0 = dense.get_weights()

    t0 = time.perf_counter()
    print("Computing features (the dataset's are cached after the first run)...", flush=True)
    feats = FeatureCache(tf, model)
    data = Path(args.dataset)
    fr, fr_flip, yr = dataset_features(feats, data / "Training", names, args.replay_per_class,
                                       Path(args.base), not args.no_cache)
    if args.test_per_class:
        ft, ft_flip, yt = dataset_features(feats, data / "Testing", names, args.test_per_class,
                                           Path(args.base), not args.no_cache)
    else:
        ft = ft_flip = np.zeros((0, w0.shape[0]))
        yt = np.zeros(0, dtype=int)
    if args.cache_only:
        print(f"Dataset features cached in {time.perf_counter() - t0:.0f}s.")
        return 0
    fu_tr, fu_tr_flip = feats([s.path for s in train_u])
    fu_va, fu_va_flip = feats([s.path for s in val_u]) if val_u else (np.zeros((0, w0.shape[0])),) * 2
    yu_tr = np.array([s.label for s in train_u], dtype=int)
    yu_va = np.array([s.label for s in val_u], dtype=int)
    print(f"  done in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- training set
    # images and their mirror images; your recordings of each letter weigh user_weight times
    # as much as that letter's replay images
    x = np.concatenate([fr, fr_flip, fu_tr, fu_tr_flip])
    y = np.concatenate([yr, yr, yu_tr, yu_tr])
    replay_count = Counter(yr.tolist())
    user_count = Counter(yu_tr.tolist())
    w_user = np.array([args.user_weight * replay_count[c] / user_count[c] for c in yu_tr], dtype=np.float32)
    weights = np.concatenate([np.ones(2 * len(yr), np.float32), w_user, w_user])

    def head(weights_and_bias):
        inp = tf.keras.Input((w0.shape[0],))
        out = tf.keras.layers.Dense(n, activation="softmax")(tf.keras.layers.Dropout(0.2)(inp))
        m = tf.keras.Model(inp, out)
        m.layers[-1].set_weights(weights_and_bias)
        return m

    before = head([w0, b0])
    probs_before = lambda f: before.predict(f, verbose=0, batch_size=256)  # noqa: E731
    print("\nBefore fine-tuning:")
    m_before = {"your_heldout_take": accuracy_report(probs_before, fu_va, fu_va_flip, yu_va, names, "your held-out take"),
                "dataset_test": accuracy_report(probs_before, ft, ft_flip, yt, names, "RealSign test sample")}

    # only the recorded letters' weights may change (see the module docstring)
    recorded = sorted(user_count)
    layer = masked_dense(tf, w0, b0, recorded)
    inp = tf.keras.Input((w0.shape[0],))
    trainer = tf.keras.Model(inp, layer(tf.keras.layers.Dropout(0.2)(inp)))
    trainer.compile(optimizer=tf.keras.optimizers.Adam(args.lr), loss="sparse_categorical_crossentropy")
    trainer.fit(x, y, sample_weight=weights, epochs=args.epochs, batch_size=64, shuffle=True, verbose=0)
    new_weights = layer.final_weights()
    model_head = head(new_weights)
    probs_after = lambda f: model_head.predict(f, verbose=0, batch_size=256)  # noqa: E731
    print(f"After fine-tuning (only {''.join(names[c] for c in recorded)} changed):")
    m_after = {"your_training_takes": accuracy_report(probs_after, fu_tr, fu_tr_flip, yu_tr, names, "your training takes"),
               "your_heldout_take": accuracy_report(probs_after, fu_va, fu_va_flip, yu_va, names, "your held-out take"),
               "dataset_test": accuracy_report(probs_after, ft, ft_flip, yt, names, "RealSign test sample")}
    changed = {k: (m_before["dataset_test"]["per_letter"][k], v)
               for k, v in m_after.get("dataset_test", {}).get("per_letter", {}).items()
               if abs(v - m_before["dataset_test"]["per_letter"][k]) > 1e-9}
    if changed:
        print("  RealSign letters that changed: " + ", ".join(
            f"{k} {b * 100:.0f}%->{a * 100:.0f}%" for k, (b, a) in sorted(changed.items())))

    # ---------------------------------------------------------------- save the full model
    out = PROJECT_ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    dense.set_weights(new_weights)
    model_path = out / "SignSpeak.keras"
    model.save(model_path)
    (out / "class_names.json").write_text(json.dumps(names), encoding="utf-8")

    # sanity check: the saved file reproduces the fine-tuned head's predictions
    check = [s.path for s in (val_u or train_u)[:8]]
    reloaded = load_keras_model(model_path)
    batch = np.stack([to_model_input(cv2.imread(str(p)), model.input_shape[1]) for p in check])
    diff = float(np.abs(reloaded.predict(batch, verbose=0) - probs_after(feats.extractor.predict(batch, verbose=0))).max())
    if diff > 1e-4:
        sys.exit(f"Saved model does not match the fine-tuned head (max diff {diff}); not using it.")

    (out / "metrics.json").write_text(json.dumps({
        "before": m_before, "after": m_after, "recordings": dict(per_letter), "args": vars(args),
    }, indent=2), encoding="utf-8")
    drop = (m_before["dataset_test"].get("accuracy", 0) - m_after["dataset_test"].get("accuracy", 0)) * 100
    if drop > 2:
        print(f"\nWarning: accuracy on the RealSign sample fell {drop:.1f} points; "
              "try --user-weight 0.5 or record more varied takes.")
    print(f"\nSaved {model_path}")
    print(f"Try it:  python webcam.py --model {model_path} --classes {out / 'class_names.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
