"""Train the SignSpeak letter classifier on the RealSign ISL dataset.

This reproduces the recipe recovered from models/SignSpeak_finetuned.keras:
  MobileNetV2 (ImageNet, 224x224) -> global average pooling -> dropout 0.2 -> dense softmax,
  with light augmentation (rotation, zoom, translation, contrast), trained in two stages:
    1. head only, backbone frozen
    2. fine-tune from block_14 onwards at a low learning rate, BatchNorm layers kept frozen
Images go in as RGB 0..255; scaling to [-1, 1] happens inside the model (Rescaling layer),
which is exactly what webcam.py and evaluate.py feed it.

Output goes to runs/<name>/ and never overwrites models/. To use a trained model:
  python webcam.py --model runs/<name>/SignSpeak.keras --classes runs/<name>/class_names.json

Examples
  python train.py                                       # full training (slow on CPU)
  python train.py --head-epochs 1 --finetune-epochs 1 --limit-per-class 20 --name smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from signspeak.dataset import list_split
from signspeak.model import CLASS_NAMES_PATH, PROJECT_ROOT, load_class_names


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(PROJECT_ROOT / "data" / "RealSign"),
                   help="dataset root containing Training/Validation/Testing")
    p.add_argument("--classes", default=str(CLASS_NAMES_PATH), help="class order (JSON list)")
    p.add_argument("--name", default=time.strftime("%Y%m%d-%H%M%S"), help="run folder name under runs/")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=1e-5)
    p.add_argument("--finetune-from", default="block_14_expand",
                   help="first MobileNetV2 layer to unfreeze in stage 2")
    p.add_argument("--patience", type=int, default=3, help="early-stopping patience (epochs)")
    p.add_argument("--limit-per-class", type=int, default=0,
                   help="use at most N images per class per split (quick experiments; 0 = all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    if args.limit_per_class < 0:
        p.error("--limit-per-class must be 0 (all images) or more")
    for name in ("img_size", "batch_size"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("head_epochs", "finetune_epochs", "patience"):
        if getattr(args, name) < 0:
            p.error(f"--{name.replace('_', '-')} cannot be negative")
    return args


def make_dataset(tf, paths, labels, num_classes, img_size, batch_size, shuffle, seed):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))  # bilinear, float32 in 0..255, RGB
        img.set_shape((img_size, img_size, 3))
        return img, tf.one_hot(label, num_classes)

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_model(tf, num_classes: int, img_size: int):
    keras, layers = tf.keras, tf.keras.layers
    augmentation = keras.Sequential([
        layers.RandomRotation(0.05, fill_mode="reflect"),
        layers.RandomZoom(0.1, fill_mode="reflect"),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect"),
        layers.RandomContrast(0.1, value_range=(0, 255)),
    ], name="augmentation")
    base = keras.applications.MobileNetV2(input_shape=(img_size, img_size, 3),
                                          include_top=False, weights="imagenet")
    base.trainable = False

    inputs = keras.Input((img_size, img_size, 3), name="image")
    x = augmentation(inputs)
    # identical to mobilenet_v2.preprocess_input (x / 127.5 - 1) but needs no custom_objects to load
    x = layers.Rescaling(1 / 127.5, offset=-1.0, name="mobilenet_preprocess")(x)
    x = base(x, training=False)  # keep BatchNorm in inference mode, also while fine-tuning
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="letters")(x)
    return keras.Model(inputs, outputs, name="signspeak"), base


def main(argv=None) -> int:
    args = parse_args(argv)
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    data = Path(args.data)
    for split in ("Training", "Validation", "Testing"):
        if not (data / split).is_dir():
            sys.exit(f"Missing {data / split}. Download the dataset first - see 'Dataset' in README.md.")

    class_names = load_class_names(args.classes)
    n = len(class_names)
    out = PROJECT_ROOT / "runs" / args.name

    # check everything that can fail before spending any time on data or training
    model, base = build_model(tf, n, args.img_size)
    layer_names = [layer.name for layer in base.layers]
    if args.finetune_epochs > 0 and args.finetune_from not in layer_names:
        sys.exit(f"--finetune-from {args.finetune_from!r} is not a MobileNetV2 layer "
                 f"(e.g. block_13_expand, block_14_expand, block_15_expand, Conv_1)")

    splits = {}
    for split in ("Training", "Validation", "Testing"):
        items, skipped = list_split(data / split, class_names, args.limit_per_class)
        for p in skipped:
            print(f"  skipping {p}: not an image file")
        if not items:
            sys.exit(f"No images found in {data / split}")
        paths = [str(p) for p, _ in items]
        labels = [label for _, label in items]
        splits[split] = make_dataset(tf, paths, labels, n, args.img_size, args.batch_size,
                                     shuffle=(split == "Training"), seed=args.seed)
        print(f"{split}: {len(paths)} images", flush=True)

    early = dict(monitor="val_accuracy", patience=args.patience, restore_best_weights=True)
    history = {}

    print("\nStage 1: training the classification head (backbone frozen)", flush=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.head_lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(splits["Training"], validation_data=splits["Validation"], epochs=args.head_epochs,
                   callbacks=[tf.keras.callbacks.EarlyStopping(**early)], verbose=2)
    history["head"] = h1.history

    if args.finetune_epochs > 0:
        start = layer_names.index(args.finetune_from)
        base.trainable = True
        for i, layer in enumerate(base.layers):
            layer.trainable = i >= start and not isinstance(layer, tf.keras.layers.BatchNormalization)
        print(f"\nStage 2: fine-tuning from {args.finetune_from} "
              f"({sum(layer.trainable for layer in base.layers)} backbone layers trainable)", flush=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.finetune_lr),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(splits["Training"], validation_data=splits["Validation"],
                       epochs=args.finetune_epochs,
                       callbacks=[tf.keras.callbacks.EarlyStopping(**early)], verbose=2)
        history["finetune"] = h2.history

    test_loss, test_acc = model.evaluate(splits["Testing"], verbose=0)
    print(f"\nTest accuracy: {test_acc * 100:.2f}%  (loss {test_loss:.4f})")

    out.mkdir(parents=True, exist_ok=True)
    model_path = out / "SignSpeak.keras"
    model.save(model_path)
    (out / "class_names.json").write_text(json.dumps(class_names), encoding="utf-8")
    to_float = lambda h: {k: [float(v) for v in vals] for k, vals in h.items()}  # noqa: E731
    (out / "metrics.json").write_text(json.dumps({
        "test_accuracy": float(test_acc), "test_loss": float(test_loss),
        "history": {stage: to_float(h) for stage, h in history.items()},
        "args": vars(args),
    }, indent=2), encoding="utf-8")
    print(f"Saved {model_path}")
    print(f"Try it:  python webcam.py --model {model_path} --classes {out / 'class_names.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
