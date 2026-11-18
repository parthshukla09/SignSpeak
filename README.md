# SignSpeak

Real-time **Indian Sign Language (ISL) fingerspelling → text → speech**, using your webcam.

SignSpeak finds your hands with MediaPipe, crops the sign, classifies it as one of the 26 letters
A–Z with a fine-tuned MobileNetV2 model, builds words from the letters you hold, and reads finished
words aloud with your computer's built-in voice. Everything runs offline on a normal laptop CPU.

## Quick start

Requires Python 3.11 (64-bit) and a webcam.

```powershell
# Windows (PowerShell), from the project folder
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py        # the web app; opens in your browser
```

```bash
# macOS / Linux
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

The install is roughly 670 MB, mostly TensorFlow. The scripts can be started from any folder.

## The web app

`python app.py` starts a local server and opens **http://127.0.0.1:8000** in your browser. Allow
camera access when the browser asks. It has three tabs:

- **Translate:** sign letters to build text. The current letter, its confidence, the
  hold-to-type bar and the reference photo of the letter appear beside the video. Use the buttons
  or keys to speak, add a space, delete, clear and copy. Spoken sentences go into a history list.
- **Learn:** practise the alphabet. The app shows a letter and its reference photo, and you sign
  it until it is recognised. It keeps per-letter statistics and has an alphabet chart.
- **Personalise:** record your own signs and fine-tune the model from the browser, with
  before/after accuracy per letter (see "Teach it your signs" below).

**Settings** (the gear icon) has the model choice (original or a personalised one), recognition
timing and confidence, mirror view, theme, and voice and speech options. The model defaults to
your newest personalised model, if you have one.

The server listens only on `127.0.0.1` and refuses requests from other websites, so a page you
visit in the same browser cannot reach it. Camera frames never leave your computer. Options:
`--port 8080` and `--no-browser`. The browser/server protocol is documented in `web/PROTOCOL.md`.

`webcam.py` is still available as a simple OpenCV window version.

## How to use it (webcam.py)

| To… | Do this |
|---|---|
| Type a letter | Hold the sign steady until the green bar fills (1 s) |
| Type the same letter twice (the LL in HELLO) | Relax or drop your hands briefly, then sign the letter again |
| Finish a word | Lower your hands out of view for 2 s. A space is added and the word is spoken |
| Add a space / speak the word | `SPACE` |
| Delete the last character | `BACKSPACE` |
| Clear all text | `C` |
| Speak everything typed so far | `ENTER` |
| Quit | `Q` or `Esc`, or close the window |

Most ISL letters use **both hands**, so keep both hands in view. The video is shown mirrored, like
a selfie camera. The green box marks the area the model classifies. The **model sees** panel on
the right shows that exact image, stretched as the model receives it, with the model's top 3
guesses.

### If letters come out wrong

- Sign the **Indian** Sign Language alphabet, not American (ASL). Most ISL letters use both hands,
  while ASL letters use one, so ASL signs will be misread.
- Sign at chest height, with both hands inside the green box and the "model sees" panel showing
  your hands large and clear. The training photos are close-ups of hands in front of a T-shirt.
- Use even lighting and a plain top. The model was trained on only four signers, so very different
  lighting, backgrounds or camera angles reduce accuracy.
- The most reliable letters are C, H, I, O, E, Y, F and L. Z, S, V and W are often confused (see Accuracy).

Useful options (`python webcam.py --help` lists all of them):

```
--camera 1          use a different camera
--hold 0.7          type letters faster (seconds a sign must be held)
--auto-space 0      turn off automatic spaces
--threshold 0.7     require more confidence before a letter counts
--no-speech         turn off text-to-speech
--speak-letters     also say each letter as it is typed
--no-mirror         show the camera image unmirrored
--no-flip-average   classify only the camera image, not also its mirror image (a bit faster)
--no-preview        hide the "model sees" panel
--model PATH --classes PATH   use a model you trained yourself
```

## Teach it your signs (recommended)

The model learned from four signers, so some of your letters may be misread, for example if you
form D, N or T a little differently. Fixing that takes about 10 minutes and **needs nothing
beyond this repository**: no dataset download and no GPU. The easiest way is the web app's
**Personalise** tab (`python app.py`). From the command line:

```bash
python collect.py --letters ALL --per-take 20   # about 8 minutes; skips letters already recorded
python finetune.py                              # about a minute on a laptop; saves runs/finetune-<time>/
python webcam.py --model runs/finetune-<time>/SignSpeak.keras --classes runs/finetune-<time>/class_names.json
```

Fine-tuning mixes in features of about 2,600 dataset images, so letters stay calibrated, and
checks about 780 more for regressions. Those features ship with the project in
`models/replay_features.npz` (15 MB), computed from the RealSign dataset (CC0), so the 657 MB
dataset is not needed. If you do download the dataset to `data/RealSign`, it is used instead.

- **`collect.py`** shows the dataset's reference photo of each letter next to your camera. For each
  letter you record 3 short takes while moving your hands slightly. The images are exactly the
  crops the app classifies, saved to `data/user/` (ignored by git).
- **`finetune.py`** retrains only the model's final layer, and only the weights of the letters you
  recorded. It mixes in dataset images, holds out each letter's last take, and prints accuracy
  before and after, plus the per-letter change on a dataset sample. `models/` is never modified.

**Record all letters, not just the wrong ones.** Recording only D/P/N/M/T/K raised those letters
from 34% to 96% on a held-out take, but F, H and R then stopped working for that user. The F, H
and R weights were unchanged. The cause was that every training image from the user's own camera
and room was one of the six recorded letters, so the model partly learned "this camera means one
of those six". The dataset check could not catch it, because it uses the dataset's photos, not
the user's. With every letter recorded, the camera no longer predicts the letter, and the model
has to rely on the hand shape.

With all 26 letters recorded (3 takes of 20–30 images each), accuracy on that user's held-out takes
went from **57.6% with the original model to 98.6%**. Every letter scored 100% except H (80%) and
M (87%). The held-out take was recorded in the same session, so expect somewhat lower accuracy on
another day or in different light. Record a few more takes then and run `finetune.py` again.

## How it works

1. **Hand detection:** MediaPipe Hands finds up to two hands. The largest hand, plus any other
   hand close to it, form the sign. This keeps both hands of a two-handed letter together while
   ignoring a stray hand elsewhere in the frame.
2. **Crop:** the hands are framed the way the training images were, with margins of about a third
   of the hands' width and height. The margins were measured on the dataset's Validation split.
   The crop is cut out *before* anything is drawn on the frame.
3. **Classify:** the crop is stretched to 224×224 RGB, exactly as the training images were, and
   passed to the model, which scales the pixels itself. Keeping the training images' framing and
   stretching matters: on test images placed in a larger frame, it scores 84.5–85.5% against
   80.6% for a square crop that keeps the aspect ratio. Each crop is also classified mirrored and
   the two results are averaged. The training photos mix mirrored and unmirrored captures, so this
   makes left- and right-handed signing equally reliable (test accuracy 90.7% as stored, 91.1%
   flipped, 91.3% averaged). Predictions are averaged over the last 7 frames, and a letter only
   counts at 60% confidence or more.
4. **Compose:** a letter is typed after it has been held for 1 s, and must be released before it
   can repeat. Brief mispredictions while your hands move between signs are never typed.
5. **Speak:** finished words are spoken on a background thread, so the video never freezes.
   Windows uses the built-in SAPI voice; macOS and Linux use `pyttsx3`.

## Dataset

The model is trained on the **[RealSign Indian Sign Language Dataset](https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset)**
(CC0 1.0): about 26,000 images of fingerspelled ISL letters A–Z from four signers, already split
into `Training` (700 per letter), `Validation` (100) and `Testing` (200).

The dataset is not stored in this repository. You only need it for `evaluate.py` and `train.py`;
the app and personalisation work without it. To rebuild the bundled fine-tuning features after
changing the base model, run `python finetune.py --export-bundle`. To download the dataset:

```bash
mkdir data
curl -L -o data/Dataset.zip https://media.githubusercontent.com/media/RealSign62/RealSign-Indian-Sign-Language-Dataset/main/Dataset.zip
# extract so that you get data/RealSign/Training, data/RealSign/Validation, data/RealSign/Testing
```

On Windows, extract `Dataset.zip` into `data\RealSign` with Explorer or
`tar -xf data\Dataset.zip -C data\RealSign` (create the folder first). The `data/` folder is
ignored by git.

## Accuracy

Measured with `evaluate.py` on the dataset's `Testing` split (5,200 images):

| Input to the model | Accuracy |
|---|---|
| Dataset images as stored (how the model was trained) | **90.6%** (4,708 / 5,198) |
| Simulated webcam: each image placed in a larger frame, then the app's own hand detection and crop (`--crop hands`) | **84.8%** (4,407 / 5,198) |

The simulated-webcam figure is the better guide to live use. On the stored images, 21 of the 26
letters score 99% or more. Through the webcam path, the weak letters are **Z 27%, S 51%, V 61%,
W 66%, Q 72%, N 75% and X 75%**. The most common mix-ups are V→U, S→I, Z→E/T, N→H and Q→R. Two of
the 5,200 test files are not images (they are Python scripts named `301.jpg` inside the dataset)
and are skipped.

These figures come from the same four signers the model was trained on. Expect lower accuracy on
a new person, new lighting or a new camera. To reproduce them:

```bash
python evaluate.py                 # dataset images as stored
python evaluate.py --crop hands    # through the webcam app's MediaPipe crop
python evaluate.py --confusion-csv confusion.csv --json summary.json
```

## Training your own model

`train.py` rebuilds the model with the recipe recovered from the original model file:
MobileNetV2 (ImageNet weights) → global average pooling → dropout 0.2 → 26-way softmax, with light
augmentation. Training runs in two stages: first the new head with the backbone frozen (Adam 1e-3),
then fine-tuning from `block_14` onward at 1e-5 with BatchNorm frozen.

```bash
python train.py                                   # full run; saves to runs/<timestamp>/
python train.py --limit-per-class 20 --head-epochs 1 --finetune-epochs 1 --name smoke   # quick check
python webcam.py --model runs/<name>/SignSpeak.keras --classes runs/<name>/class_names.json
```

On a laptop CPU, one epoch over the full training split takes roughly 45 minutes. A GPU, for
example Google Colab, is much faster. Training never overwrites `models/`; copy a model there
yourself if you want it to become the default.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The tests cover the crop geometry (inside the frame, contains both hands, never empty), the typing
rules (hold-to-confirm, double letters, auto-space, editing), dataset listing, argument checks
and the on-screen layout at several frame sizes. They need neither a camera nor the model.

## Project layout

```
app.py               the web app (starts the local server and opens the browser)
webcam.py            the OpenCV window version
web/                 the web frontend (index.html, styles.css, app.js), reference photos, PROTOCOL.md
evaluate.py          accuracy on a dataset split
train.py             train / reproduce the model
collect.py           record your own signs
finetune.py          adapt the model to your recordings
signspeak/
  hands.py           MediaPipe wrapper and crop geometry
  model.py           model loading, input preparation, prediction smoothing
  composer.py        letters → text (hold-to-confirm, double letters, spaces)
  speech.py          background text-to-speech
  dataset.py         dataset listing (skips non-image files hidden in the archive)
  userdata.py        your recordings, split into training and held-out takes
  camera.py          robust webcam opening (waits for frames, DirectShow fallback on Windows)
  engine.py          the per-session recognition pipeline shared by the web app and webcam.py
  practice.py        Learn mode logic
  registry.py        the models the app can use (original + runs/)
  server.py          the web server (FastAPI + WebSocket)
models/SignSpeak_finetuned.keras   trained model (Keras 3, 22 MB)
models/replay_features.npz         dataset features used by fine-tuning (15 MB, no dataset needed)
class_names.json     output index → letter
tests/               unit tests
```

## Known limitations

- **J and Z involve motion** in ISL, but the model classifies single frames. It recognises the
  static pose captured in the training images, not the movement.
- **Weakest letters:** Z, S, V, W, Q, N and X (see Accuracy). Z (a motion letter) and S are the
  least reliable. Retraining with more varied examples of these letters is the most useful
  improvement.
- **Small signer pool:** the dataset has four signers, so expect lower accuracy for other people,
  skin tones, backgrounds and lighting until the model is trained on more varied data.
- **Letters only:** there are no digits or word-level signs yet.

## Credits

Dataset: Boggaram, A., Boggaram, A., Srinivasa Ramanujan, A., Sharma, A., & R, B.
*RealSign-Indian-Sign-Language-Dataset: An Open-Source Indian Sign Language Database.*
https://github.com/RealSign62/RealSign-Indian-Sign-Language-Dataset (CC0 1.0).
The 26 reference photos in `web/reference/` are the dataset's `Letters` images.
