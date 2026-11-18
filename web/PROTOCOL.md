# SignSpeak web app: browser ↔ server protocol

The server (`signspeak/server.py`, started by `python app.py`) listens on `http://127.0.0.1:8000`
only. The browser owns the camera. It plays the video locally at full frame rate and sends JPEG
frames over a WebSocket. The server runs the recognition pipeline (`signspeak/engine.py`) and
answers each frame with a JSON result.

All coordinates are normalised to 0..1 of the frame the browser sent, and are **not mirrored**. If
the video is shown mirrored, the browser draws at `x' = 1 - x`.

## Static files

| URL | Serves |
|---|---|
| `GET /` | `web/index.html` |
| `GET /static/<path>` | `web/<path>`, e.g. `/static/app.js`, `/static/styles.css`, `/static/reference/A.jpg` |

`web/reference/<LETTER>.jpg` holds the RealSign reference photo for each of the 26 letters (CC0).

## HTTP API (JSON)

### `GET /api/state`
```json
{
  "classes": ["A", "B", "...", "Z"],
  "models": [{"id": "original", "label": "Original (RealSign dataset)", "kind": "original", "summary": "4 signers, 26 letters"},
             {"id": "run:finetune-user2", "label": "Personalised · finetune-user2", "kind": "personalised", "summary": "26 letters recorded, held-out 98.6%"}],
  "default_model": "run:finetune-user2",
  "config_defaults": {"threshold": 0.6, "smoothing": 7, "hold": 1.0, "release": 0.3, "auto_space": 2.0, "flip_average": true},
  "config_limits": {"threshold": [0, 1], "smoothing": [1, 30], "hold": [0, 5], "release": [0, 3], "auto_space": [0, 10]},
  "recordings": {"A": {"takes": 3, "images": 60}, "B": {"takes": 0, "images": 0}, "...": {}},
  "dataset_available": true,
  "finetune": { /* FinetuneStatus, see below */ }
}
```
`models` is ordered: the original first, then runs newest first. `kind` is `original`,
`personalised` (made by finetune.py) or `trained` (made by train.py). `dataset_available` is
false when `data/RealSign` is missing, and fine-tuning needs it.

### `POST /api/recordings/{letter}/delete-last`
Deletes the highest-numbered take of `letter`. → `{"removed": 20, "recordings": {...}}`.
Returns 404 for an unknown letter.

### `POST /api/finetune`
Starts fine-tuning in the background (it runs `finetune.py` as a subprocess). Body: `{}`.
- `202` → `{"finetune": FinetuneStatus}`
- `409` if a job is already running; `400` with `{"detail": "..."}` if there are no recordings
  or the dataset is missing.

### `GET /api/finetune` → `{"finetune": FinetuneStatus}`
```json
{
  "state": "idle | running | done | failed",
  "run": "web-20261118-221500",
  "model_id": "run:web-20261118-221500",
  "log": ["last", "200", "lines of output"],
  "started": 1795012345.6, "finished": null,
  "missing_letters": ["Q", "Z"],
  "result": {
    "before": {"your_heldout_take": {"accuracy": 0.576, "per_letter": {"A": 0.0, "...": 1.0}},
               "dataset_test": {"accuracy": 0.915, "per_letter": {...}}},
    "after":  {"your_heldout_take": {...}, "dataset_test": {...}, "your_training_takes": {...}}
  }
}
```
`result` is null until `state` is `done`, and `model_id` is null until then. `missing_letters`
lists the classes with no recordings when the job started. The UI should warn about them, because
unrecorded letters can get "stolen" by recorded ones (see README).

## WebSocket `/ws`

One connection per tab. Text messages are JSON objects with a `type`. Binary messages are
camera frames.

### Browser → server

| Message | Meaning |
|---|---|
| `{"type": "hello", "model": "<id>" or null, "config": {...partial config}}` | First message. Null model = server default. |
| *binary*: JPEG bytes | One unmirrored camera frame (about 640×480). **Send the next frame only after the `result` for the previous one arrives** (one in flight). |
| `{"type": "config", "config": {...partial}}` | Change recognition settings. Unknown keys are ignored and values are clamped to `config_limits`. |
| `{"type": "model", "model": "<id>"}` | Switch model (loading can take a few seconds). |
| `{"type": "mode", "mode": "translate" \| "learn" \| "record"}` | Text is only composed in `translate` mode. |
| `{"type": "command", "action": "space" \| "backspace" \| "clear"}` | Edit the text. |
| `{"type": "learn_start", "letters": ["A", "B"], "order": "random" \| "sequential", "hold": 0.8}` | Start practice (also switches to learn mode). An empty or missing `letters` means all. |
| `{"type": "learn_skip"}` | Skip the current target. |
| `{"type": "record_start", "letter": "D", "per_take": 20}` | Start saving crops for one take of `letter` (also switches to record mode). `per_take` is clamped to 5..60. |
| `{"type": "record_cancel"}` | Stop the current take and delete its files. |

### Server → browser

| Message | When |
|---|---|
| `{"type": "ready", "model": "<id>", "model_label": "...", "config": {...full}, "text": "...", "classes": [...]}` | After `hello` or `model`, once the model is loaded. Frames sent before `ready` are dropped. |
| `{"type": "loading", "model": "<id>"}` | A model is being loaded. |
| `{"type": "config", "config": {...full}}` | After `config`. |
| `{"type": "text", "text": "...", "word": "HELLO" or null}` | After `command`. `word` is set when a space finished a word (speak it). |
| `{"type": "learn", "learn": LearnState}` | After `learn_start` / `learn_skip`. |
| `{"type": "record", "record": RecordState, "recordings": {...}}` | After `record_start` / `record_cancel`. |
| `{"type": "result", ...}` | One per binary frame (see below). |
| `{"type": "error", "message": "..."}` | Bad message, unknown model, decode failure... The connection stays open. |

#### Frame result
```json
{
  "type": "result",
  "mode": "translate",
  "hands": [[[0.41, 0.52], "... 21 [x, y] points per hand (MediaPipe hand landmarks)"]],
  "box": [0.30, 0.35, 0.62, 0.80],
  "best": "D", "confidence": 0.94, "letter": "D",
  "top": [["D", 0.94], ["P", 0.04], ["K", 0.01]],
  "typed": null, "word": null,
  "text": "HELLO WOR",
  "progress": 0.45, "waiting_release": false,
  "ms": 118,
  "learn": null,
  "record": null
}
```
- `hands`: empty when no hand is visible; `box` and `best` are then null and `confidence` is 0.
- `letter` is `best` when `confidence >= threshold`, otherwise null (show "uncertain").
- `typed` is the character typed on this frame (a letter or `" "`). When it is `" "`, `word` holds
  the finished word, which should be spoken if speech is on.
- `progress` (0..1) is the hold-to-type progress. `waiting_release` is true while the current
  letter was just typed: "relax your hands to type it again".
- `ms`: server processing time for this frame.
- `learn` (learn mode only) is `LearnState` plus `"event": "correct"` on the frame a target is
  completed:
  `{"target": "K", "progress": 0.3, "streak": 4, "stats": {"A": {"attempts": 2, "correct": 2, "best_seconds": 1.4}, ...}, "event": null}`
- `record` (record mode only) is `RecordState`:
  `{"letter": "D", "active": true, "saved": 12, "needed": 20, "take": 4, "done": false}`.
  On the frame the take completes, `done` is true and `active` false, and the result also contains
  `"recordings": {...}` (the updated per-letter summary).
