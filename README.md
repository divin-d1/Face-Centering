# Face-Locking

## Face centering extension

This working copy adds a motorized centering path over MQTT. Follow
[`FACE_CENTERING.md`](FACE_CENTERING.md) for broker setup, camera app options,
MQTT message format, and the ESP8266 firmware for the ULN2003 stepper setup.
The original project remains outside this working copy and is unchanged.

A CPU-first face recognition, tracking, and identity-lock demo using face
detection, five-point alignment, ArcFace ONNX embeddings, interactive enrollment,
and threshold-based recognition. Built for the *Face Recognition with ArcFace
ONNX and 5-Point Alignment* practical (Benax Technologies Ltd / Rwanda Coding
Academy).

## Architecture / Pipeline

```
Camera
  -> MediaPipe Full-Range Face Detection (Haar fallback)
  -> MediaPipe FaceMesh 5-Point Landmarks (left eye, right eye, nose tip, left mouth corner, right mouth corner)
  -> Similarity-Transform Alignment (rotation + scale + translation -> 112x112)
  -> ArcFace ONNX Embedding (512-D, L2-normalized)
  -> Enrollment (GUI stores multiple captured templates; CLI averages samples) / Recognition (nearest-template cosine distance + threshold)
  -> Known <name> / Unknown
```

| File | Responsibility |
|---|---|
| `src/app.py` | **Unified live app**: full pipeline (detect + landmarks + align + embed + recognize) in one fullscreen window, with enrollment built in |
| `src/camera.py` | Camera sanity check (open, read, FPS) |
| `src/detect.py` | Haar face bounding-box sanity check |
| `src/landmarks.py` | 5-point landmark sanity check |
| `src/align.py` | Alignment sanity check (112x112 output) |
| `src/embed.py` | ArcFace ONNX embedding + `ArcFaceEmbedderONNX` class |
| `src/haar_5pt.py` | MediaPipe/Haar face proposals + ROI FaceMesh landmarks + 5-point alignment helpers |
| `src/camera_utils.py` | Shared `--camera` CLI argument / device opening helper |
| `src/enroll.py` | Multi-sample enrollment -> averaged, L2-normalized template |
| `src/recognize.py` | Live multi-face recognition against the enrolled database |
| `src/evaluate.py` | Genuine/impostor distance evaluation and threshold suggestion |

## Study Reference

The companion [Face Recognition with ArcFace ONNX and 5-Point Alignment PDF](../FaceRecognitionwithArcFaceONNXand5-PointAlignment.pdf)
is the design reference for this implementation. The most relevant sections
are:

- **Chapter 1:** validate detection, landmarks, alignment, and embeddings as
  separate stages; ArcFace uses an aligned 112×112 face and returns a
  normalized 512-value embedding.
- **Chapter 2:** enroll several samples and L2-normalize the embeddings.
  The standalone `src.enroll` tool follows the PDF by averaging them into one
  template; the GUI keeps each captured pose as a separate template so
  recognition can match the closest pose. Both save aligned crops for
  inspection and evaluation.
- **Chapter 3:** select the distance threshold from genuine and impostor
  comparisons using false-accept and false-reject rates. The app and
  `src.recognize` default to `.50`. On the currently saved enrollment crops,
  `src.evaluate` measured 0% FAR and 0% FRR across 146 impostor and 64 genuine
  comparisons. Treat that result as a small local check, not a guarantee across
  new lighting, distance, cameras, or people; collect more samples and
  re-evaluate when conditions change. Enrollment data is ignored by Git, so a
  fresh clone must enroll people and run the evaluation again to reproduce a
  local threshold.
- **Chapters 4 and 6:** heavy stages need not run every frame; stable identity
  decisions matter more than peak FPS. The app follows the CPU-first advice by
  detecting every three frames and refreshing ArcFace comparisons every 0.30
  seconds. GUI enrollment preserves each captured pose as a template, and
  recognition compares against the nearest template instead of averaging pose
  variation away. For higher assurance, tune the threshold using varied
  genuine and impostor samples.

The app now uses MediaPipe's full-range face detector, with Haar as a fallback,
to improve on the book's original Haar-only face proposals for smaller/farther
faces. FaceMesh still supplies the landmarks used by the same alignment and
ArcFace stages described in the PDF.

## Dependencies

```
opencv-python
numpy
onnxruntime
scipy
tqdm
mediapipe==0.10.21
paho-mqtt==2.1.0
```

### Why Python 3.11 (not whatever `python3` defaults to)

`mediapipe==0.10.21` is pinned because this project uses the legacy
`mediapipe.solutions.face_mesh` API for landmarks. The 0.10.21 release provides
wheels for Python 3.9-3.12, but not Python 3.13. Some later wheels, including
0.10.31, no longer expose `mediapipe.solutions`, so don't upgrade MediaPipe
without migrating the detector to the Tasks API. If your system
Python is 3.13 (`python3 --version`), build an isolated Python 3.11 with
pyenv rather than changing the pinned dependency:

```bash
# One-time: build dependencies (Debian/Kali; needs sudo)
sudo apt-get update && sudo apt-get install -y \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev tk-dev

# Install pyenv and build Python 3.11 locally (no system Python changes)
git clone --depth 1 https://github.com/pyenv/pyenv.git ~/.pyenv
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.11.16
```

If your system already has a Python 3.9-3.12 interpreter, skip pyenv and use
it directly instead.

## Environment Setup

```bash
# Run these commands from the `FaceLocking` working-copy directory.
~/.pyenv/versions/3.11.16/bin/python3.11 -m venv .venv   # or: python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python init_project.py   # idempotent; creates data/, models/, src/ if missing
```

Verify the interpreter and key imports before continuing:

```bash
python --version          # Python 3.11.x
python -c "import cv2, numpy, onnxruntime, scipy, mediapipe, paho.mqtt.client; \
           from mediapipe.solutions import face_mesh; print('all imports OK')"
```

## Model Setup (ArcFace ONNX)

The pipeline needs `models/embedder_arcface.onnx`, the `w600k_r50.onnx`
ResNet-50 ArcFace embedder from InsightFace's `buffalo_l` model pack.

```bash
curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
unzip -o buffalo_l.zip
cp w600k_r50.onnx models/embedder_arcface.onnx
rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
```

**If the SourceForge mirror is slow or unreachable** (observed in some
network environments), download the identical `w600k_r50.onnx` file
directly from a public Hugging Face mirror instead:

```bash
curl -L -o models/embedder_arcface.onnx \
  "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx"
```

Both sources serve the exact same InsightFace `buffalo_l/w600k_r50.onnx`
file (only the ArcFace embedder is used here — the other files in
`buffalo_l.zip`, such as the InsightFace face/landmark detectors, are not
needed because this project uses Haar + MediaPipe for detection/landmarks).

Validate the model loads and produces a correct embedding:

```bash
python -m src.embed --camera /dev/video2
```
Expected: a window opens showing `embedding dim: 512`, `norm(before L2):`
around 15-30, and (after a couple of frames) `cos(prev,this):` close to 1.0
for a stationary face.

## External Camera Setup

List connected cameras and their device nodes:

```bash
v4l2-ctl --list-devices
```

Camera-facing scripts accept `--camera <path-or-index>` (default:
`/dev/video2`, or override via the `FACE_REC_CAMERA` environment variable):

```bash
python -m src.camera --camera /dev/video2
```

OpenCV index `0` is **not guaranteed** to map to a particular `/dev/videoN`
node when multiple cameras are attached — always pass the explicit device
path for the camera you intend to use.

## Quick Start: Unified App

`src/app.py` runs the whole pipeline together in one fullscreen window
(1280x720 by default) with enrollment built in — the fastest way to
actually use the system day to day:

```bash
python -m src.app --camera /dev/video2
```

Use the clickable controls for enrollment, capture, auto-capture, save,
cancel, identity lock, database reload, and quit. The live image scales to the
desktop window even when a camera returns a lower capture resolution. To enroll,
click **Enroll**, type the name in the native name dialog, then use **Capture**
or **Auto Capture**. Auto Capture takes a sample about every half-second and
shows the collected count on its button and in the enrollment banner. Capture
at least five samples, then click **Save** to add them to the database; Auto
Capture only collects samples and does not save them. Click a recognized face
to select its tracked ID, then click **Lock** to pin recognition to that
person; click **Unlock**
to clear the lock. A locked track displays `LOCKED` when the match remains the
same and `IDENTITY MISMATCH` if it is recognized as somebody else. The tracker
assigns a stable `T<n>` ID while the face remains in view and reattaches a lock
if the same recognized identity returns under a new track ID. Keyboard
shortcuts:
`+`/`-` adjust the recognition threshold and `q`/Esc quit.

For each detected face, the app also reports the nose tip's direction and
pixel offset from the frame center, plus heuristic smile/frown/sad/grimace,
eye-state, and blink-count cues. A colored warning appears when no face is
detected, and while locked it warns when the selected identity disappears or
no longer matches. Expression labels are landmark-based demo heuristics, not
a validated emotion-recognition model. Use the provided external HD camera for
enrollment and the live demo; verify the selected `/dev/videoN` in the camera
list before starting.

For CPU responsiveness, the app defaults to face detection every three camera
frames and repeats each track's ArcFace match every 0.30 seconds. Detection and
recognition run in a single background worker; the preview and button events
stay on the responsive UI loop, and the buttons have large click targets with
hover feedback. The preview continues between detections and the current
display FPS appears in the header. To refresh boxes more often at the cost of
speed, pass `--detect-every 1`.

The per-stage scripts below remain available for isolated debugging that
matches the PDF's Chapter 1 validation steps (confirm one stage at a time
if something looks wrong in the unified app).

### Assessment practice: detection, recognition, tracking, identity lock

Use the provided **external HD camera** for both enrollment and the live demo;
keep the same camera and lighting for both so embedding distances stay
comparable. Check detection first with `python -m src.detect --camera /dev/video2`,
then enroll at least two people and use the unified app to demonstrate:

- A face is detected and aligned from the five landmarks.
- An enrolled face gets its name; a non-enrolled face remains `Unknown`.
- The `T<n>` tracking ID stays with a moving face across frames.
- Selecting a known face and locking it keeps the lock tied to that track;
  a changed match reports `IDENTITY MISMATCH`.

The related student project provided as a reference is
[dojdev153/FaceLocking](https://github.com/dojdev153/FaceLocking). This project
brings enrollment, recognition, tracking, and identity lock together in one
GUI, with a persistent track ID for the selected face.

## Validate Each Stage

```bash
python -m src.camera --camera /dev/video2       # q to quit
python -m src.detect --camera /dev/video2        # q to quit
python -m src.landmarks --camera /dev/video2     # q to quit
python -m src.align --camera /dev/video2         # q to quit
python -m src.embed --camera /dev/video2         # q to quit
```

## Enrollment

```bash
python -m src.enroll --camera /dev/video2 --name Divin
```
(Omit `--name` to be prompted interactively.) Position your face in frame,
press `SPACE` to capture a sample (or `a` to toggle auto-capture), collect
at least 5-15 samples across slightly different angles/expressions, then
press `s` to save. Repeat for every person you want the system to know.
Aligned 112x112 crops are saved under `data/enroll/<name>/` for later
evaluation. This standalone CLI stores one averaged, L2-normalized template in
`data/db/face_db.npz` / `data/db/face_db.json`; the unified GUI stores each
captured embedding for nearest-template matching.

## Recognition

```bash
python -m src.recognize --camera /dev/video2
```
`q` quits, `+`/`-` adjust the cosine-distance accept threshold live, `r`
reloads the database from disk. Enrolled faces are labeled with their name
and a green box; unrecognized faces are labeled `Unknown` with a red box. The
unified app also overlays a track ID and supports identity lock.

## Evaluation

```bash
python -m src.evaluate
```
Computes genuine (same-person) vs. impostor (different-person) cosine
distance distributions from the aligned crops saved during enrollment,
prints a threshold sweep (FAR/FRR at each threshold), and suggests a
threshold for a 1% target false-accept rate. Enroll at least two people
with saved crops for a meaningful impostor distribution.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `pip install mediapipe==0.10.21` fails on Python 3.13 | Use the pyenv + Python 3.11 setup above; this pin has no `cp313` wheel. |
| `ImportError: cannot import name 'face_mesh' from 'mediapipe.solutions'` | The installed MediaPipe build does not expose the legacy Solutions API. Reinstall the pinned `mediapipe==0.10.21` under Python 3.9-3.12. |
| Camera not opening | Confirm the device with `v4l2-ctl --list-devices`, pass the correct `--camera /dev/videoN`, and make sure no other app holds the camera. |
| `RuntimeError: No enrollment DB` in recognize | Run `python -m src.enroll` for at least one person first. |
| Flat/incorrect embeddings, meaningless distances | Confirm `models/embedder_arcface.onnx` is the real ArcFace model (~166 MB), not a placeholder; re-run `python -m src.embed` to sanity-check. |
| Impostor distances empty in evaluate | Enroll at least two different people with saved crops. |
| Flickering identity labels | Adjust the threshold with `+`/`-` while observing the reported distance; collect varied, clear enrollment samples under the demo lighting. |

## Expected Output

- `python -m src.embed`: `embedding dim: 512`, `norm(before L2): ~15-30`, unit-norm after L2 normalization.
- `python -m src.enroll`: `data/db/face_db.npz` and `data/db/face_db.json` created/updated with one averaged 512-D template per identity. GUI enrollment stores multiple 512-D templates per identity.
- `python -m src.recognize`: enrolled faces labeled with name + green box + `dist=`/`sim=`; unknown faces labeled `Unknown` + red box.
- `python -m src.evaluate`: genuine distances clustered low, impostor distances clustered high, and a suggested threshold near the observed separation point.

## What Not to Commit

`.gitignore` already excludes:
- `.venv/`, `__pycache__/`, `*.pyc`
- `models/*.onnx` (the ArcFace model — downloaded during setup, not tracked)
- `data/enroll/*` and `data/db/*` (personal biometric images and embeddings)

Only `.gitkeep` placeholders are tracked for `data/enroll/` and `data/db/`
so the expected project structure is preserved without committing anyone's
face data.
