# Drowsiness Detection

Real-time driver drowsiness detection using eye-aspect-ratio (EAR), mouth-aspect-ratio (MAR / yawn), and PERCLOS, computed on [MediaPipe Face Mesh](https://developers.google.com/mediapipe) landmarks. Works on a webcam feed or an uploaded video, from the CLI or a Streamlit dashboard.

## What changed from the original version

- Replaced `face_recognition`/dlib with MediaPipe Face Mesh — no `cmake`/compiled-dlib install step, faster CPU inference.
- Replaced the raw notebook script with a proper package (`src/drowsiness/`) + CLI + tests.
- Alarm now has a cooldown instead of spawning a new sound thread every frame.
- Added PERCLOS (percentage of eye closure over a rolling time window) instead of an unbounded score counter.
- Added per-user threshold calibration (`--calibrate-seconds`) instead of fixed EAR/MAR constants.
- Added a real Streamlit dashboard (`app.py`) — the original README described one but the script never had it.
- Added unit tests + CI.

## Install

```bash
pip install -r requirements.txt
```

## Run — CLI

```bash
# Webcam
python -m drowsiness.cli --source 0

# Video file
python -m drowsiness.cli --source path/to/video.mp4

# Headless (server / CI, no cv2 window)
python -m drowsiness.cli --source path/to/video.mp4 --no-display
```

On startup it calibrates EAR/MAR thresholds from a few seconds of baseline footage (override with `--ear-threshold`/`--mar-threshold` to skip that).

## Run — dashboard

```bash
streamlit run app.py
```

Upload a video or switch to webcam, tune thresholds live in the sidebar, and watch a running PERCLOS chart.

## Tests

```bash
pytest tests/ -v
```

Geometry, calibration, and alarm-cooldown logic are tested without needing OpenCV/MediaPipe installed.

## Alarm sound

Drop a `.wav` file at `assets/alarm.wav` (or point `--alarm-file` / the dashboard's "Alarm sound path" field elsewhere).

## Possible next steps


- Multi-face support for driver + passenger monitoring.
- Export a session log (CSV/JSON) of PERCLOS over time instead of just a console summary.
- Package a Dockerfile for one-command deployment.
- Swap `playsound` for a cross-platform, non-blocking audio backend (e.g. `simpleaudio`) if `playsound` proves flaky on Windows.
