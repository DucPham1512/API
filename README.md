# VSR API

Visual Speech Recognition API — streams webcam frames over WebSocket and returns transcription segments in real time.

## Requirements

- Python 3.10+
- A working webcam
- `libGL` / display server (needed by OpenCV and PyQt6 on Linux)

## Install

```bash
pip install -r requirements.txt
```

> **dlib** compiles from source on most systems. You need `cmake` and a C++ compiler:
> ```bash
> # Ubuntu / Debian
> sudo apt install cmake build-essential
> # macOS
> brew install cmake
> ```

## Download the face landmark model

The MediaPipe face landmarker model (~30 MB) is downloaded automatically on first run. To pre-fetch it manually:

```bash
python - <<'EOF'
from app.preprocessor.face_localizer import download_model
download_model()
EOF
```

The file is saved to `~/.cache/mediapipe/face_landmarker.task`.

## Run the API server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run the webcam stream test client

```bash
python tests/ws_stream_test.py
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--camera N` | `0` | Camera device index |
| `--url URL` | `ws://127.0.0.1:8000/vsr/stream` | WebSocket endpoint |

The preview window shows the live camera feed with MediaPipe overlays:

- **Green rectangle** — detected lip bounding box
- **Cyan dots** — lip landmark points
- **Yellow circles** — affine-alignment anchor points
- **Red "No face"** — when no face is detected

Press **Q** to stop.
