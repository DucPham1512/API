# VSR API

Visual Speech Recognition API — streams webcam frames over WebSocket and returns transcription segments in real time.

## Requirements

- Python 3.10+
- A working webcam

**Linux** — also needs `libGL` and a display server (X11 or Wayland) for OpenCV and PyQt6:

```bash
sudo apt install libgl1 libglib2.0-0
```

**Windows** — PyQt6 and OpenCV bundle everything they need; no extra system packages required.

## Install

```bash
pip install -r requirements.txt
```

`dlib` compiles from source on all platforms. Install the build tools first:

| Platform | Command |
|----------|---------|
| Ubuntu / Debian | `sudo apt install cmake build-essential` |
| macOS | `brew install cmake` |
| Windows | Install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and [CMake](https://cmake.org/download/), then add CMake to `PATH` |

## Download the face landmark model

The MediaPipe face landmarker model (~30 MB) is downloaded automatically on first run. To pre-fetch it manually:

**Linux / macOS**
```bash
python - <<'EOF'
from app.preprocessor.face_localizer import download_model
download_model()
EOF
```

**Windows (Command Prompt)**
```cmd
python -c "from app.preprocessor.face_localizer import download_model; download_model()"
```

The file is saved to `~/.cache/mediapipe/face_landmarker.task` (Linux/macOS) or `%USERPROFILE%\.cache\mediapipe\face_landmarker.task` (Windows).

## Run the API server

**Linux / macOS**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Windows**
```cmd
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Run the webcam stream test client

**Linux / macOS**
```bash
python tests/ws_stream_test.py
```

**Windows**
```cmd
python tests\ws_stream_test.py
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
