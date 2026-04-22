from pathlib import Path

BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "models" / "model.onnx"

TARGET_FPS: float = 25.0
WINDOW_FRAMES: int = 29
STRIDE: int = 10
GRAYSCALE: bool = True
