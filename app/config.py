from pathlib import Path

BASE_DIR = Path(__file__).parent
ROOT_DIR = BASE_DIR.parent

MODEL_PATH = ROOT_DIR / "models" / "model.onnx"
# SentencePiece .model file (preferred). Drop spm.model into models/ when available.
SPM_PATH = ROOT_DIR / "models" / "spm.model"
# Fallback vocabulary: ESPnet units.txt with "token id" lines (1-indexed, 0=blank).
VOCAB_PATH = ROOT_DIR / "models" / "unigram5000_units.txt"

TARGET_FPS: float = 25.0
WINDOW_FRAMES: int = 29
STRIDE: int = 10
GRAYSCALE: bool = True
