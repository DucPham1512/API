"""
Pipeline verification tests.

Run with:
    python -m pytest tests/test_pipeline.py -v
or directly:
    python tests/test_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ONNX_MODEL = ROOT / "models" / "model.onnx"
VOCAB_SIZE = 5049
WINDOW_FRAMES = 29
H, W = 96, 96


# ---------------------------------------------------------------------------
# OnnxInferencer
# ---------------------------------------------------------------------------

class TestOnnxInferencer:
    @pytest.fixture(scope="class")
    def inferencer(self):
        if not ONNX_MODEL.exists():
            pytest.skip(f"model.onnx not found at {ONNX_MODEL}")
        from app.inferencer.onnx_inferencer import OnnxInferencer
        return OnnxInferencer(ONNX_MODEL)

    def test_loads(self, inferencer):
        assert inferencer is not None

    def test_output_shape(self, inferencer):
        tensor = np.random.randn(WINDOW_FRAMES, H, W, 1).astype(np.float32)
        out = inferencer.predict(tensor)
        assert out.shape == (1, WINDOW_FRAMES, VOCAB_SIZE), (
            f"Expected (1, {WINDOW_FRAMES}, {VOCAB_SIZE}), got {out.shape}"
        )

    def test_output_is_log_probs(self, inferencer):
        """log-softmax output: all values ≤ 0, each frame sums to ~1 in prob space."""
        tensor = np.random.randn(WINDOW_FRAMES, H, W, 1).astype(np.float32)
        out = inferencer.predict(tensor)
        assert (out <= 0).all(), "log-probs must be ≤ 0"
        prob_sums = np.exp(out).sum(axis=-1)
        np.testing.assert_allclose(prob_sums, 1.0, atol=1e-4)

    def test_variable_length_input(self, inferencer):
        """Model should accept any T, not just the export window size."""
        for t in [15, 29, 50]:
            tensor = np.zeros((t, H, W, 1), dtype=np.float32)
            out = inferencer.predict(tensor)
            assert out.shape[0] == 1
            assert out.shape[2] == VOCAB_SIZE, f"Wrong vocab dim for T={t}"

    def test_input_reshape(self):
        """(T,H,W,C) → (1,C,T,H,W) transpose is correct."""
        t = np.arange(WINDOW_FRAMES * H * W, dtype=np.float32).reshape(WINDOW_FRAMES, H, W, 1)
        # simulate the reshape in predict()
        batch = t.transpose(3, 0, 1, 2)[np.newaxis]
        assert batch.shape == (1, 1, WINDOW_FRAMES, H, W)
        # value at (T=0, H=0, W=0) should be preserved
        assert batch[0, 0, 0, 0, 0] == t[0, 0, 0, 0]


# ---------------------------------------------------------------------------
# LipNormalizer
# ---------------------------------------------------------------------------

class TestLipNormalizer:
    def test_output_shape_and_dtype(self):
        from app.preprocessor.normalizer import LipNormalizer
        norm = LipNormalizer()
        frames = np.random.randint(0, 256, (WINDOW_FRAMES, H, W), dtype=np.uint8)
        out = norm.normalize(frames)
        assert out.shape == (WINDOW_FRAMES, H, W, 1)
        assert out.dtype == np.float32

    def test_placeholder_range(self):
        """Without stats, output should be in [-1, 1]."""
        from app.preprocessor.normalizer import LipNormalizer
        norm = LipNormalizer()
        frames = np.random.randint(0, 256, (WINDOW_FRAMES, H, W), dtype=np.uint8)
        out = norm.normalize(frames)
        assert out.min() >= -1.0 - 1e-5
        assert out.max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# End-to-end: normalizer → inferencer
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_normalizer_to_inferencer(self):
        if not ONNX_MODEL.exists():
            pytest.skip(f"model.onnx not found at {ONNX_MODEL}")
        from app.preprocessor.normalizer import LipNormalizer
        from app.inferencer.onnx_inferencer import OnnxInferencer

        frames = np.random.randint(0, 256, (WINDOW_FRAMES, H, W), dtype=np.uint8)
        tensor = LipNormalizer().normalize(frames)   # (T, 96, 96, 1)
        out = OnnxInferencer(ONNX_MODEL).predict(tensor)  # (1, T', 5049)

        assert out.ndim == 3
        assert out.shape[0] == 1
        assert out.shape[2] == VOCAB_SIZE
        print(f"\n  normalizer → inferencer OK  |  output shape: {out.shape}")


if __name__ == "__main__":
    # Quick smoke-test without pytest
    print("=== OnnxInferencer ===")
    if not ONNX_MODEL.exists():
        print(f"SKIP: {ONNX_MODEL} not found")
    else:
        from app.inferencer.onnx_inferencer import OnnxInferencer
        inf = OnnxInferencer(ONNX_MODEL)
        t = np.random.randn(WINDOW_FRAMES, H, W, 1).astype(np.float32)
        out = inf.predict(t)
        print(f"  input  shape : {t.shape}")
        print(f"  output shape : {out.shape}  (expect (1, {WINDOW_FRAMES}, {VOCAB_SIZE}))")
        assert out.shape == (1, WINDOW_FRAMES, VOCAB_SIZE)
        assert (out <= 0).all()
        print("  PASS")

    print("\n=== LipNormalizer ===")
    from app.preprocessor.normalizer import LipNormalizer
    frames = np.random.randint(0, 256, (WINDOW_FRAMES, H, W), dtype=np.uint8)
    out = LipNormalizer().normalize(frames)
    print(f"  output shape : {out.shape}  dtype={out.dtype}  range=[{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == (WINDOW_FRAMES, H, W, 1)
    print("  PASS")

    print("\n=== End-to-end ===")
    if ONNX_MODEL.exists():
        from app.inferencer.onnx_inferencer import OnnxInferencer
        tensor = LipNormalizer().normalize(
            np.random.randint(0, 256, (WINDOW_FRAMES, H, W), dtype=np.uint8)
        )
        out = OnnxInferencer(ONNX_MODEL).predict(tensor)
        print(f"  output shape : {out.shape}")
        print("  PASS")
