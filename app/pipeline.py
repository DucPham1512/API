"""Combines preprocessing, ONNX inference, and CTC decoding into one callable."""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

import cv2
import numpy as np

from app import config
from app.inferencer.onnx_inferencer import OnnxInferencer
from app.interface.schemas import StreamSegment, TranscriptSegment
from app.preprocessor.face_localizer import FaceLocalizer, download_model
from app.preprocessor.lip_cropper import LipROICropper
from app.preprocessor.normalizer import LipNormalizer
from app.preprocessor.pipeline import VSRPreprocessingPipeline
from app.preprocessor.temporal_segmenter import SegmentWindow, TemporalSegmenter


# ---------------------------------------------------------------------------
# CTC greedy decoder
# ---------------------------------------------------------------------------

def _load_vocab(vocab_path) -> dict[int, str] | None:
    """Load an ESPnet units.txt file (format: 'token id' per line) into an id→token dict."""
    if not vocab_path:
        return None
    from pathlib import Path
    p = Path(vocab_path)
    if not p.exists():
        return None
    vocab: dict[int, str] = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rsplit(" ", 1)
            if len(parts) == 2:
                token, idx = parts
                vocab[int(idx)] = token
    return vocab or None


def _ctc_greedy_decode(log_probs: np.ndarray, spm=None, vocab: dict | None = None) -> str:
    """Greedy CTC decode: argmax → collapse consecutive duplicates → strip blank.

    Args:
        log_probs: float32 array (1, T', vocab_size)
        spm:       SentencePieceProcessor (preferred)
        vocab:     id→token dict built from units.txt (fallback)

    Returns:
        Decoded text, or empty string if nothing recognised.
    """
    ids = np.argmax(log_probs[0], axis=-1)          # (T',)
    collapsed = [int(ids[0])]
    for i in range(1, len(ids)):
        if ids[i] != collapsed[-1]:
            collapsed.append(int(ids[i]))
    tokens = [t for t in collapsed if t != 0]        # 0 = CTC blank
    if not tokens:
        return ""

    if spm is not None:
        return spm.DecodeIds(tokens)

    if vocab is not None:
        # SentencePiece-style: ▁ marks start of a new word
        pieces = [vocab.get(t, f"<{t}>") for t in tokens]
        text = "".join(pieces).replace("▁", " ").strip()
        return text

    return " ".join(str(t) for t in tokens)


# ---------------------------------------------------------------------------
# Shared pipeline singleton
# ---------------------------------------------------------------------------

class VSRPipeline:
    """Wraps OnnxInferencer + CTC decoder.  Loaded once, reused across requests."""

    _instance: Optional[VSRPipeline] = None

    def __init__(self):
        self._inferencer = OnnxInferencer(config.MODEL_PATH)
        self._spm = _load_spm(config.SPM_PATH)
        self._vocab = _load_vocab(config.VOCAB_PATH) if not self._spm else None
        if self._spm:
            log.info("decoder: SentencePiece model loaded")
        elif self._vocab:
            log.info("decoder: units.txt vocabulary loaded (%d tokens)", len(self._vocab))
        else:
            log.warning("decoder: no vocabulary found — output will be raw token IDs")

    @classmethod
    def get(cls) -> VSRPipeline:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # File mode
    # ------------------------------------------------------------------

    def process_file(self, video_path: str) -> list[TranscriptSegment]:
        """Run full preprocessing + inference on an MP4 file (blocking)."""
        segments: list[TranscriptSegment] = []
        stats = {"frames_read": 0, "faces_detected": 0, "windows_emitted": 0, "segments_kept": 0}

        def on_window(tensor: np.ndarray, window: SegmentWindow) -> None:
            stats["windows_emitted"] += 1
            seg = self._infer_window(tensor, window)
            if seg:
                stats["segments_kept"] += 1
                log.info("segment: [%.0f–%.0f ms] %r", seg.start_ms, seg.end_ms, seg.text)
                segments.append(seg)
            else:
                log.debug("window %.0f–%.0f ms → CTC returned empty string",
                          window.start_frame_id / config.TARGET_FPS * 1000,
                          window.end_frame_id / config.TARGET_FPS * 1000)

        # Wrap the face localizer to count detections per frame
        from app.preprocessor.face_localizer import FaceLocalizer, download_model
        from app.preprocessor.lip_cropper import LipROICropper
        from app.preprocessor.normalizer import LipNormalizer
        from app.preprocessor.stream_reader import VideoStreamReader
        from app.preprocessor.temporal_segmenter import TemporalSegmenter

        download_model()
        reader = VideoStreamReader(source=video_path, target_fps=config.TARGET_FPS)
        cropper = LipROICropper(grayscale=config.GRAYSCALE)
        segmenter = TemporalSegmenter(window_frames=config.WINDOW_FRAMES, stride=config.STRIDE)
        normalizer = LipNormalizer()

        with reader, FaceLocalizer() as localizer:
            for frame_bgr, meta in reader.stream():
                stats["frames_read"] += 1
                result = localizer.locate(frame_bgr, timestamp_ms=int(meta.timestamp_ms))
                if result is None:
                    continue
                stats["faces_detected"] += 1
                crop = cropper.crop(frame_bgr, result)
                window = segmenter.push(crop, meta.frame_id)
                if window is not None:
                    tensor = normalizer.normalize(window)
                    on_window(tensor, window)

        log.info(
            "process_file done | frames=%d  faces=%d  windows=%d  segments=%d",
            stats["frames_read"], stats["faces_detected"],
            stats["windows_emitted"], stats["segments_kept"],
        )
        return segments

    # ------------------------------------------------------------------
    # Stream mode
    # ------------------------------------------------------------------

    def make_stream_processor(self) -> StreamProcessor:
        """Return a new per-connection processor sharing this inferencer."""
        return StreamProcessor(self._inferencer, self._spm, self._vocab)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _infer_window(
        self, tensor: np.ndarray, window: SegmentWindow
    ) -> Optional[TranscriptSegment]:
        log_probs = self._inferencer.predict(tensor)
        text = _ctc_greedy_decode(log_probs, self._spm, self._vocab)
        if not text:
            return None
        fps = config.TARGET_FPS
        return TranscriptSegment(
            text=text,
            start_ms=round(window.start_frame_id / fps * 1000, 1),
            end_ms=round(window.end_frame_id / fps * 1000, 1),
        )


# ---------------------------------------------------------------------------
# Per-connection WebSocket stream processor
# ---------------------------------------------------------------------------

class StreamProcessor:
    """Stateful processor for one WebSocket connection.

    Lifecycle:
        processor = VSRPipeline.get().make_stream_processor()
        try:
            segments = processor.push_frame(jpeg_bytes)
            ...
            final   = processor.flush()
        finally:
            processor.close()
    """

    def __init__(self, inferencer: OnnxInferencer, spm=None, vocab=None):
        self._inferencer = inferencer
        self._spm = spm
        self._vocab = vocab
        self._fps: float = config.TARGET_FPS
        self._frame_id: int = 0

        download_model()
        self._localizer_cm = FaceLocalizer()
        self._localizer = self._localizer_cm.__enter__()
        self._cropper = LipROICropper(grayscale=config.GRAYSCALE)
        self._segmenter = TemporalSegmenter(
            window_frames=config.WINDOW_FRAMES,
            stride=config.STRIDE,
        )
        self._normalizer = LipNormalizer()

    def configure(self, fps: float) -> None:
        self._fps = fps

    def push_frame(self, jpeg_bytes: bytes) -> list[StreamSegment]:
        """Decode one JPEG frame and return any newly completed segments."""
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame_bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return []

        ts_ms = int(self._frame_id / self._fps * 1000)
        result = self._localizer.locate(frame_bgr, timestamp_ms=ts_ms)
        self._frame_id += 1

        if result is None or not result.is_valid:
            return []

        crop = self._cropper.crop(frame_bgr, result)
        window = self._segmenter.push(crop, self._frame_id)
        if window is None:
            return []

        tensor = self._normalizer.normalize(window)
        log_probs = self._inferencer.predict(tensor)
        text = _ctc_greedy_decode(log_probs, self._spm, self._vocab)
        if not text:
            return []

        fps = self._fps
        return [StreamSegment(
            text=text,
            start_ms=round(window.start_frame_id / fps * 1000, 1),
            end_ms=round(window.end_frame_id / fps * 1000, 1),
        )]

    def flush(self) -> list[StreamSegment]:
        """Return segments from any remaining buffered frames."""
        # The segmenter only emits full windows; a partial tail is discarded.
        return []

    def close(self) -> None:
        self._localizer_cm.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_spm(spm_path):
    if not spm_path:
        return None
    from pathlib import Path
    if not Path(spm_path).exists():
        return None
    try:
        import sentencepiece as spm_lib
        processor = spm_lib.SentencePieceProcessor()
        processor.Load(str(spm_path))
        return processor
    except Exception:
        return None
