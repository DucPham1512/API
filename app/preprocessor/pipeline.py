"""
Main pipeline entry point.

Wires all stages together:
    VideoStreamReader → FaceLocalizer → LipROICropper → TemporalSegmenter → LipNormalizer

Usage — file mode:
    python -m vsr_preprocessor.pipeline --source path/to/video.mp4

Usage — live camera mode:
    python -m vsr_preprocessor.pipeline --source 0

Output: prints tensor shape and value range for each emitted window.
Replace the callback in run() with your model's inference call.
"""

import argparse
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from .face_localizer import FaceLocalizer, download_model
from .lip_cropper import LipROICropper
from .normalizer import LipNormalizer
from .stream_reader import VideoStreamReader
from .temporal_segmenter import SegmentWindow, TemporalSegmenter


class VSRPreprocessingPipeline:
    """End-to-end inference preprocessing pipeline.

    Args:
        source:       Camera index (int) or video file path (str/Path).
        target_fps:   Target frame rate. Must match what the model was trained on.
        window_frames: Frames per window. Coordinate with model builder.
        stride:       Stride between windows. Coordinate with model builder.
        grayscale:    True = single-channel output (recommended for most VSR models).
        stats_path:   Path to normalization stats .npz from the model builder.
                      If None, uses [-1, 1] placeholder normalization.
        dlib_model_path: Path to dlib shape_predictor_68_face_landmarks.dat (optional).
        on_window:    Callback called with each (tensor, window) pair.
                      tensor shape: (T, 96, 96, 1) float32.
                      If None, tensors are collected and returned by run().
    """

    def __init__(
        self,
        source: Union[int, str, Path],
        target_fps: float = 25.0,
        window_frames: int = 29,
        stride: int = 10,
        grayscale: bool = True,
        stats_path: Optional[Union[str, Path]] = None,
        dlib_model_path: Optional[str] = None,
        on_window: Optional[Callable[[np.ndarray, SegmentWindow], None]] = None,
    ):
        self.source = source
        self.target_fps = target_fps
        self.window_frames = window_frames
        self.stride = stride
        self.grayscale = grayscale
        self.stats_path = stats_path
        self.dlib_model_path = dlib_model_path
        self.on_window = on_window

    def run(self) -> list[np.ndarray]:
        """Process the full source. Returns collected tensors if no on_window callback."""
        download_model()  # no-op if model file already exists

        collected = []

        reader = VideoStreamReader(source=self.source, target_fps=self.target_fps)
        cropper = LipROICropper(grayscale=self.grayscale)
        segmenter = TemporalSegmenter(window_frames=self.window_frames, stride=self.stride)
        normalizer = LipNormalizer(stats_path=self.stats_path)

        with reader, FaceLocalizer(dlib_model_path=self.dlib_model_path) as localizer:
            for frame_bgr, meta in reader.stream():
                result = localizer.locate(frame_bgr, timestamp_ms=int(meta.timestamp_ms))
                if result is None:
                    continue

                crop = cropper.crop(frame_bgr, result)
                window = segmenter.push(crop, meta.frame_id)

                if window is not None:
                    tensor = normalizer.normalize(window)

                    if self.on_window is not None:
                        self.on_window(tensor, window)
                    else:
                        collected.append(tensor)

        return collected


def _default_callback(tensor: np.ndarray, window: SegmentWindow):
    lo, hi = tensor.min(), tensor.max()
    print(
        f"Window frames {window.start_frame_id}–{window.end_frame_id} | "
        f"shape={tensor.shape} dtype={tensor.dtype} | "
        f"range=[{lo:.3f}, {hi:.3f}]"
    )


def main():
    parser = argparse.ArgumentParser(description="VSR Inference Preprocessing Pipeline")
    parser.add_argument("--source", default="0",
                        help="Camera index (default: 0) or video file path")
    parser.add_argument("--fps", type=float, default=25.0, help="Target FPS (default: 25)")
    parser.add_argument("--window", type=int, default=29, help="Frames per window (default: 29)")
    parser.add_argument("--stride", type=int, default=10, help="Window stride (default: 10)")
    parser.add_argument("--rgb", action="store_true", help="Use RGB instead of grayscale")
    parser.add_argument("--stats", default=None, help="Path to normalization stats .npz")
    parser.add_argument("--dlib-model", default=None,
                        help="Path to dlib shape_predictor_68_face_landmarks.dat")
    args = parser.parse_args()

    source: Union[int, str] = int(args.source) if args.source.isdigit() else args.source

    pipeline = VSRPreprocessingPipeline(
        source=source,
        target_fps=args.fps,
        window_frames=args.window,
        stride=args.stride,
        grayscale=not args.rgb,
        stats_path=args.stats,
        dlib_model_path=args.dlib_model,
        on_window=_default_callback,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
