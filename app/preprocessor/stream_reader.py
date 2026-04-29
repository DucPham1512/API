import time
from dataclasses import dataclass
from typing import Generator, Tuple, Union

import cv2
import numpy as np


@dataclass
class FrameMetadata:
    frame_id: int
    timestamp_ms: float
    source_fps: float


class VideoStreamReader:
    """Reads frames from a video file or camera device at a fixed target FPS.

    Args:
        source:     Camera index (int) or video file path (str). Default: 0 (built-in webcam).
        target_fps: Target frame rate to enforce.
    """

    def __init__(self, source: Union[int, str] = 0, target_fps: float = 25.0):
        self.source = source
        self.target_fps = target_fps
        self._cap: cv2.VideoCapture = None

    def __enter__(self):
        self._cap = self._open()
        if not self._cap.isOpened():
            raise IOError(
                f"Cannot open video source: {self.source!r}\n"
                "Check that the webcam hardware switch is on and no other app is using the camera."
            )
        return self

    def _open(self) -> cv2.VideoCapture:
        # Try MSMF first for integer indices — avoids triggering NVIDIA Broadcast DSH errors.
        # Fall back to DSHOW, then let OpenCV choose.
        if isinstance(self.source, int):
            for backend in (cv2.CAP_MSMF, cv2.CAP_DSHOW, cv2.CAP_ANY):
                cap = cv2.VideoCapture(self.source, backend)
                if cap.isOpened():
                    return cap
                cap.release()

        # File path: let OpenCV choose the best backend
        return cv2.VideoCapture(self.source)

    def __exit__(self, *_):
        if self._cap is not None:
            self._cap.release()

    def stream(self) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """Yield (frame_bgr, FrameMetadata) at target_fps.

        For video files: samples by video timestamp — works at any read speed.
        For cameras:     samples by wall-clock time — throttles live input.
        """
        if self._cap is None:
            raise RuntimeError("Use VideoStreamReader as a context manager.")

        source_fps = self._cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        frame_interval_s = 1.0 / self.target_fps
        is_file = not isinstance(self.source, int)

        frame_id = 0
        last_yield_s = float("-inf")

        while True:
            ret, frame = self._cap.read()
            if not ret:
                break

            if is_file:
                current_s = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                current_s = time.perf_counter()

            if current_s - last_yield_s < frame_interval_s:
                continue

            timestamp_ms = self._cap.get(cv2.CAP_PROP_POS_MSEC)
            yield frame, FrameMetadata(
                frame_id=frame_id,
                timestamp_ms=timestamp_ms,
                source_fps=source_fps,
            )

            frame_id += 1
            last_yield_s = current_s
