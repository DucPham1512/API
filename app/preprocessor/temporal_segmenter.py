from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class SegmentWindow:
    frames: np.ndarray        # (T, H, W) or (T, H, W, C) uint8
    start_frame_id: int
    end_frame_id: int


class TemporalSegmenter:
    """Buffers lip crops and emits fixed-length windows with a sliding stride.

    A ring buffer of `window_frames` crops is maintained.  Every `stride`
    new frames, a snapshot of the buffer is emitted as a SegmentWindow.

    Args:
        window_frames: Number of frames per window (default 29 ≈ 1.16 s @ 25 fps).
        stride:        New frames between consecutive windows (default 10, ~65% overlap).
    """

    def __init__(self, window_frames: int = 29, stride: int = 10):
        self.window_frames = window_frames
        self.stride = stride

        self._buffer: deque = deque(maxlen=window_frames)
        self._frame_ids: deque = deque(maxlen=window_frames)
        self._frames_since_emit: int = 0

    def push(self, crop: np.ndarray, frame_id: int) -> Optional[SegmentWindow]:
        """Add one crop to the buffer; return a SegmentWindow when ready, else None."""
        self._buffer.append(crop)
        self._frame_ids.append(frame_id)
        self._frames_since_emit += 1

        if len(self._buffer) == self.window_frames and self._frames_since_emit >= self.stride:
            self._frames_since_emit = 0
            frames = np.stack(list(self._buffer), axis=0)  # (T, H, W[, C])
            return SegmentWindow(
                frames=frames,
                start_frame_id=self._frame_ids[0],
                end_frame_id=self._frame_ids[-1],
            )

        return None

    def reset(self):
        """Clear the buffer — call this on stream interruption."""
        self._buffer.clear()
        self._frame_ids.clear()
        self._frames_since_emit = 0
