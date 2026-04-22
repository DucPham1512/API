from pathlib import Path
from typing import Optional, Union

import numpy as np

from .temporal_segmenter import SegmentWindow


class LipNormalizer:
    """Converts a uint8 window to a normalized float32 tensor.

    Normalization modes (in priority order):
      1. Z-score using per-channel mean/std loaded from a stats .npz file
         (provided by the model builder after training).
      2. Placeholder [-1, 1] linear rescale when no stats file is available.

    Output contract:
        shape  : (T, H, W, 1)  — channel dim always present for consistency
        dtype  : float32
        values : ~[-3, 3] with z-score stats, [-1, 1] with placeholder
    """

    def __init__(self, stats_path: Optional[Union[str, Path]] = None):
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        if stats_path is not None:
            self._load_stats(stats_path)

    def _load_stats(self, path: Union[str, Path]):
        data = np.load(str(path))
        self._mean = data["mean"].astype(np.float32)
        self._std = data["std"].astype(np.float32)

    def normalize(self, window: Union[SegmentWindow, np.ndarray]) -> np.ndarray:
        """Return a float32 tensor of shape (T, H, W, 1).

        Accepts either a SegmentWindow or a raw numpy array (T, H, W) or (T, H, W, C).
        """
        frames = window.frames if isinstance(window, SegmentWindow) else window

        x = frames.astype(np.float32) / 255.0  # [0, 1]

        # Ensure shape is (T, H, W, C)
        if x.ndim == 3:
            x = x[..., np.newaxis]  # (T, H, W) → (T, H, W, 1)

        if self._mean is not None and self._std is not None:
            x = (x - self._mean) / (self._std + 1e-6)
        else:
            # Placeholder: [0, 1] → [-1, 1]
            x = x * 2.0 - 1.0

        return x.astype(np.float32)
