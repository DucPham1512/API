from typing import Tuple

import cv2
import numpy as np

from .face_localizer import LipBBox, LocalizerResult

# Fixed target positions of the 4 alignment anchors within the 96×96 output crop.
# Order: left corner, right corner, top center (nasal base), bottom center (chin tip)
_TARGET_ANCHORS = np.array([
    [16, 48],
    [80, 48],
    [48, 20],
    [48, 76],
], dtype=np.float32)

OUTPUT_SIZE = (96, 96)


class LipROICropper:
    """Crops and affine-aligns the lip region to a fixed OUTPUT_SIZE patch.

    Affine alignment removes head rotation and scale variation between frames
    so the temporal model sees only lip movement, not head movement.
    """

    def __init__(self, output_size: Tuple[int, int] = OUTPUT_SIZE, grayscale: bool = True):
        self.output_size = output_size
        self.grayscale = grayscale

    def crop(self, frame_bgr: np.ndarray, result: LocalizerResult) -> np.ndarray:
        """Return a single aligned crop (H, W) uint8 (grayscale) or (H, W, 3) uint8 (BGR).

        Falls back to a plain bbox resize if affine estimation fails.
        """
        crop = self._affine_crop(frame_bgr, result.anchor_landmarks)
        if crop is None:
            crop = self._bbox_crop(frame_bgr, result.bbox)

        if self.grayscale:
            if crop.ndim == 3:
                # BT.601 perceptual luminance — NOT a simple average
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # shape: (H, W)
        return crop

    def _affine_crop(self, frame_bgr: np.ndarray, anchor_landmarks: np.ndarray) -> np.ndarray | None:
        if anchor_landmarks is None or len(anchor_landmarks) < 4:
            return None

        src_pts = anchor_landmarks.astype(np.float32)
        dst_pts = _TARGET_ANCHORS.copy()

        # estimateAffinePartial2D: rotation + isotropic scale only (no shear)
        M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.LMEDS)
        if M is None:
            return None

        h, w = self.output_size
        try:
            warped = cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
        except cv2.error:
            return None

        return warped

    def _bbox_crop(self, frame_bgr: np.ndarray, bbox: LipBBox) -> np.ndarray:
        fh, fw = frame_bgr.shape[:2]
        x1 = int(max(0, bbox.x))
        y1 = int(max(0, bbox.y))
        x2 = int(min(fw, bbox.x + bbox.w))
        y2 = int(min(fh, bbox.y + bbox.h))

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            # Blank frame if bbox is degenerate
            c = 1 if self.grayscale else 3
            return np.zeros((*self.output_size, c), dtype=np.uint8)

        h, w = self.output_size
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
