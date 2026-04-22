"""Face and lip region detection using the MediaPipe Tasks API (mediapipe >= 0.10).

The face_landmarker.task model file must be present before use.
Run download_model.py in the project root to fetch it automatically.
"""
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

try:
    import dlib
    _DLIB_AVAILABLE = True
except ImportError:
    _DLIB_AVAILABLE = False

# Store model in user home dir to avoid spaces/Unicode in path (MediaPipe C++ limitation)
_DEFAULT_MODEL_PATH = Path.home() / ".cache" / "mediapipe" / "face_landmarker.task"

# MediaPipe FaceMesh lip landmark indices (same across old and new API)
_OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
               308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]
_INNER_LIP = [191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
               324, 318, 402, 317, 14, 87, 178, 88, 95]
_ALL_LIP_INDICES = list(dict.fromkeys(_OUTER_LIP + _INNER_LIP))

# Four stable anchor points used by LipROICropper for affine alignment
ALIGNMENT_ANCHORS = [61, 291, 0, 17]  # left corner, right corner, nasal base, chin tip

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def download_model(dest: Path = _DEFAULT_MODEL_PATH) -> Path:
    """Download the face_landmarker.task model file if not already present."""
    if dest.exists():
        return dest
    print(f"Downloading face landmark model to {dest} ...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(_MODEL_URL, dest)
    print("Download complete.")
    return dest


@dataclass
class LipBBox:
    x: float
    y: float
    w: float
    h: float
    confidence: float

    def is_valid(self) -> bool:
        return self.w > 0 and self.h > 0 and self.confidence > 0


@dataclass
class LocalizerResult:
    bbox: LipBBox
    landmarks: np.ndarray         # (N, 2) float32, pixel coords of all lip points
    anchor_landmarks: np.ndarray  # (4, 2) float32, pixel coords of alignment anchors
    is_valid: bool


class FaceLocalizer:
    """Detects lip region using MediaPipe FaceLandmarker (Tasks API).

    Requires the face_landmarker.task model file. Call download_model() once
    before first use, or pass model_path explicitly.

    Usage:
        download_model()
        with FaceLocalizer() as loc:
            result = loc.locate(frame_bgr)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.75,
        padding: float = 0.20,
        model_path: Optional[Path] = None,
        dlib_model_path: Optional[str] = None,
        max_invalid_frames: int = 3,
    ):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe is required: pip install mediapipe")

        self.confidence_threshold = confidence_threshold
        self.padding = padding
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL_PATH
        self.dlib_model_path = dlib_model_path
        self.max_invalid_frames = max_invalid_frames

        self._landmarker: Optional[mp_vision.FaceLandmarker] = None
        self._dlib_detector = None
        self._dlib_predictor = None
        self._last_valid_result: Optional[LocalizerResult] = None
        self._consecutive_invalid: int = 0
        self._frame_timestamp_ms: int = 0

    def __enter__(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Run: from vsr_preprocessor.face_localizer import download_model; download_model()"
            )

        base_options = mp_tasks.BaseOptions(model_asset_path=str(self.model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

        if _DLIB_AVAILABLE and self.dlib_model_path:
            self._dlib_detector = dlib.get_frontal_face_detector()
            self._dlib_predictor = dlib.shape_predictor(self.dlib_model_path)

        return self

    def __exit__(self, *_):
        if self._landmarker is not None:
            self._landmarker.close()

    def locate(self, frame_bgr: np.ndarray, timestamp_ms: Optional[int] = None) -> Optional[LocalizerResult]:
        """Return LocalizerResult for the lip region, or None if detection fails."""
        h, w = frame_bgr.shape[:2]

        if timestamp_ms is None:
            self._frame_timestamp_ms += 40  # assume 25 fps → 40 ms/frame
            timestamp_ms = self._frame_timestamp_ms

        result = self._try_mediapipe(frame_bgr, w, h, timestamp_ms)

        if result is None or not result.bbox.is_valid():
            result = self._try_dlib_fallback(frame_bgr, w, h)

        if result is None or not result.bbox.is_valid():
            self._consecutive_invalid += 1
            if self._consecutive_invalid <= self.max_invalid_frames and self._last_valid_result is not None:
                return self._last_valid_result
            return None

        self._consecutive_invalid = 0
        self._last_valid_result = result
        return result

    def _try_mediapipe(self, frame_bgr: np.ndarray, w: int, h: int, timestamp_ms: int) -> Optional[LocalizerResult]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        detection = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not detection.face_landmarks:
            return None

        # Landmarks are normalized [0, 1] — convert to pixel coords
        raw = detection.face_landmarks[0]
        all_pts = np.array([[lm.x * w, lm.y * h] for lm in raw], dtype=np.float32)

        lip_pts = all_pts[_ALL_LIP_INDICES]
        anchor_pts = all_pts[ALIGNMENT_ANCHORS]

        # Tasks API does not expose a per-face confidence score; use a fixed value
        bbox = self._bbox_from_points(lip_pts, w, h, confidence=1.0)
        return LocalizerResult(bbox=bbox, landmarks=lip_pts, anchor_landmarks=anchor_pts, is_valid=True)

    def _try_dlib_fallback(self, frame_bgr: np.ndarray, w: int, h: int) -> Optional[LocalizerResult]:
        if self._dlib_detector is None:
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._dlib_detector(gray, 1)
        if not faces:
            return None

        shape = self._dlib_predictor(gray, faces[0])
        lip_pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(48, 68)], dtype=np.float32)
        bbox = self._bbox_from_points(lip_pts, w, h, confidence=0.60)
        anchor_pts = np.array([
            [shape.part(48).x, shape.part(48).y],
            [shape.part(54).x, shape.part(54).y],
            [shape.part(33).x, shape.part(33).y],
            [shape.part(8).x,  shape.part(8).y],
        ], dtype=np.float32)

        return LocalizerResult(bbox=bbox, landmarks=lip_pts, anchor_landmarks=anchor_pts, is_valid=True)

    def _bbox_from_points(self, points: np.ndarray, frame_w: int, frame_h: int, confidence: float) -> LipBBox:
        x_min, y_min = points[:, 0].min(), points[:, 1].min()
        x_max, y_max = points[:, 0].max(), points[:, 1].max()

        pad_x = (x_max - x_min) * self.padding
        pad_y = (y_max - y_min) * self.padding

        x = max(0.0, x_min - pad_x)
        y = max(0.0, y_min - pad_y)
        x2 = min(float(frame_w), x_max + pad_x)
        y2 = min(float(frame_h), y_max + pad_y)

        return LipBBox(x=x, y=y, w=x2 - x, h=y2 - y, confidence=confidence)
