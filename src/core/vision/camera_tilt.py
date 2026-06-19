from cv2.typing import MatLike
import cv2
import numpy as np


class CameraTiltDetector:
    """
    Detects the rotational tilt (in-plane rotation) of the camera in a video stream.

    The accumulated tilt angle is normalized and clamped to the range [-5, 5],
    where -5 represents the minimum (max counter-clockwise) and +5 the maximum
    (max clockwise) tilt.

    Positive values → camera tilted clockwise.
    Negative values → camera tilted counter-clockwise.
    """

    # Normalization bounds (raw degrees)
    RAW_MIN_DEG = -30.0
    RAW_MAX_DEG = 30.0

    # Output bounds
    OUT_MIN = -5.0
    OUT_MAX = 5.0

    def __init__(self):
        self.minimum_distance = 8.0
        self.alpha = 0.40

        self.accum_angle_deg = 0.0
        self.last_angle_deg = 0.0

        self.old_gray: MatLike | None = None
        self.old_features = None
        self.mask: np.ndarray | None = None

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )

        self.features_params = dict(
            maxCorners=150,
            qualityLevel=0.3,
            minDistance=8,
            blockSize=7,
        )

    def start(self, first_frame: MatLike) -> None:
        """Initialize the detector with the very first frame of the video."""
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        h, w = self.old_gray.shape
        self.mask = np.zeros((h, w), dtype=np.uint8)
        # Use side strips — same strategy as CameraScaleDetector — so that
        # features spread across the frame give a better rotation estimate.
        self.mask[:, :30] = 1
        self.mask[:, -180:] = 1

        self.old_features = self._detect_features(self.old_gray)

    def update(self, frame: MatLike) -> float:
        """
        Process a single frame and return the normalized tilt value in [-5, 5].

        Positive → clockwise rotation accumulated since start().
        Negative → counter-clockwise rotation accumulated since start().
        """
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_features is None or len(self.old_features) == 0:
            self.old_features = self._detect_features(frame_gray)
            self.old_gray = frame_gray.copy()
            return self._normalize(self.accum_angle_deg)

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray, # type: ignore
            frame_gray,
            self.old_features,
            None,  # type: ignore
            **self.lk_params,  # type: ignore
        )

        if new_features is None or status is None:
            self._reset_features(frame_gray)
            return self._normalize(self.accum_angle_deg)

        good_new = new_features[status == 1]
        good_old = self.old_features[status == 1]

        angle_deg, dist = self._estimate_tilt(good_new, good_old)

        self.old_gray = frame_gray.copy()
        self.old_features = good_new.reshape(-1, 1, 2).astype(np.float32)

        if dist > self.minimum_distance:
            self.old_features = self._detect_features(frame_gray)

        angle_smooth = (self.alpha * angle_deg) + ((1 - self.alpha) * self.last_angle_deg)
        self.last_angle_deg = angle_smooth
        self.accum_angle_deg += angle_smooth

        return self._normalize(self.accum_angle_deg)

    def get_current_tilt(self) -> float:
        """Return the current normalized tilt value in [-5, 5] without advancing the state."""
        return self._normalize(self.accum_angle_deg)

    def get_current_angle_deg(self) -> float:
        """Return the raw accumulated rotation angle in degrees (unbounded)."""
        return self.accum_angle_deg

    def _estimate_tilt(self, new_features: np.ndarray, old_features: np.ndarray):
        """
        Estimate the in-plane rotation angle (degrees) between two feature sets.

        Returns (angle_deg, median_dist) where angle_deg is positive for
        clockwise rotation.
        """
        if len(new_features) < 8 or len(new_features) != len(old_features):
            return 0.0, 0.0

        deltas = new_features - old_features
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        dist = float(np.sqrt(dx**2 + dy**2))

        M, _ = cv2.estimateAffinePartial2D(
            old_features,
            new_features,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
        )

        if M is None:
            return 0.0, dist

        # M = [[cos θ, -sin θ, tx],
        #      [sin θ,  cos θ, ty]]
        angle_rad = float(np.arctan2(M[1, 0], M[0, 0]))
        angle_deg = float(np.degrees(angle_rad))

        return angle_deg, dist

    def _normalize(self, angle_deg: float) -> float:
        """
        Map the raw accumulated angle from [RAW_MIN_DEG, RAW_MAX_DEG] to
        [OUT_MIN, OUT_MAX] and clamp the result.
        """
        t = (angle_deg - self.RAW_MIN_DEG) / (self.RAW_MAX_DEG - self.RAW_MIN_DEG)
        value = self.OUT_MIN + t * (self.OUT_MAX - self.OUT_MIN)
        return float(np.clip(value, self.OUT_MIN, self.OUT_MAX))

    def _detect_features(self, gray: MatLike):
        return cv2.goodFeaturesToTrack(
            gray, mask=self.mask, **self.features_params  # type: ignore
        )  # type: ignore

    def _reset_features(self, gray: MatLike) -> None:
        self.old_features = self._detect_features(gray)
        self.old_gray = gray.copy()


tilt_detector = CameraTiltDetector()