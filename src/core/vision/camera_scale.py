from cv2.typing import MatLike
import cv2
import logfire
import numpy as np

class CameraScaleDetector:
    def __init__(
        self,):
        self.minimum_distance = 8.0
        self.alpha = 0.40
        
        self.accum_scale = 1.0
        self.accum_dx = 0.0
        self.accum_dy = 0.0

        self.last_scale = 1.0
        self.last_dx = 0.0
        self.last_dy = 0.0

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
            # mask=self.mask,
        )


    def start(self, first_frame: MatLike):
        self.old_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        h,w = self.old_gray.shape
        self.mask = np.zeros((h,w), dtype=np.uint8)
        self.mask[:, :30] = 1
        self.mask[:, -180:] = 1
        self.old_features = self._detect_features(self.old_gray)
        
    def update(self, frame: MatLike) -> float:
        """
        Procesa UN SOLO FRAME y retorna el movimiento:
        (dx, dy)

        dx > 0 → camara se mueve hacia la derecha
        dy > 0 → camara se mueve hacia abajo
        """

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.old_features is None or len(self.old_features) == 0:
            self.old_features = self._detect_features(frame_gray)
            self.old_gray = frame_gray.copy()
            return self.accum_scale

        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.old_gray,
            frame_gray,
            self.old_features,
            None,  # type: ignore
            **self.lk_params,  # type: ignore
        )

        if new_features is None or status is None:
            self._reset_features(frame_gray)
            return self.accum_scale


        good_new = new_features[status == 1]
        good_old = self.old_features[status == 1]

        dx, dy, dist, scale = self._estimate_motion(
            good_new, good_old
        )

        self.old_gray = frame_gray.copy()
        self.old_features = good_new.reshape(-1, 1, 2).astype(np.float32)

        if dist > self.minimum_distance:
            self.old_features = self._detect_features(frame_gray)

        scale_smooth = (self.alpha * scale) + ((1 - self.alpha) * self.last_scale)
        dx_smooth = (self.alpha * dx) + ((1 - self.alpha) * self.last_dx)
        dy_smooth = (self.alpha * dy) + ((1 - self.alpha) * self.last_dy)

        self.last_scale = scale_smooth
        self.accum_dx += dx_smooth
        self.accum_dy += dy_smooth
        self.accum_scale *= scale_smooth

        self.last_dx, self.last_dy = dx_smooth, dy_smooth

        return self.accum_scale

    def get_current_scale(self) -> float:
        return self.accum_scale

    def _estimate_motion(self, new_features, old_features):
        if len(new_features) < 8 or len(new_features) != len(old_features):
            return 0.0, 0.0, 0.0, 1.0

        deltas = new_features - old_features
        dx = float(np.median(deltas[:, 0]))
        dy = float(np.median(deltas[:, 1]))
        dist = np.sqrt(dx**2 + dy**2)

        M, _ = cv2.estimateAffinePartial2D(
            old_features, new_features, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )

        if M is None:
            scale = 1.0
        else:
            scale = np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2)

        return dx, dy, dist, scale
    
    def _detect_features(self, gray: MatLike):
        return cv2.goodFeaturesToTrack(
            gray, mask=self.mask, **self.features_params # type: ignore
        ) # type: ignore

    def _reset_features(self, gray):
        self.old_features =  self._detect_features(gray)
        self.old_gray = gray.copy()
    

scale_motion_detector = CameraScaleDetector()