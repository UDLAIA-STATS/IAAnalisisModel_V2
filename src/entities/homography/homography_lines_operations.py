from typing import Optional

import cv2
import numpy as np


class HomographyLinesOperation:
    def __init__(
        self,
        canny_low: int,
        canny_high: int,
        hough_threshold: int,
        hough_min_line_length: int,
        hough_max_line_gap: int,
    ):
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap

    def _preprocess_lines(self, frame: np.ndarray) -> np.ndarray:
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

        white=cv2.inRange(
            hsv,
            (0,0,160),  # type: ignore
            (180,80,255),  # type: ignore
        ) # type: ignore

        kernel=np.ones((5,5),np.uint8)

        white=cv2.morphologyEx(
            white,
            cv2.MORPH_CLOSE,
            kernel,
        )

        white=cv2.morphologyEx(
            white,
            cv2.MORPH_OPEN,
            kernel,
        )

        return white

    def _detect_lines(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect white line segments on the pitch.

        Returns
        -------
        np.ndarray of shape (N, 4) — each row is (x1, y1, x2, y2).
        """
        frame = self._preprocess_lines(frame)
        edges = cv2.Canny(frame, self.canny_low, self.canny_high)

        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

        if raw is None:
            return np.empty((0, 4), dtype=np.float32)

        return raw.reshape(-1, 4).astype(np.float32)

    def _extend_segment(
        self, seg: np.ndarray, frame_w: int, frame_h: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extend a segment to a full line across the frame.
        """
        x1, y1, x2, y2 = seg.astype(float)
        if abs(x2 - x1) < 1e-6:
            return np.array([x1, 0.0]), np.array([x1, float(frame_h)])
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        xa, ya = 0.0, b
        xb, yb = float(frame_w), m * frame_w + b
        return np.array([xa, ya]), np.array([xb, yb])


    def _wall_floor_fallback_line(self, frame: np.ndarray) -> Optional[np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        roi_y = int(h * 0.65)
        roi = gray[roi_y:, :]

        edges = cv2.Canny(roi, self.canny_low, self.canny_high)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=max(30, self.hough_threshold // 2),
            minLineLength=max(40, int(w * 0.35)),
            maxLineGap=self.hough_max_line_gap,
        )

        if lines is None:
            return None

        raw = lines.reshape(-1, 4)

        def score(l: np.ndarray) -> float:
            x1, y1, x2, y2 = l.astype(float)
            length = np.hypot(x2 - x1, y2 - y1)
            return length - abs((y1 + y2) * 0.5 - roi.shape[0] * 0.5) * 0.5

        best = max(raw, key=score)
        x1, y1, x2, y2 = best.astype(float)

        return np.array([x1, y1 + roi_y, x2, y2 + roi_y], dtype=np.float32)