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
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

        l_channel = lab[:, :, 0]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(l_channel)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, -5
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

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
