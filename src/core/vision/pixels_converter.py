import traceback

import cv2
from cv2.typing import MatLike
import logfire
import numpy as np

class PixelsConverter:
    def __init__(self):
        self.pixels_to_meters = 0.1048
        self.alpha = float(0.15)
        self.top_margin_fraction = float(0.15)
        self.vertical_camp_meters = 40.2
        self.horizontal_camp_meters = 40.32
        self.min_scale: float = 0.01
        self.max_scale: float = 1.0

    def calculate_area_boundary_ends(
        self, frame: MatLike,
        ) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(
                hsv, np.array([0, 0, 140]), np.array([180, 65, 255])
            )
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_and(gray, gray, mask=mask)
            edges = cv2.Canny(gray, 30, 150, apertureSize=3)
            
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=50,
                minLineLength=int(0.30 * frame.shape[1]),
                maxLineGap=20,
            )

            if lines is None:
                logfire.debug(
                    "[CalculateAreaBoundaryEnds] No se detectaron lineas en el frame."
                )
                return None

            # quedarse solo con horizontales
            hor = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = abs(np.arctan2(y2 - y1, x2 - x1))
                if angle < 0.105:
                    hor.append(line[0])

            if not hor:
                logfire.debug(
                    "[CalculateAreaBoundaryEnds] No se detectaron lineas horizontales en el frame."
                )
                return None

            # la mas baja (linea de area inferior)
            frame_h = frame.shape[0]
            candidates = []

            for line in hor:
                y_avg = (line[1] + line[3]) / 2
                length = np.hypot(line[2] - line[0], line[3] - line[1])

                if y_avg > frame_h * self.top_margin_fraction:
                    candidates.append((length, line))

            if not candidates:
                candidates = [(np.hypot(l[2]-l[0], l[3]-l[1]), l) for l in hor]

            _, best_line = max(candidates, key=lambda x: x[0])
            x1, y1, x2, y2 = best_line

            logfire.debug(
                "[CalculateAreaBoundaryEnds] Linea de area detectada en coordenadas: "
                f"({x1}, {y1}), ({x2}, {y2})"
            )
            if x1 <= x2:        
                A = np.array([x1, float(y1)], dtype=float)
                B = np.array([x2, float(y2)], dtype=float)
            else:
                A = np.array([x2, float(y2)], dtype=float)
                B = np.array([x1, float(y1)], dtype=float)

            logfire.debug(
                "[CalculateAreaBoundaryEnds] Extremos de la linea de area: "
                f"A={A}, B={B}"
            )
            return A, B
        except Exception as e:
            logfire.error(f"[CalculateAreaBoundaryEnds] Error: {traceback.format_exc()}")
            return None
        
    def get_current_conversion(self):
        return self.pixels_to_meters

    def calculate_value(self, frame: MatLike):
        area_boundary = self.calculate_area_boundary_ends(frame)

        if area_boundary is not None:
            dist_px = np.linalg.norm(area_boundary[1] - area_boundary[0])
            if dist_px > 130:
                px_to_meters = float(self.horizontal_camp_meters / dist_px)
                self._update_conversion(px_to_meters)

        return self.pixels_to_meters

    def _update_conversion(self, new_value: float):
        clamped = float(np.clip(new_value, self.min_scale, self.max_scale))
        self.pixels_to_meters = (
            self.alpha * clamped + (1 - self.alpha) * self.pixels_to_meters
        )


pixel_conversion_handler = PixelsConverter()