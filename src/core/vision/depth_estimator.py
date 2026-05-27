import logfire
import numpy as np

from typing import List, Optional

from src.entities.models.app.depth_estimator_base import DepthEstimatorBase


class DepthEstimator(DepthEstimatorBase):
    def process_player_depth(
        self,
        frame: np.ndarray,
        bbox: List[int],
        frame_num: int,
        current_camera_scale: float,
    ) -> Optional[float]:
        """
        Calcula profundidad para un jugador especifico usando el centro de su cintura.

        Args:
            frame: Frame completo (BGR)
            bbox: [x1, y1, x2, y2] del jugador
            frame_num: Numero de frame actual
            current_camera_scale: Escala actual de la camara

        Returns:
            Profundidad en metros o None si no se calculo/no se detecto pose
        """
        logfire.info("[DepthCalculator] Calculando profundidad para jugador...")
        if not self.should_calculate_depth(frame_num, current_camera_scale):
            return None

        self.last_calculation_frame = frame_num
        self.last_camera_scale = current_camera_scale

        x1, y1, x2, y2 = bbox
        h_frame, w_frame = frame.shape[:2]
        logfire.info(f"[DepthCalculator] Dimensiones del frame: {w_frame}x{h_frame}")

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(int(w_frame), int(x2)), min(int(h_frame), int(y2))
        logfire.info(
            f"[DepthCalculator] Coordenadas extraidas del bbox: x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )

        if x2 <= x1 or y2 <= y1:
            logfire.warning(f"[DepthCalculator] BBox invalido: {bbox}")
            return None

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        waist_center_norm = self._get_waist_center(roi)

        if waist_center_norm is None:
            logfire.debug(
                f"[DepthCalculator] Frame {frame_num}: No se detecto pose, usando centro del bbox"
            )
            waist_x = (x1 + x2) / 2.0
            waist_y = (y1 + y2) / 2.0
        else:
            waist_x = x1 + waist_center_norm[0] * (x2 - x1)
            waist_y = y1 + waist_center_norm[1] * (y2 - y1)

        depth = self._calculate_depth_at_point(frame, int(waist_x), int(waist_y))

        logfire.info(
            f"[DepthCalculator] Frame {frame_num}: "
            f"Profundidad en cintura ({int(waist_x)}, {int(waist_y)}): {depth:.2f}m "
            f"(scale: {current_camera_scale:.3f})"
        )

        return depth
