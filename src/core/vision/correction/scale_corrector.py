import numpy as np
from typing import Dict, List, Optional
from sqlmodel import Session, select

from src.entities.models.homography.homography_models import HomographyResult
from src.config.constants import PITCH_LENGTH, PITCH_WIDTH


class ScaleCorrector:
    SOFT_MARGIN = 2.0

    def __init__(self, session: Optional[Session] = None):
        self.session = session
        self._scales_cache: Dict[int, List[float]] = {}

    def estimate_scale(self, H: np.ndarray) -> float:
        """Estima la escala usando el ancho del área penal (16.5m)."""
        p1 = np.array([0.0, 54.16, 1.0])
        p2 = np.array([0.0, 37.66, 1.0])
        proj1 = H @ p1
        proj1 /= proj1[2]
        proj2 = H @ p2
        proj2 /= proj2[2]
        dist_px = np.linalg.norm(proj1[:2] - proj2[:2])
        if dist_px > 0:
            return 16.5 / float(dist_px)
        return 1.0

    def add_scale(self, match_id: int, scale: float) -> None:
        """Almacena una escala para un partido."""
        if match_id not in self._scales_cache:
            self._scales_cache[match_id] = []
        self._scales_cache[match_id].append(scale)

    def get_robust_scale(self, match_id: int) -> float:
        """Retorna la mediana de las escalas almacenadas para el partido."""
        scales = self._scales_cache.get(match_id, [])
        if not scales:
            if self.session:
                stmt = select(HomographyResult).where(
                    HomographyResult.match_id == match_id,
                    HomographyResult.is_valid == True,
                    HomographyResult.scale_estimate > 0
                )
                results = self.session.exec(stmt).all()
                scales = [r.scale_estimate for r in results]
                if scales:
                    self._scales_cache[match_id] = scales
        if scales:
            return float(np.median(scales))
        return 1.0

    def correct_point(
        self,
        x: float,
        y: float,
        match_id: int,
        robust_scale: Optional[float] = None
    ) -> tuple[float, float]:
        """
        Corrige un punto proyectado que está fuera de los límites de la cancha.
        Usa un clamp radial: mueve el punto al borde más cercano en la dirección
        desde el centro de la cancha, preservando la dirección.
        """
        if (-self.SOFT_MARGIN <= x <= PITCH_LENGTH + self.SOFT_MARGIN and
            -self.SOFT_MARGIN <= y <= PITCH_WIDTH + self.SOFT_MARGIN):
            return x, y

        cx = PITCH_LENGTH / 2
        cy = PITCH_WIDTH / 2

        dx = x - cx
        dy = y - cy

        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return x, y

        max_x = PITCH_LENGTH - cx if dx >= 0 else cx  # distancia al borde en x
        max_y = PITCH_WIDTH - cy if dy >= 0 else cy   # distancia al borde en y

        factor_x = max_x / abs(dx) if abs(dx) > 0 else float('inf')
        factor_y = max_y / abs(dy) if abs(dy) > 0 else float('inf')

        factor = min(factor_x, factor_y)

        if factor >= 1.0:
            return x, y

        factor *= 0.99

        x_corr = cx + dx * factor
        y_corr = cy + dy * factor

        return x_corr, y_corr