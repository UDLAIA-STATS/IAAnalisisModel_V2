from typing import Tuple

from src.entities.utils.spark_instance import spark


class GoalValidatorBase:
    """Base class with constants for goal post validation."""

    MIN_CONFIDENCE: float = 0.25          # umbral para arcos (promedio 0.3)
    MIN_BBOX_AREA: float = 50.0           # área mínima en píxeles²
    MAX_NUM_POSTS: int = 2                # máximo de arcos en la escena
    MIN_CONSECUTIVE_FRAMES: int = 3       # mínimo de detecciones para considerar un arco real
    CLUSTER_DISTANCE_THRESHOLD: float = 150.0  # px, para separar clusters

    def __init__(self):
        self.spark = spark

    @staticmethod
    def _compute_area(x1: float, y1: float, x2: float, y2: float) -> float:
        return max(0.0, (x2 or 0) - (x1 or 0)) * max(0.0, (y2 or 0) - (y1 or 0))

    @staticmethod
    def _compute_centroid(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        cx = ((x1 or 0) + (x2 or 0)) / 2.0
        cy = ((y1 or 0) + (y2 or 0)) / 2.0
        return cx, cy

