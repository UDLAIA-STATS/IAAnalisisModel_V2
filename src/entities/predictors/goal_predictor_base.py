from typing import Tuple

from src.entities.utils.spark_instance import spark


class GoalValidatorBase:
    """Base class with constants and shared utilities for goal validation."""

    MAX_FRAME_GAP: int = 5  # frames
    MAX_TIMESTAMP_GAP: float = 0.25  # seconds (250ms)

    GOAL_IOU_THRESHOLD: float = 0.30
    MIN_CONFIDENCE: float = 0.50
    MIN_BBOX_AREA: float = 100.0  # minimum valid bbox area in px²

    CONFIDENCE_WEIGHT: float = 0.60
    AREA_WEIGHT: float = 0.25
    TEMPORAL_WEIGHT: float = 0.15

    def __init__(self):
        self.spark = spark

    @staticmethod
    def _compute_area(x1: float, y1: float, x2: float, y2: float) -> float:
        """Compute bbox area safely."""
        return max(0.0, (x2 or 0) - (x1 or 0)) * max(0.0, (y2 or 0) - (y1 or 0))

    @staticmethod
    def _compute_centroid(
        x1: float, y1: float, x2: float, y2: float
    ) -> Tuple[float, float]:
        """Compute bbox centroid."""
        cx = ((x1 or 0) + (x2 or 0)) / 2.0
        cy = ((y1 or 0) + (y2 or 0)) / 2.0
        return cx, cy
