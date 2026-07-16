from typing import Tuple

from src.entities.utils.spark_instance import spark

class GoalScorerDetectorBase:
    """Base class with constants and shared utilities for goal scorer detection."""

    POSSESSION_LOOKBACK_FRAMES: int = 15
    POSSESSION_LOOKBACK_SECONDS: float = 0.5
    SHOT_LOOKAHEAD_FRAMES: int = 12
    SHOT_LOOKAHEAD_SECONDS: float = 0.5

    NEAR_GOAL_CM_THRESHOLD: float = 150.0  # 1.5 meters = 150 centimeters
    NEAR_GOAL_PX_FALLBACK: float = 15.0

    GOAL_WIDTH_METERS: float = 7.32
    GOAL_HEIGHT_METERS: float = 2.44

    MIN_SHOT_SPEED_KMH: float = 20.0
    MAX_SHOT_SPEED_KMH: float = 130.0

    MIN_GOAL_CONFIDENCE: float = 0.50
    MIN_BALL_CONFIDENCE_FOR_SHOT: float = 0.40

    DEFAULT_METER_PER_PIXEL: float = 0.055

    def __init__(self):
        self.spark = spark

    @staticmethod
    def _compute_centroid(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float]:
        cx = ((x1 or 0) + (x2 or 0)) / 2.0
        cy = ((y1 or 0) + (y2 or 0)) / 2.0
        return cx, cy

    @staticmethod
    def _px_to_meters(px: float, meter_per_px: float = 0.055) -> float:
        return px * meter_per_px

    @staticmethod
    def _cm_to_meters(cm: float) -> float:
        return cm / 100.0

    @staticmethod
    def _meters_to_cm(m: float) -> float:
        return m * 100.0