from typing import Tuple, Optional
from src.entities.utils.spark_instance import spark


class BallPossessionAnalyzerBase:
    # Temporal thresholds
    MAX_POSSESSION_GAP: float = 0.5
    GOAL_ASSOCIATION_WINDOW: float = 1.0

    # Shot detection (heuristic fallback)
    SHOT_DISTANCE_THRESHOLD: float = 25.0
    SHOT_ANGLE_THRESHOLD: float = 45.0

    MIN_POSSESSION_SECONDS: float = 0.1

    # ML inference confidence threshold
    ML_PREDICTION_THRESHOLD: float = 0.65

    def __init__(self, model_path: Optional[str] = None):
        self.spark = spark
        self.model_path = model_path

    @staticmethod
    def _compute_distance(px: float, py: float, qx: float, qy: float) -> float:
        return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5