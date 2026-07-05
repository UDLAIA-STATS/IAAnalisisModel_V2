from dataclasses import dataclass
from typing import Optional


@dataclass
class NumberObservation:
    player_id: int
    frame_number: int
    number: Optional[int]
    confidence: float
    visible_prob: float
    tens_pred: int
    tens_prob: float
    tens_none_prob: float
    units_pred: int
    units_prob: float


@dataclass
class BBox:
    player_id: int
    frame_number: int
    x1: float
    y1: float
    x2: float
    y2: float