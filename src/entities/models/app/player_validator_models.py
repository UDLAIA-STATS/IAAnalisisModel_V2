from pydantic import BaseModel


class PlayerCentroid(BaseModel):
    player_id: int
    tracker_id: int
    cx: float
    cy: float
    color: str
    conf: float
    frame_number: int


class TrackSummary(BaseModel):
    player_id: int
    tracker_id: int
    frame_start: int
    frame_end: int
    detection_count: int

    timestamp_start: float
    timestamp_end: float

    avg_color: str
    avg_height: float
    avg_width: float
    avg_confidence: float

    first_cx: float
    first_cy: float

    last_cx: float
    last_cy: float

    mid_cx: float
    mid_cy: float
    mid_x1: float
    mid_y1: float
    mid_x2: float
    mid_y2: float

    first_x1: float
    first_y1: float
    first_x2: float
    first_y2: float

    last_x1: float
    last_y1: float
    last_x2: float
    last_y2: float

