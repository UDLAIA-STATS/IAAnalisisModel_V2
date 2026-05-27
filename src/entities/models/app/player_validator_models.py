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
    avg_color: str
    first_cx: float
    first_cy: float
    last_cx: float
    last_cy: float
