from pydantic import BaseModel


class ReportRow(BaseModel):
    frame_number: int
    object_type: str
    track_id: int
    bbox: str
    dx_meters: str = ""
    dy_meters: str = ""
    confidence: float
    shirt_color: str = ""
    shirt_number: str = ""
    speed: float = 0.0
    distance: float = 0.0
    acceleration: float = 0.0
    timestamp: float
    homography_results: str = ""

