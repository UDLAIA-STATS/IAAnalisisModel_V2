from sqlmodel import Field

from entities.models.base_models import NumericIdModel
from entities.models.soccer.soccer_base_models import BBoxModel


class DetectorReport(NumericIdModel, BBoxModel, table=True):
    __tablename__ = "detector_reports"  # type: ignore
    frame_number: int = Field(index=True)
    detection_class: str = Field(index=True)
    track_id: int
    shirt_color: str = Field(default=None, nullable=True)
    shirt_number: int = Field(default=None, nullable=True)
    speed: float = Field(default=0)
    distance: float = Field(default=0)
    confidence: float = Field(index=True)
    timestamp_ms: int = Field(index=True)
