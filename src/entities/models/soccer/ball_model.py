from sqlmodel import Field

from src.entities.models.base_models import AuditTable, NumericIdModel
from src.entities.models.soccer.soccer_base_models import BBoxModel, DynamicMovementModel, SoccerFrameData


class BallState(NumericIdModel, AuditTable, BBoxModel, SoccerFrameData, DynamicMovementModel, table=True):
    __tablename__ = "ball_states"  # type: ignore
    match_id: int = Field(index=True)
