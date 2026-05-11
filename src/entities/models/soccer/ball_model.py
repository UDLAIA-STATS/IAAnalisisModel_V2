from datetime import timezone

from logfire.query_client import datetime
from sqlmodel import Field, SQLModel

from entities.models.base_models import AuditTable, NumericIdModel
from entities.models.soccer.soccer_base_models import BBoxModel, DynamicMovementModel, SoccerFrameData


class BallState(
    NumericIdModel,
    AuditTable,
    BBoxModel,
    SoccerFrameData,
    DynamicMovementModel,
    table=True):
    __tablename__ = "ball_states" # type: ignore
    match_id: int = Field(index=True)

    __table_args__ = (
        {"indexes": [
            {"columns": ["match_id", "frame_num"]},
            {"columns": ["match_id", "timestamp_ms"]},
            {"columns": ["match_id", "confidence"]},
        ]},
    )
