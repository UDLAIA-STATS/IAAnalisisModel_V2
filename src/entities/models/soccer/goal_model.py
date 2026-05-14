from sqlmodel import Field

from entities.models.base_models import AuditTable, NumericIdModel
from entities.models.soccer.soccer_base_models import BBoxModel, SoccerFrameData


class GoalModel(NumericIdModel, AuditTable, SoccerFrameData, BBoxModel, table=True):
    __tablename__ = "goals"  # type: ignore

    match_id: int = Field(index=True)

    __table_args__ = (
        {
            "indexes": [
                {"columns": ["match_id", "frame_number"]},
                {"columns": ["match_id", "timestamp_ms"]},
                {"columns": ["match_id", "confidence"]},
            ]
        },
    )
