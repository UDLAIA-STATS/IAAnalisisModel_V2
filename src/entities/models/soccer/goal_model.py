from sqlmodel import Field

from src.entities.models.base_models import AuditTable, NumericIdModel
from src.entities.models.soccer.soccer_base_models import BBoxModel, SoccerFrameData


class GoalModel(NumericIdModel, AuditTable, SoccerFrameData, BBoxModel, table=True):
    __tablename__ = "goals"  # type: ignore

    match_id: int = Field(index=True)
