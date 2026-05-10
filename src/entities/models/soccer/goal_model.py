from sqlmodel import Field, SQLModel


class GoalModel(SQLModel, table=True):
    __tablename__ = "goals" # type: ignore
    id: int = Field(primary_key=True, index=True)
    match_id: int = Field(index=True)
    frame_num: int = Field(index=True)
    timestamp_ms: float = Field(index=True)
    x1: float = Field(nullable=True)
    y1: float = Field(nullable=True)
    x2: float = Field(nullable=True)
    y2: float = Field(nullable=True)
    conf: float