from sqlmodel import Field, SQLModel


class BallState(SQLModel, table=True):
    __tablename__ = "ball_states" # type: ignore
    id: int = Field(primary_key=True, index=True)
    match_id: int = Field(index=True)
    frame_num: int = Field(index=True)
    timestamp_ms: float = Field(index=True)
    conf: float
    x1: float = Field(nullable=True)
    y1: float = Field(nullable=True)
    x2: float = Field(nullable=True)
    y2: float = Field(nullable=True)
    dx: float = Field(nullable=True)
    dy: float = Field(nullable=True)
    vx: float = Field(nullable=True)
    vy: float = Field(nullable=True)
    ax: float = Field(nullable=True)
    ay: float = Field(nullable=True)