from sqlmodel import Field, Relationship

from src.entities.models.base_models import AuditTable, NumericIdModel
from src.entities.models.soccer.soccer_base_models import BBoxModel, DynamicMovementModel, SoccerFrameData


class PlayerModel(NumericIdModel, AuditTable, table=True):
    __tablename__: str = "players"  # type: ignore

    match_id: int = Field(index=True)
    track_id: int = Field(index=True)
    team_id: int = Field(index=True, default=None, nullable=True)
    team_color: str = Field(index=True, default=None, nullable=True)
    goals: int = Field(default=0)  # Goles del jugador
    shirt_number: int = Field(nullable=True, default=None)
    crop_path: str = Field(nullable=True, default=None)

    states: list["PlayerState"] = Relationship(back_populates="player")
    numbers: list["PlayerNumbers"] = Relationship(back_populates="player")


class PlayerNumbers(NumericIdModel, AuditTable, table=True):
    __tablename__: str = "players_numbers"  # type: ignore
    player_id: int = Field(foreign_key="players.id", index=True)
    number: int
    confidence: float
    frame_number: int

    player: PlayerModel = Relationship(back_populates="numbers")


class PlayerState(NumericIdModel, AuditTable, BBoxModel, SoccerFrameData, DynamicMovementModel, table=True):
    __tablename__: str = "players_states"  # type: ignore
    player_id: int = Field(foreign_key="players.id", index=True)

    is_goal: bool = Field(description="True si el jugador metio gol", default=False)
    has_ball: bool = Field(description="True si el jugador tiene poseasion de la pelota", default=False)

    ball_x: float | None = Field(default=None, description="Posicion de la pelota en x, solo se ingresa si el jugador posee el balon", nullable=True)
    ball_y: float | None = Field(default=None, description="Posicion de la pelota en y, solo se ingresa si el jugador posee el balon", nullable=True)

    ball_id: int | None = Field(foreign_key="ball_states.id", default=None, description="Id de la pelota si el jugador posee el balon", nullable=True)

    player: PlayerModel = Relationship(back_populates="states")

    def recalculate_physics(self):
        self.dx = 0
        self.dy = 0
        self.delta_x = 0
        self.delta_y = 0
        self.distance_meters = 0

        self.vx = 0
        self.vy = 0
        self.speed_kmh = 0

        self.acceleration = 0
        self.ax = 0
        self.ay = 0
