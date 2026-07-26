from sqlmodel import Field, Relationship

from src.entities.models.base_models import AuditTable, NumericIdModel
from src.entities.models.soccer.soccer_base_models import BBoxModel, DynamicMovementModel, SoccerFrameData


class PlayerModel(NumericIdModel, AuditTable, table=True):
    __tablename__: str = "players"  # type: ignore

    match_id: int = Field(index=True)
    track_id: int = Field(index=True)
    team_id: int = Field(index=True, default=None, nullable=True)
    team_color: str = Field(index=True, default=None, nullable=True)
    team_goals: int = Field(default=0)

    goals: int = Field(default=0)
    shots: int = Field(default=0)
    shirt_number: int = Field(nullable=True, default=None)
    ball_possession_time: float = Field(default=0)

    crop_path: str = Field(nullable=True, default=None)
    heatmap_path: str = Field(nullable=True, default=None)
    team_heatmap_path: str = Field(nullable=True, default=None)
    movement_trajectories_path: str = Field(nullable=True, default=None)
    player_movement_trajectories_path: str = Field(nullable=True, default=None)
    team_color_time_kde_path: str = Field(nullable=True, default=None)
    voronoi_territories_path: str = Field(nullable=True, default=None)

    states: list["PlayerState"] = Relationship(back_populates="player", cascade_delete=True)
    numbers: list["PlayerNumbers"] = Relationship(back_populates="player", cascade_delete=True)
    depth_history: list["DepthHistory"] = Relationship(back_populates="player", cascade_delete=True)


class PlayerNumbers(NumericIdModel, AuditTable, table=True):
    __tablename__: str = "players_numbers"  # type: ignore
    player_id: int = Field(foreign_key="players.id", index=True)
    frame_number: int = Field(index=True)
    number: int = Field(nullable=True, default=None)
    tens_none_prob: float
    confidence: float
    visible_prob: float
    tens_pred: int
    tens_prob: float
    units_pred: int
    units_prob: float

    player: PlayerModel = Relationship(back_populates="numbers")


class DepthHistory(NumericIdModel, AuditTable, table=True):
    __tablename__ = "depth_history"  # type: ignore

    match_id: int = Field(index=True)
    frame_num: int = Field(index=True)
    timestamp: float = Field(index=True, decimal_places=10)

    depth: float = Field(
        default=1.0,
        decimal_places=6,
        description="Profundidad del campo en relacion con la camara",
    )
    pixels_to_meters: float = Field(
        default=1.0, decimal_places=6, description="Conversion de pixeles a metros"
    )
    camera_scale: float = Field(
        default=1.0,
        decimal_places=6,
        description="Escala de la camara (nivel de zoom o aumento focal)",
    )
    camera_tilt: float = Field(
        default=0.0,
        decimal_places=6,
        description="Inclinacion de la camara en grados",
    )
    player_id: int = Field(
        foreign_key="players.id",
        index=True,
        description="Id del jugador al que pertenece la constante",
    )

    constant: float = Field(
        default=1.0,
        decimal_places=6,
        description="Constante de conversion, resultado de depth * pixels_to_meters, camera scale is already considered in depth",
    )

    player: "PlayerModel" = Relationship(back_populates="depth_history")



class PlayerState(NumericIdModel, AuditTable, BBoxModel, SoccerFrameData, DynamicMovementModel, table=True):
    __tablename__: str = "players_states"  # type: ignore
    player_id: int = Field(foreign_key="players.id", index=True)

    is_goal: bool = Field(description="True si el jugador metio gol", default=False)
    has_ball: bool = Field(description="True si el jugador tiene poseasion de la pelota", default=False)

    ball_x: float | None = Field(default=None, description="Posicion de la pelota en x, solo se ingresa si el jugador posee el balon", nullable=True)
    ball_y: float | None = Field(default=None, description="Posicion de la pelota en y, solo se ingresa si el jugador posee el balon", nullable=True)

    ball_id: int | None = Field(foreign_key="ball_states.id", default=None, description="Id de la pelota si el jugador posee el balon", nullable=True)

    player: PlayerModel = Relationship(back_populates="states")
