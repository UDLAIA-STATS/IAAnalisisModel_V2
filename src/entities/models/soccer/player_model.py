from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class PlayerModel(SQLModel, table=True):
    __tablename__: str = "players" # type: ignore
    id: int = Field(primary_key=True, index=True)
    match_id: int = Field(index=True)
    track_id: int = Field(index=True)
    team: int = Field(index=True)
    team_color: str
    goals: int = Field(default=0) # Goles del jugador
    shirt_number: int = Field(nullable=True, default=None)
    create_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    update_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    

class PlayerNumbers(SQLModel, table=True):
    __tablename__: str = "players_numbers" # type: ignore
    id: int = Field(primary_key=True, index=True)
    player_id: int = Field(foreign_key="players.id", index=True)
    number: int
    confiability: float
    frame_number: int

class PlayerState(SQLModel, table=True):
    __tablename__: str = "players_states" # type: ignore
    id: int = Field(primary_key=True, index=True)
    player_id: int = Field(foreign_key="players.id", index=True)
    timestamp_ms: float = Field(index=True)
    frame_number: int = Field(index=True)
    confiability: float
    x1: float = Field(nullable=True)
    y1: float = Field(nullable=True)
    x2: float = Field(nullable=True)
    y2: float = Field(nullable=True)
    dx: float = Field(nullable=True, description="Posicion en x cruda a partir de las coordenadas del bbox")
    dy: float = Field(nullable=True, description="Posicion en y cruda a partir de las coordenadas del bbox")
    dx_meters: float = Field(nullable=True)
    dy_meters: float = Field(nullable=True)
    is_goal: bool = Field(description="True si el jugado metio gol", default=False)
    has_ball: bool = Field(description="True si el jugador tiene poseasion de la pelota", default=False)
    distance_meters: float = Field(default=0)
    delta_x: float = Field(default=0, description="Distancia vectorial a partir del calculo realizado en el modulo correspondiente")
    delta_y: float = Field(default=0, description="Distancia vectorial a partir del calculo realizado en el modulo correspondiente")
    acceleration: float = Field(default=0, description="Acceleracion escalar del jugador en m/s^2")
    ax: float = Field(default=0, description="Aceleracion en x en m/s^2")
    ay: float = Field(default=0, description="Aceleracion en y en m/s^2")
    speed_kmh: float = Field(default=0, description="Velocidad del jugador en km/h")
    vx: float = Field(default=0, description="Velocidad en x en m/s")
    vy: float = Field(default=0, description="Velocidad en y en m/s")
    ball_x: float | None = Field(default=None, description="Posicion de la pelota en x, solo se ingresa si el jugador posee el balon", nullable=True)
    ball_y: float | None = Field(default=None, description="Posicion de la pelota en y, solo se ingresa si el jugador posee el balon", nullable=True)
    ball_id: int = Field(
        foreign_key="ball_states.id",
        default=None,
        description="Id de la pelota si el jugador posee el balon",
        nullable=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
