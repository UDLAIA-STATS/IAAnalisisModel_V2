from sqlmodel import Field, SQLModel


class BBoxModel(SQLModel):
    x1: float = Field(nullable=True, default=None)
    y1: float = Field(nullable=True, default=None)
    x2: float = Field(nullable=True, default=None)
    y2: float = Field(nullable=True, default=None)


class DynamicMovementModel(SQLModel):
    dx: float = Field(nullable=True, description="Posicion en x cruda a partir de las coordenadas del bbox", default=None)
    dy: float = Field(nullable=True, description="Posicion en y cruda a partir de las coordenadas del bbox", default=None)
    dx_meters: float = Field(nullable=True, default=None)
    dy_meters: float = Field(nullable=True, default=None)

    distance_meters: float = Field(default=0)
    delta_x: float = Field(default=0, description="Distancia vectorial a partir del calculo realizado en el modulo correspondiente")
    delta_y: float = Field(default=0, description="Distancia vectorial a partir del calculo realizado en el modulo correspondiente")

    acceleration: float = Field(default=0, description="Acceleracion escalar del jugador en m/s^2")
    ax: float = Field(default=0, description="Aceleracion en x en m/s^2")
    ay: float = Field(default=0, description="Aceleracion en y en m/s^2")

    speed_kmh: float = Field(default=0, description="Velocidad del jugador en km/h")
    vx: float = Field(default=0, description="Velocidad en x en m/s")
    vy: float = Field(default=0, description="Velocidad en y en m/s")


class SoccerFrameData(SQLModel):
    frame_number: int = Field(index=True)
    timestamp_ms: int = Field(index=True)
    confidence: float
