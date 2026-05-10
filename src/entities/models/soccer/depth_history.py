from sqlmodel import Field, SQLModel


class DepthHistory(SQLModel, table=True):
    __tablename__ = "depth_history"  # type: ignore

    id: int = Field(primary_key=True, index=True)
    frame_num: int = Field(index=True)
    timestamp_ms: float = Field(index=True)
    depth: float = Field(default=1.0, description="Profundidad del campo en relacion con la camara")
    pixels_to_meters: float = Field(default=1.0, description="Conversion de pixeles a metros")
    camera_scale: float = Field(default=1.0, description="Escala de la camara (nivel de zoom o aumento focal)")
    constant: float = Field(
        default=1.0,
        description="Constante de conversion, resultado de depth * pixels_to_meters * camera_scale")
