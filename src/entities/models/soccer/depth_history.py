from sqlmodel import Field

from src.entities.models.base_models import AuditTable, NumericIdModel


class DepthHistory(NumericIdModel, AuditTable, table=True):
    __tablename__ = "depth_history"  # type: ignore

    match_id: int = Field(index=True)
    frame_num: int = Field(index=True)
    timestamp: int = Field(index=True)

    depth: float = Field(default=1.0, description="Profundidad del campo en relacion con la camara")
    pixels_to_meters: float = Field(default=1.0, description="Conversion de pixeles a metros")
    camera_scale: float = Field(default=1.0, description="Escala de la camara (nivel de zoom o aumento focal)")
    player_id: int = Field(foreign_key="players.id", index=True, description="Id del jugador al que pertenece la constante")

    constant: float = Field(default=1.0, description="Constante de conversion, resultado de depth * pixels_to_meters * camera_scale")

