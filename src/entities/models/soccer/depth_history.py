from datetime import datetime, timezone

from sqlmodel import Field, SQLModel

from entities.models.base_models import AuditTable, NumericIdModel


class DepthHistory(NumericIdModel, AuditTable, table=True):
    __tablename__ = "depth_history"  # type: ignore


    match_id: int = Field(index=True)
    frame_num: int = Field(index=True)
    timestamp_ms: int = Field(index=True)

    depth: float = Field(default=1.0, description="Profundidad del campo en relacion con la camara")
    pixels_to_meters: float = Field(default=1.0, description="Conversion de pixeles a metros")
    camera_scale: float = Field(default=1.0, description="Escala de la camara (nivel de zoom o aumento focal)")

    constant: float = Field(
        default=1.0,
        description="Constante de conversion, resultado de depth * pixels_to_meters * camera_scale")

    @staticmethod
    def create_depth_history(
        depth: float,
        pixels_to_meters:
        float,
        camera_scale: float,
        frame_num: int,
        timestamp_ms: int,
        match_id: int):
        constant = depth * pixels_to_meters * camera_scale
        return DepthHistory(
            depth=depth,
            pixels_to_meters=pixels_to_meters,
            camera_scale=camera_scale,
            constant=constant,
            frame_num=frame_num,
            timestamp_ms=timestamp_ms,
            match_id=match_id
        )
