from collections import defaultdict
from dataclasses import dataclass
import json
from typing import List, Optional

import numpy as np
from sqlmodel import Column, Field, Relationship, Text

from src.entities.models import NumericIdModel, AuditTable

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


class FeatureDetectorBase:
    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta keypoints y líneas en el frame.
        Retorna un diccionario con:
        - 'kp_dict': dict[int, dict[str, float]] -> {keypoint_id: {'x': px, 'y': py}}
        - 'lines_dict': dict[int, dict[str, float]] -> {line_id: {'x_1': px, 'y_1': py, 'x_2': px, 'y_2': py}}
        - 'confidence': float (opcional)
        - 'image_width': int
        - 'image_height': int
        """
        raise NotImplementedError


class CalibratorBase:
    def calibrate(
        self,
        kp_dict: Dict[int, Dict[str, float]],
        lines_dict: Dict[int, Dict[str, float]],
        tilt: float,  # ángulo de inclinación (grados)
        zoom: float,  # factor de zoom (escala)
        image_width: int,
        image_height: int,
    ) -> Optional["CalibrationResult"]:
        """
        Estima la homografía y parámetros de cámara.
        Retorna un CalibrationResult con H, intrínsecos, extrínsecos, error, etc.
        """
        raise NotImplementedError


class DetectedKeypoint(NumericIdModel, AuditTable, table=True):
    __tablename__ = "detected_keypoints"  # type: ignore

    name: str
    source: str
    confidence: float = 1.0

    image_pt_x: float
    image_pt_y: float
    field_pt_x: float
    field_pt_y: float

    homography_result_id: int = Field(foreign_key="homography_results.id")
    homography_result: "HomographyResult" = Relationship(back_populates="keypoints")

    @property
    def image_pt(self):
        return (self.image_pt_x, self.image_pt_y)

    @property
    def field_pt(self):
        return (self.field_pt_x, self.field_pt_y)


class HomographyResult(NumericIdModel, AuditTable, table=True):
    __tablename__ = "homography_results"  # type: ignore

    H_json: str = Field(sa_column=Column(Text), default="")

    reprojection_error: float
    inlier_count: int
    is_valid: bool
    confidence: float = 1.0
    frame_num: int
    match_id: int

    intrinsics_json: str = Field(sa_column=Column(Text), default="")
    distortion_json: str = Field(sa_column=Column(Text), default="")
    rotation_matrix_json: str = Field(sa_column=Column(Text), default="")
    position_meters_json: str = Field(sa_column=Column(Text), default="")
    pan_tilt_roll_json: str = Field(sa_column=Column(Text), default="")
    method_used: str = Field(default="")
    line_inlier_count: int = Field(default=0)
    scale_estimate: float = Field(default=1.0)

    keypoints: List[DetectedKeypoint] = Relationship(back_populates="homography_result")

    @property
    def H(self) -> np.ndarray:
        return np.array(json.loads(self.H_json), dtype=np.float32)

    @H.setter
    def H(self, value: np.ndarray):
        self.H_json = json.dumps(value.tolist())


@dataclass
class CleanHomographyFrame:
    frame_num: int
    H: np.ndarray
    reprojection_error: float
    inlier_count: int
    total_kpts: int
    detected_kpts: int
    predetermined_kpts: int
    avg_conf: Optional[float]
    avg_det_conf: Optional[float]


@dataclass
class CalibrationResult:
    H: np.ndarray  # Homografía 3x3
    intrinsics: np.ndarray  # Matriz de cámara K (3x3)
    distortion: np.ndarray  # Coeficientes de distorsión (5,)
    rotation_matrix: np.ndarray  # Matriz de rotación R (3x3)
    position_meters: np.ndarray  # Vector de traslación t (3,)
    pan_deg: float
    tilt_deg: float
    roll_deg: float
    reprojection_error: float
    inlier_count: int
    line_inlier_count: Optional[int]  # Número de líneas que apoyan la calibración
    method_used: str  # ej. "pnl_full", "pnl_ground", "ransac"
    scale_estimate: float  # Escala estimada (metros/píxel) a partir de la homografía
    is_valid: bool
    frame_num: int
    match_id: int
    keypoints_used: Dict[int, Dict[str, float]]
    lines_used: Dict[int, Dict[str, float]]
