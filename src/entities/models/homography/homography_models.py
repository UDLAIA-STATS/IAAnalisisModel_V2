from dataclasses import dataclass
import json
from typing import List, Optional

import numpy as np
from sqlmodel import Column, Field, Relationship, Text

from src.entities.models import NumericIdModel, AuditTable

class DetectedKeypoint(NumericIdModel, AuditTable, table = True):
    __tablename__ = "detected_keypoints" # type: ignore

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


class HomographyResult(NumericIdModel, AuditTable, table = True):
    __tablename__ = "homography_results" # type: ignore

    H_json: str  = Field(sa_column=Column(Text), default = "")

    reprojection_error: float
    inlier_count: int
    is_valid: bool
    confidence: float = 1.0
    frame_num: int
    match_id: int

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
    total_kpts: int
    detected_kpts: int
    avg_det_conf: Optional[float]
