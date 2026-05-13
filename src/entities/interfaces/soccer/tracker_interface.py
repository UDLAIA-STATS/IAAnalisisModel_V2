from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

import torch
from ultralytics.models import YOLO
from ultralytics.engine.results import Results

from entities.types.detector_types import DetectorTypes
from supervision.detection.core import Detections

class IDetector(ABC):
    def __init__(
            self,
            model: YOLO,
            tracker_config_file: Path,
            type: DetectorTypes = DetectorTypes.DETECTION):
        self.model: YOLO = model
        self.tracker_config_file: str = tracker_config_file.as_posix()
        self.type: DetectorTypes = type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @abstractmethod
    def detect(self, frame):
        """Detect objects in a frame."""
        if self.type == DetectorTypes.TRACKING:
            return self.model.track(
                frame,
                tracker=self.tracker_config_file,
                persist=True,
                conf=0.5,
                iou=0.6,
                verbose=False,
                device=self.device
                )
        else:
            return self.model(
                frame,
                conf=0.3,
                iou=0.45,
                verbose=False,
                device=self.device
            )
    
    def extract_detections(self, results: List[Results], objects_ids: List[int]) -> dict[int, Detections]:
        detections_map = {}
        detections = Detections.from_ultralytics(results[0])


        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            detections_map[object_id] = filtered_detections

        return detections_map

    @abstractmethod
    def get_tracks(self, frame):
        pass