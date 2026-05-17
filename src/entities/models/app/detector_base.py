from pathlib import Path
from typing import List, Sequence, Union

import logfire
import torch
from ultralytics.models import YOLO
from ultralytics.engine.results import Results

from entities.models.app.video_item import VideoItem
from entities.models.soccer.player_model import PlayerState
from entities.services.annotator_service import AnnotatorServiceBase
from entities.types.detector_types import DetectorTypes
from supervision.detection.core import Detections
from ultralytics.engine.results import Results

class DetectorBase():
    def __init__(self, model: YOLO, tracker_config_file: Path, type: DetectorTypes = DetectorTypes.DETECTION):
        self.model: YOLO = model
        self.tracker_config_file: str = tracker_config_file.as_posix()
        self.type: DetectorTypes = type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame) -> Sequence[Union[Results, Detections]]:
        """Detect objects in a frame."""
        if self.type == DetectorTypes.TRACKING:
            return self.model.track(frame, tracker=self.tracker_config_file, persist=True, conf=0.5, iou=0.6, verbose=False, device=self.device)
        else:
            return self.model(frame, conf=0.3, iou=0.45, verbose=False, device=self.device)

    def extract_detections(self, results: Sequence[Union[Results, Detections]], objects_ids: List[int]) -> dict[int, Union[Results, Detections]]:
        detections_map = {}
        detections = Detections.from_ultralytics(results[0])

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            detections_map[object_id] = filtered_detections

        return detections_map


    def get_tracks(self, video_item: VideoItem, object_ids: List[int]):
        detections = self.detect(video_item.frame)
        filtered_detections = self.extract_detections(detections, object_ids)

        for object_id, filtered_detection in filtered_detections.items():
            for i in range(len(filtered_detection)):
                if filtered_detection is None:
                    continue

                x1, y1, x2, y2 = map(int, filtered_detection.xyxy[i])
                if filtered_detection.confidence is None:
                    logfire.warning(f"[DetectorBase] No confidence for object {object_id} in frame {video_item.frame_num}")
                    conf = 0.3
                else:
                    conf = filtered_detection.confidence[i]

                track_id = 0

                if self.type == DetectorTypes.TRACKING and isinstance(filtered_detection, Results):
                    track_id = filtered_detection.track_id[i]

                yield object_id, track_id, x1, y1, x2, y2

        
