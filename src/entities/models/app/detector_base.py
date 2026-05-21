from pathlib import Path
from typing import Generator, List, Sequence, Union

import logfire
from pydantic import BaseModel, ConfigDict
from sqlmodel import Session
import torch
from ultralytics.models import YOLO
from ultralytics.engine.results import Results

from entities.models.app.track_data import TrackData
from entities.models.app.video_item import VideoItem
from entities.types.detector_types import DetectorTypes
from supervision.detection.core import Detections


class DetectorBase:
    def __init__(self, model: Path, tracker_config_file: Path | None, type: DetectorTypes = DetectorTypes.DETECTION):
        """
        Initialize detector class to get tracks in base using a YOLO model with detection only
        or tracking and detection.
        Args:
            model (Path): Path to the YOLO model.
            tracker_config_file (Path): Path to the tracker config file.
            type (DetectorTypes, optional): Type of detector. Defaults to D etectorTypes.DETECTION.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__init_model__(model)
        if tracker_config_file is not None:
            self.tracker_config_file: str = tracker_config_file.as_posix()

        self.type: DetectorTypes = type

    def __init_model__(self, model: Path, half: bool = False):
        self.model: YOLO = YOLO(model.as_posix())

        if model.suffix == ".pt":
            self.model.to(self.device)
            self.model.fuse()

        if half:
            self.model.half()

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc, tb):
        pass

    def detect(self, frame) -> Sequence[Union[Results, Detections]]:
        """Detect objects in a frame."""
        if self.type == DetectorTypes.TRACKING and self.tracker_config_file is not None:
            return self.model.track(frame, tracker=self.tracker_config_file, persist=True, conf=0.5, iou=0.6, verbose=False, device=self.device)
        else:
            return self.model(frame, conf=0.3, iou=0.45, verbose=False, device=self.device)

    def extract_detections(self, results: Sequence[Union[Results, Detections]], objects_ids: List[int]) -> dict[int, Detections]:
        detections_map = {}
        detections = Detections.from_ultralytics(results[0])

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            detections_map[object_id] = filtered_detections

        return detections_map

    def _extract_tracks_data(
        self, filtered_detections: dict[int, Detections], frame_num: int, video_item: VideoItem
    ) -> Generator[TrackData, None, None]:
        for object_id, filtered_detection in filtered_detections.items():
            for i in range(len(filtered_detection)):
                if filtered_detection is None:
                    continue

                x1, y1, x2, y2 = map(int, filtered_detection.xyxy[i])

                if filtered_detection.confidence is None:
                    logfire.warning(f"[DetectorBase] No confidence for object {object_id} in frame {frame_num}")
                    conf = 0.3
                else:
                    conf = filtered_detection.confidence[i]

                track_id = 0

                if self.type == DetectorTypes.TRACKING and isinstance(filtered_detection, Results):
                    track_id = filtered_detection.track_id[i]

                yield TrackData(xyxy=(x1, y1, x2, y2), track_id=track_id, confidence=conf)

    def get_tracks(self, video_item: VideoItem, object_ids: List[int], session: Session):
        detections = self.detect(video_item.frame)
        filtered_detections = self.extract_detections(detections, object_ids)

        track_data = self._extract_tracks_data(filtered_detections, video_item.frame_num, video_item)
        self._save_tracks(track_data, video_item, session)

    def _save_tracks(self, detected_tracks: Generator[TrackData], video_item: VideoItem, session: Session):
        pass


class TrackManagerItem(BaseModel):
    tracker: DetectorBase
    object_ids: List[int]

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
