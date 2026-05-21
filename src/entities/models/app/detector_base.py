from pathlib import Path
from typing import Generator, List, Sequence, Type, Union

import logfire
from pydantic import BaseModel, ConfigDict
from sqlmodel import SQLModel, Session
import torch
from ultralytics.models import YOLO
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

from src.core.database import connection_manager
from src.entities.models.app.track_data import TrackData
from src.entities.models.app.video_item import VideoItem
from src.entities.types.detector_types import DetectorTypes


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
        self.classes = {}
        self.types_map = {
            0: SQLModel
        }

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
            return self.model.track(
                frame, tracker=self.tracker_config_file,
                persist=True, conf=0.5, iou=0.6,
                verbose=False, device=self.device)
        else:
            return self.model(
                frame, conf=0.3, iou=0.45,
                verbose=False, device=self.device)

    def extract_detections(self, results: Sequence[Union[Results, Detections]], objects_ids: List[int], video_item: VideoItem) -> dict[int, List[TrackData]]:
        detections_map: dict[int, List[TrackData]] = {}
        detections = Detections.from_ultralytics(results[0])

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            annotator = self.classes[object_id]
            annotator.set_detections(filtered_detections)

            data = list(self._extract_tracks_data(filtered_detections, video_item)) # type: ignore
            
            if not object_id in detections_map:
                detections_map[object_id] = data
            else:
                detections_map[object_id].extend(data)

        return detections_map

    # TODO: At the return of the item the detection mixes with others objects, it needs to be separated
    def _extract_tracks_data(
        self, detections: Detections, video_item: VideoItem
    ) -> Generator[TrackData, None, None]:
        for i in range(len(detections)):
            if detections is None:
                continue

            x1, y1, x2, y2 = map(int, detections.xyxy[i])

            if detections.confidence is None:
                logfire.warning(f"[DetectorBase] No confidence for object in frame {video_item.frame_num} match id {video_item.match_id}")
                conf = 0.3
            else:
                conf = detections.confidence[i]

            if self.type == DetectorTypes.TRACKING:
                track_id = detections.tracker_id[i] # type: ignore
            else:
                track_id = 0

            yield TrackData(xyxy=(x1, y1, x2, y2), track_id=track_id, confidence=conf)

    def get_tracks(self, video_item: VideoItem, object_ids: List[int]):
        with connection_manager.create_session() as session:
            detections = self.detect(video_item.frame)
            track_data = self.extract_detections(detections, object_ids, video_item)

            for object_id, data in track_data.items():
                object = self.types_map[object_id]
                self._save_tracks(data, video_item, object, session)

    def _save_tracks(
            self,
            detected_tracks: List[TrackData],
            video_item: VideoItem,
            object: Type[SQLModel],
            session: Session):
        pass


class TrackManagerItem(BaseModel):
    tracker: DetectorBase
    object_ids: List[int]

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)
