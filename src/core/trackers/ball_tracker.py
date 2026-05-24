from pathlib import Path
from typing import Generator, List, Sequence, Union, override
import logfire
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

from sqlmodel import SQLModel, Session

from src.config.routes import BALL_MODEL_PATH
from src.core.repository.ball_repository import BallRepository
from src.entities.models.app.detector_base import DetectorBase
from src.entities.models.app.track_data import TrackData
from src.entities.models.app.video_item import VideoItem
from src.entities.models.soccer.ball_model import BallState
from src.entities.types.detector_types import DetectorTypes

from src.core.video import ball_annotator


class BallTracker(DetectorBase):
    def __init__(self, tracker_config_file: Path | None, model: Path = BALL_MODEL_PATH, type: DetectorTypes = DetectorTypes.DETECTION):
        super().__init__(model, tracker_config_file, type)
        self.classes = {0: ball_annotator}

    @override
    def detect(self, frame) -> Sequence[Union[Results, Detections]]:
        """Detect objects in a frame."""
        return self.model(
            frame,
            conf=0.1,
            verbose=False,
            iou=0.45,
            device=self.device,
        )
    
    @override
    def extract_detections(
        self, results: Sequence[Union[Results, Detections]], objects_ids: List[int], video_item: VideoItem
    ) -> dict[int, List[TrackData]]:
        detections_map: dict[int, List[TrackData]] = {}
        detections = Detections.from_ultralytics(results[0])

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            logfire.notice(f"[BallTracker] Number of detections: {len(filtered_detections)}")
            annotator = self.classes[object_id]
            annotator.set_detections(filtered_detections)

            data = list(self._extract_tracks_data(filtered_detections, video_item))  # type: ignore

            if object_id not in detections_map:
                detections_map[object_id] = data
            else:
                detections_map[object_id].extend(data)

        return detections_map

    @override
    def _extract_tracks_data(self, detections: Detections, video_item: VideoItem) -> Generator[TrackData, None, None]:
        for i in range(len(detections)):
            if detections is None:
                continue

            x1, y1, x2, y2 = map(int, detections.xyxy[i])

            if detections.confidence is None:
                logfire.warning(f"[BallTracker] No confidence for object in frame {video_item.frame_num} match id {video_item.match_id}")
                conf = 0.3
            else:
                conf = detections.confidence[i]

            track_id = 0

            yield TrackData(xyxy=(x1, y1, x2, y2), track_id=track_id, confidence=conf)

    @override
    def _save_tracks(self, detected_tracks: List[TrackData], video_item: VideoItem, object: type[SQLModel], session: Session):
        for track in detected_tracks:
            ball = BallRepository.get_ball_by_frame_num(video_item.match_id, video_item.frame_num, session)

            if ball is None:
                new_ball = BallState(
                    match_id=video_item.match_id,
                    frame_number=video_item.frame_num,
                    x1=track.xyxy[0],
                    y1=track.xyxy[1],
                    x2=track.xyxy[2],
                    y2=track.xyxy[3],
                    timestamp_ms=int(video_item.timestamp),
                    confidence=track.confidence,
                )
                session.add(new_ball)
                session.flush()
                continue
