from pathlib import Path
from typing import Generator, override
from supervision.detection.core import Detections

import logfire
from sqlmodel import Session

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
    def _extract_tracks_data(
        self, filtered_detections: dict[int, Detections], frame_num: int, video_item: VideoItem
    ) -> Generator[TrackData, None, None]:
        for object_id, filtered_detection in filtered_detections.items():
            annotator = self.classes[object_id]
            annotator.set_detections(filtered_detection)

            for i in range(len(filtered_detection)):
                if filtered_detection is None:
                    continue

                x1, y1, x2, y2 = map(int, filtered_detection.xyxy[i])

                if filtered_detection.confidence is None:
                    logfire.warning(f"[BallDetector] No confidence for object {object_id} in frame {frame_num}")
                    conf = 0.3
                else:
                    conf = filtered_detection.confidence[i]

                track_id = 0

                if self.type == DetectorTypes.TRACKING and hasattr(filtered_detection, "track_id"):
                    track_id = filtered_detection.track_id[i]  # type: ignore

                video_item.annotated_frame = annotator.annotate(
                    annotated_frame=video_item.annotated_frame, detections=filtered_detection, label=f"Conf: {conf}"
                )

                yield TrackData(xyxy=(x1, y1, x2, y2), track_id=track_id, confidence=conf)

    @override
    def _save_tracks(self, detected_tracks: Generator[TrackData, None, None], video_item: VideoItem, session: Session):
        for track in detected_tracks:
            ball = BallRepository.get_ball_by_frame_num(video_item.match_id, video_item.frame_num, session)

            if not ball:
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
