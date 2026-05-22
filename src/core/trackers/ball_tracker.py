from pathlib import Path
from typing import List, override

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
