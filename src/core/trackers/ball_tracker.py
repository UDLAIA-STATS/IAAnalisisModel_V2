from pathlib import Path
from typing import Generator, List, override

from sqlmodel import Session

from core.repository.ball_repository import BallRepository
from entities.models.app.detector_base import DetectorBase
from entities.models.app.track_data import TrackData
from entities.models.app.video_item import VideoItem
from entities.models.soccer.ball_model import BallState
from entities.types.detector_types import DetectorTypes


class BallTracker(DetectorBase):
    def __init__(self, model: Path, tracker_config_file: Path, type: DetectorTypes = DetectorTypes.DETECTION):
        super().__init__(model, tracker_config_file, type)

    @override
    def _save_tracks(self, detected_tracks: Generator[TrackData, None, None], video_item: VideoItem, session: Session):
        for track in detected_tracks:
            ball = BallRepository.get_ball_by_frame_num(
                video_item.match_id, video_item.frame_num, session)
            
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
