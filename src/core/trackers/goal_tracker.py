from pathlib import Path
from typing import Generator, List, override

from sqlmodel import Session

from config.routes import MODEL_GOALS_PATH
from core.repository.goal_repository import GoalRepository
from entities.models.app.detector_base import DetectorBase
from entities.models.app.track_data import TrackData
from entities.models.app.video_item import VideoItem
from entities.models.soccer.goal_model import GoalModel
from entities.types.detector_types import DetectorTypes


class GoalTracker(DetectorBase):
    def __init__(
            self,
            tracker_config_file: Path | None,
            model: Path = MODEL_GOALS_PATH,
            type: DetectorTypes = DetectorTypes.DETECTION):
        super().__init__(model, tracker_config_file, type)

    @override
    def _save_tracks(self, detected_tracks: Generator[TrackData, None, None], video_item: VideoItem, session: Session):
        for track in detected_tracks:
            goals = GoalRepository.get_goals_by_frame_num(
                video_item.match_id, video_item.frame_num, session)
            
            if goals is None:
                new_goal = GoalModel(
                    match_id=video_item.match_id,
                    frame_number=video_item.frame_num,
                    x1=track.xyxy[0],
                    y1=track.xyxy[1],
                    x2=track.xyxy[2],
                    y2=track.xyxy[3],
                    timestamp_ms=int(video_item.timestamp),
                    confidence=track.confidence,
                )

                session.add(new_goal)
