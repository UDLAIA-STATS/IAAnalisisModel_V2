from pathlib import Path
from typing import Generator, List, Type, override
import logfire
from supervision.detection.core import Detections

from sqlmodel import SQLModel, Session

from src.core.repository.goal_repository import GoalRepository
from src.core.video import goal_annotator

from src.config.routes import MODEL_GOALS_PATH
from src.entities.models.app.detector_base import DetectorBase
from src.entities.models.app.track_data import TrackData
from src.entities.models.app.video_item import VideoItem
from src.entities.models.soccer.goal_model import GoalModel
from src.entities.types.detector_types import DetectorTypes


class GoalTracker(DetectorBase):
    def __init__(self, tracker_config_file: Path | None, model: Path = MODEL_GOALS_PATH, type: DetectorTypes = DetectorTypes.DETECTION):
        super().__init__(model, tracker_config_file, type)
        self.classes = {0: goal_annotator}
        self.types_map = {0: GoalModel}

    @override
    def _save_tracks(self, detected_tracks: List[TrackData], video_item: VideoItem, object: type[SQLModel], session: Session):
        goals_added = []
        for track in detected_tracks:
            goals = GoalRepository.get_goals_by_frame_num(video_item.match_id, video_item.frame_num, session)

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

                goals_added.append(new_goal)

        session.add_all(goals_added)
        session.flush()