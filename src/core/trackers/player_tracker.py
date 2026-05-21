from pathlib import Path
from typing import Generator, List, Type, override

import logfire
from matplotlib.pylab import f
from sqlmodel import SQLModel, Session
from supervision.detection.core import Detections

from src.config.routes import BYTETRACK_CONFIG_PATH, PLAYER_MODEL_PATH
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.detector_base import DetectorBase
from src.entities.models.app.track_data import TrackData
from src.entities.models.app.video_item import VideoItem
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.types.detector_types import DetectorTypes
from src.core.video import player_annotator


class PlayerTracker(DetectorBase):
    def __init__(
        self, tracker_config_file: Path | None = BYTETRACK_CONFIG_PATH, model: Path = PLAYER_MODEL_PATH, type: DetectorTypes = DetectorTypes.TRACKING
    ):
        super().__init__(model, tracker_config_file, type)
        self.classes = {0: player_annotator}
        self.types_map = {0: PlayerModel}

    @override
    def _save_tracks(self, detected_tracks: List[TrackData], video_item: VideoItem, object: type[SQLModel], session: Session):
        states = []
        for track_data in detected_tracks:
            x1, y1, x2, y2 = track_data.xyxy

            player, state = PlayerStatesRepository.get_state_by_track_id(
                frame_number=video_item.frame_num, match_id=video_item.match_id, track_id=track_data.track_id, session=session
            )

            if player is None:
                new_player = PlayerModel(match_id=video_item.match_id, track_id=track_data.track_id)
                session.add(new_player)
                session.flush()
                new_state = PlayerState(
                    player_id=new_player.id,
                    frame_number=video_item.frame_num,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    timestamp_ms=int(video_item.timestamp),
                    confidence=track_data.confidence,
                )
                states.append(new_state)

            if player and state is None:
                new_state = PlayerState(
                    player_id=player.id,
                    frame_number=video_item.frame_num,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    timestamp_ms=int(video_item.timestamp),
                    confidence=track_data.confidence,
                )
                states.append(new_state)

        session.add_all(states)
        session.flush()
