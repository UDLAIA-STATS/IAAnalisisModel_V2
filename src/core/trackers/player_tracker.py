from pathlib import Path
from typing import Generator, List, override

from sqlmodel import Session

from config.routes import BYTETRACK_CONFIG_PATH, PLAYER_MODEL_PATH
from core.repository.player_states_repository import PlayerStatesRepository
from entities.models.app.detector_base import DetectorBase
from entities.models.app.track_data import TrackData
from entities.models.app.video_item import VideoItem
from entities.models.soccer.player_model import PlayerModel, PlayerState
from entities.types.detector_types import DetectorTypes


class PlayerTracker(DetectorBase):

    def __init__(
            self,
            tracker_config_file: Path | None = BYTETRACK_CONFIG_PATH,
            model: Path = PLAYER_MODEL_PATH,
            type: DetectorTypes = DetectorTypes.TRACKING):
        super().__init__(model, tracker_config_file, type)

    @override
    def _save_tracks(self, detected_tracks: Generator[TrackData, None, None], video_item: VideoItem, session: Session):
        for track_data in detected_tracks:
            x1, y1, x2, y2 = track_data.xyxy

            player, state = PlayerStatesRepository.get_player_state_by_track_id(
                frame_number=video_item.frame_num,
                match_id=video_item.match_id,
                track_id=track_data.track_id,
                session=session
            )

            if not player and not state:
                new_player = PlayerModel(
                    match_id=video_item.match_id,
                    track_id=track_data.track_id
                )
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
                session.add(new_player)
                session.add(new_state)
            
            if player and not state:
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
                session.add(new_state)
