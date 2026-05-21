from pathlib import Path
from typing import Generator, override

import logfire
from sqlmodel import Session
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
                    logfire.warning(f"[PlayerTracker] No confidence for object {object_id} in frame {frame_num}")
                    conf = 0.3
                else:
                    conf = filtered_detection.confidence[i]

                track_id = 0

                if self.type == DetectorTypes.TRACKING and hasattr(filtered_detection, "track_id"):
                    track_id = filtered_detection.track_id[i]  # type: ignore
                else:
                    logfire.error(f"[PlayerTracker] No track_id for player in frame {frame_num}")

                yield TrackData(xyxy=(x1, y1, x2, y2), track_id=track_id, confidence=conf)

    @override
    def _save_tracks(self, detected_tracks: Generator[TrackData, None, None], video_item: VideoItem, session: Session):
        for track_data in detected_tracks:
            x1, y1, x2, y2 = track_data.xyxy

            player, state = PlayerStatesRepository.get_state_by_track_id(
                frame_number=video_item.frame_num, match_id=video_item.match_id, track_id=track_data.track_id, session=session
            )

            if not player:
                new_player = PlayerModel(match_id=video_item.match_id, track_id=track_data.track_id)
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
                session.commit()

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
