from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence, Union, override
import logfire
import numpy as np
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

from sqlmodel import SQLModel, Session

from src.entities.types.bucket_types import FilePurposeTypes
from src.core.repository.r2_repository import files_repository 
from src.core.utils.video_utils import extract_player_torso
from src.core.repository.player_repository import PlayerRepository
from src.config.routes import BYTETRACK_CONFIG_PATH, PLAYER_MODEL_PATH
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.detector_base import DetectorBase
from src.entities.models.app.track_data import TrackData
from src.entities.models.app.video_item import VideoItem
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.types.detector_types import DetectorTypes
from src.core.video import player_annotator

executor = ThreadPoolExecutor(max_workers=10)

class PlayerTracker(DetectorBase):
    def __init__(
        self, tracker_config_file: Path | None = BYTETRACK_CONFIG_PATH, model: Path = PLAYER_MODEL_PATH, type: DetectorTypes = DetectorTypes.TRACKING
    ):
        super().__init__(model, tracker_config_file, type)
        self.classes = {0: player_annotator}
        self.types_map = {0: PlayerModel}

    @override
    def detect(self, frame) -> Sequence[Union[Results, Detections]]:
        """Detect objects in a frame."""
        return self.model.track(
                frame,
                tracker=self.tracker_config_file,
                persist=True,
                conf=0.15,
                iou=0.55,
                verbose=False,
                device=self.device)

    @override
    def extract_detections(
        self, results: Sequence[Union[Results, Detections]], objects_ids: List[int], video_item: VideoItem
    ) -> dict[int, List[TrackData]]:
        detections_map: dict[int, List[TrackData]] = {}
        detections = Detections.from_ultralytics(results[0])
        logfire.info(f"[PlayerTracker] Number of detections: {len(detections)}")

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            annotator = self.classes[object_id]
            annotator.set_detections(filtered_detections)

            data = list(self._extract_tracks_data(filtered_detections, video_item))  # type: ignore

            if object_id not in detections_map:
                detections_map[object_id] = data
            else:
                detections_map[object_id].extend(data)

        return detections_map

    @override
    def _save_tracks(self, detected_tracks: List[TrackData], video_item: VideoItem, object: type[SQLModel], session: Session):
        states = []
        pending_uploads = []
        for track_data in detected_tracks:
            x1, y1, x2, y2 = track_data.xyxy

            player, state = PlayerStatesRepository.get_state_by_track_id(
                frame_number=video_item.frame_num, match_id=video_item.match_id, track_id=track_data.track_id, session=session
            )

            if player is None:
                player_crop = video_item.frame.copy()[int(y1) : int(y2), int(x1) : int(x2)]
                image_name = f"{video_item.match_id}_{video_item.frame_num}_{track_data.track_id}"
                key = files_repository.generate_key(
                    match_id=video_item.match_id,
                    filename=image_name,
                    purpose_type=FilePurposeTypes.PLAYER_IMAGE,
                    file_extension="png")
                url = files_repository.get_public_url(key=key)
                future = executor.submit(files_repository.upload_player_image, key, player_crop)
                pending_uploads.append(future)
                # files_repository.upload_player_image(key=key, crop=player_crop)

                new_player = PlayerModel(match_id=video_item.match_id, track_id=track_data.track_id, crop_path=url)
                player_id  = PlayerRepository.upsert_player(new_player, session)

                new_state = PlayerState(
                    player_id=player_id,
                    frame_number=video_item.frame_num,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    timestamp=int(video_item.timestamp),
                    confidence=track_data.confidence,
                )
                states.append(new_state)
                continue


            if player and state is None:
                new_state = PlayerState(
                    player_id=player.id,
                    frame_number=video_item.frame_num,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    timestamp=int(video_item.timestamp),
                    confidence=track_data.confidence,
                )
                states.append(new_state)

        session.add_all(states)
        session.flush()

        for future in pending_uploads:
            future.result()
