from pathlib import Path
from typing import List, Sequence, Union, override
import cv2
import logfire
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from supervision.detection.core import Detections

from sqlmodel import SQLModel, Session

from src.entities.types.bucket_types import FilePurposeTypes
from src.core.repository import files_repository, PlayerRepository
from src.config.routes import BYTETRACK_CONFIG_PATH, PLAYER_MODEL_PATH
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app import TrackData, VideoItem, DetectorBase
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.types.detector_types import DetectorTypes
from src.core.video import player_annotator
from src.core.services import ExecutorService


class PlayerTracker(DetectorBase):
    def __init__(
        self,
        tracker_config_file: Path | None = BYTETRACK_CONFIG_PATH,
        model: Path = PLAYER_MODEL_PATH,
        type: DetectorTypes = DetectorTypes.TRACKING,
    ):
        super().__init__(model, tracker_config_file, type)
        self.classes = {0: player_annotator}
        self.types_map = {0: PlayerModel}


    @override
    def __init_model__(self, model: Path, half: bool = False):
        self.model: YOLO = YOLO("yolo26x.pt")

        if model.suffix == ".pt":
            self.model.to(self.device)
            self.model.fuse()

        if half:
            self.model.half()

    @override
    def detect(self, frame) -> Sequence[Union[Results, Detections]]:
        """Detect objects in a frame."""
        tracks = self.model.track(
            frame,
            tracker=self.tracker_config_file,
            persist=True,
            conf=0.1,
            iou=0.6,
            verbose=False,
            device=self.device,
            # stream=True,
            # augment=True,
            agnostic_nms=True,
            end2end=True
        )

        # logfire.info(f"[PlayerTracker] Number of tracks: {len(list(tracks))}")
        return tracks

    @override
    def extract_detections(
        self,
        results: Sequence[Union[Results, Detections]],
        objects_ids: List[int],
        video_item: VideoItem,
    ) -> dict[int, List[TrackData]]:
        detections_map: dict[int, List[TrackData]] = {}
        detections = Detections.from_ultralytics(results[0])
        # logfire.info(f"[PlayerTracker] Number of detections: {len(detections)}")

        if len(detections) == 0:
            return {}

        for object_id in objects_ids:
            filtered_detections = detections[detections.class_id == object_id]
            annotator = self.classes[object_id]
            annotator.set_detections(filtered_detections)

            data = list(self._extract_tracks_data(filtered_detections))  # type: ignore

            if object_id not in detections_map:
                detections_map[object_id] = data
            else:
                detections_map[object_id].extend(data)

        return detections_map

    @override
    def _save_tracks(
        self,
        detected_tracks: List[TrackData],
        video_item: VideoItem,
        object: type[SQLModel],
        session: Session,
    ):
        states = []
        pending_uploads = []
        with ExecutorService() as executor:
            for track_data in detected_tracks:
                x1, y1, x2, y2 = track_data.xyxy

                player, state = PlayerStatesRepository.get_state_by_track_id(
                    frame_number=video_item.frame_num,
                    match_id=video_item.match_id,
                    track_id=track_data.track_id,
                    session=session,
                )

                if player is None:
                    player_crop = video_item.frame.copy()[
                        int(y1) : int(y2), int(x1) : int(x2)
                    ]
                    h, w = video_item.frame.shape[:2]
                    player_crop = cv2.resize(player_crop, (h // 2, w // 2))
                    image_name = f"{video_item.match_id}_{video_item.frame_num}_{track_data.track_id}"
                    key = files_repository.generate_key(
                        match_id=video_item.match_id,
                        filename=image_name,
                        purpose_type=FilePurposeTypes.PLAYER_IMAGE,
                        file_extension="png",
                    )
                    url = files_repository.get_public_url(key=key)
                    executor.submit(
                        files_repository.upload_player_image, key, player_crop
                    )
                    # files_repository.upload_player_image(key=key, crop=player_crop)

                    new_player = PlayerModel(
                        match_id=video_item.match_id,
                        track_id=track_data.track_id,
                        crop_path=url,
                    )
                    player_id = PlayerRepository.upsert_player(new_player, session)

                    new_state = PlayerState(
                        player_id=player_id,
                        frame_number=video_item.frame_num,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        timestamp=video_item.timestamp,
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
                        timestamp=video_item.timestamp,
                        confidence=track_data.confidence,
                    )
                    states.append(new_state)

            session.add_all(states)
            session.flush()
