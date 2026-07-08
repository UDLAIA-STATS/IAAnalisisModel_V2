from typing import List
import uuid

import cv2
import numpy as np
from sqlmodel import Session

from src.entities.models.soccer.player_model import PlayerNumbers
from src.core.utils.video_utils import extract_player_torso
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.video_item import VideoItem
from src.entities.models.number_recognizer.number_classifier import (
    JerseyNumberClassifier,
)
from src.config.routes import NUMBER_MODEL_PATH, OUTPUT_IMAGES_DIR


class NumberRecognizer:
    JERSEY_IMG_SIZE = (128, 80)
    JERSEY_BACKBONE = "resnet18"

    def __init__(self):
        self.classifier = JerseyNumberClassifier(
            NUMBER_MODEL_PATH, self.JERSEY_IMG_SIZE, self.JERSEY_BACKBONE
        )

    def predict(self, match_id: int, video_item: VideoItem, session: Session) -> List[PlayerNumbers]:
        states = PlayerStatesRepository.get_states_by_frame(
            match_id, video_item.frame_num, session
        )
        predicted_numbers = []

        for state in states:
            bbox = state.x1, state.y1, state.x2, state.y2
            crop = extract_player_torso(video_item.frame, np.array(bbox))
            parent_dir = OUTPUT_IMAGES_DIR / str(video_item.match_id)
            parent_dir.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(parent_dir / f"{uuid.uuid4()}.jpg", crop)
            player_number = self.classifier.predict(state.player.id, video_item.frame_num, crop)
            predicted_numbers.append(player_number)
        
        session.add_all(predicted_numbers)
        session.flush()

        return predicted_numbers


number_predictor = NumberRecognizer()
