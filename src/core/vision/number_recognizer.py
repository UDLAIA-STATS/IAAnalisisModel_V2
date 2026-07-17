import re
from typing import List, Optional, Tuple
import uuid

from easyocr import easyocr
import logfire
import torch

from addons.sam3.sam3.model_builder import build_sam3_image_model
from addons.sam3.sam3.model.sam3_image_processor import Sam3Processor

import cv2
import numpy as np
from sqlmodel import Session
from PIL import Image

from src.entities.models.soccer.player_model import PlayerNumbers
from src.core.utils.video_utils import extract_player_torso
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.video_item import VideoItem
from src.entities.models.number_recognizer.number_classifier import (
    JerseyNumberClassifier,
    NONE_TENS,
)
from src.config.routes import NUMBER_MODEL_PATH, OUTPUT_IMAGES_DIR


class NumberRecognizer:
    JERSEY_IMG_SIZE = (128, 80)
    JERSEY_BACKBONE = "resnet18"

    def __init__(self):
        self.classifier = JerseyNumberClassifier(
            NUMBER_MODEL_PATH, self.JERSEY_IMG_SIZE, self.JERSEY_BACKBONE
        )

        self.sam3_model = None
        self.sam3_processor = None

        try:
            self.sam3_model = build_sam3_image_model()
            self.sam3_model = self.sam3_model.float()
            self.sam3_processor = Sam3Processor(
                self.sam3_model,
                device="cuda" if torch.cuda.is_available() else "cpu",
                confidence_threshold=0.3,
            )
            logfire.info(
                f"SAM3 device type: {next(self.sam3_model.parameters()).dtype}"
            )
        except Exception as e:
            logfire.warning(f"Warning: SAM3 model could not be loaded: {e}")
            self.sam3_model = None
            self.sam3_processor = None

        try:
            self.ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        except Exception as e:
            logfire.warning(f"Warning: EasyOCR could not be initialized: {e}")
            self.ocr_reader = None

    def predict(
        self, match_id: int, video_item: VideoItem, session: Session
    ) -> List[PlayerNumbers]:
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
            player_number = self.classifier.predict(
                state.player.id, video_item.frame_num, crop
            )

            if (
                player_number.visible_prob > 0.75
                and player_number.confidence < 0.5
                and crop is not None
                and crop.size > 0
            ):
                sam3_number, sam3_conf = self._sam3_predict(crop)
                if sam3_number is not None:
                    player_number = self._build_player_number_from_sam3(
                        player_number, sam3_number, sam3_conf
                    )
                    logfire.info(f"[NumberRecognizer] SAM3: {player_number}")

            predicted_numbers.append(player_number)

        session.add_all(predicted_numbers)
        session.flush()

        return predicted_numbers

    def _sam3_predict(self, crop_bgr: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Use SAM3 to locate the jersey number region, then OCR that region.
        Returns (number, confidence) or (None, 0.0).
        """
        if self.sam3_model is None or self.sam3_processor is None:
            return None, 0.0

        try:
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(crop_rgb)

            with torch.no_grad(), torch.inference_mode(), torch.autocast(
                enabled=False,
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float32,
            ):
                inference_state = self.sam3_processor.set_image(pil_img)
                # logfire.info(f" SAM3 inference state keys: {inference_state.keys()}")
                output = self.sam3_processor.set_text_prompt(
                    state=inference_state, prompt="jersey number"
                )

            boxes = output.get("boxes")
            scores = output.get("scores")

            if boxes is None:
                logfire.warning("[SAM3] output sin key 'boxes'")
                return None, 0.0

            if isinstance(boxes, torch.Tensor):
                if boxes.numel() == 0:
                    logfire.warning(
                        "[SAM3] boxes vacío (tensor), sin detecciones para 'jersey number'"
                    )
                    return None, 0.0
            else:
                if len(boxes) == 0:
                    logfire.warning(
                        "[SAM3] boxes vacío (lista), sin detecciones para 'jersey number'"
                    )
                    return None, 0.0

            if scores is None:
                logfire.warning("[SAM3] output sin key 'scores'")
                return None, 0.0

            scores = scores.detach().to(dtype=torch.float32, device="cpu")
            boxes = boxes.detach().to(dtype=torch.float32, device="cpu")

            logfire.info(
                f"[SAM3] {len(boxes)} detecciones, mejor score: {float(scores.max()) if isinstance(scores, torch.Tensor) else max(scores)}"
            )

            best_idx = int(np.argmax(scores).item())
            best_box = boxes[best_idx]
            sam3_score = float(scores[best_idx].item())

            x1, y1, x2, y2 = map(int, best_box.tolist())
            h, w = crop_bgr.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None, 0.0

            number_roi = crop_bgr[y1:y2, x1:x2]
            if number_roi.size == 0:
                return None, 0.0

            ocr_number, ocr_conf = self._ocr_digits(number_roi)
            logfire.info(f"[SAM3] OCR: {ocr_number} ({ocr_conf})")
            if ocr_number is None:
                return None, 0.0

            combined_conf = sam3_score * ocr_conf
            return ocr_number, combined_conf

        except Exception as e:
            print(f"SAM3 fallback failed: {e}")
            return None, 0.0

    def _ocr_digits(self, image_np: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Use EasyOCR to extract digits from a cropped image region.
        Returns (number, confidence) or (None, 0.0).
        """
        if self.ocr_reader is None:
            return None, 0.0
        try:
            # EasyOCR expects RGB image; our crop is BGR from OpenCV
            rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            results = self.ocr_reader.readtext(rgb, allowlist="0123456789", detail=1)
            if not results:
                return None, 0.0
            # Take the first detection with highest confidence
            best_text, best_conf = None, 0.0
            for bbox, text, conf in results:
                digits = re.sub(r"\D", "", text)
                if digits and conf > best_conf:
                    best_text = digits
                    best_conf = conf
            if best_text:
                return int(best_text), best_conf
            return None, 0.0
        except Exception as e:
            print(f"OCR failed: {e}")
            return None, 0.0

    def _build_player_number_from_sam3(
        self, original: PlayerNumbers, sam3_number: int, sam3_confidence: float
    ) -> PlayerNumbers:
        """Create a new PlayerNumbers object based on SAM3 prediction."""
        # Determine tens and units
        if sam3_number < 10:
            tens_pred = NONE_TENS
            units_pred = sam3_number
        else:
            tens_pred = sam3_number // 10
            units_pred = sam3_number % 10

        return PlayerNumbers(
            player_id=original.player_id,
            frame_number=original.frame_number,
            number=sam3_number,
            confidence=sam3_confidence,
            visible_prob=original.visible_prob,  # keep visibility from classifier
            tens_none_prob=1.0 if tens_pred == NONE_TENS else 0.0,
            tens_pred=tens_pred,
            tens_prob=1.0,
            units_pred=units_pred,
            units_prob=1.0,
        )


number_predictor = NumberRecognizer()
