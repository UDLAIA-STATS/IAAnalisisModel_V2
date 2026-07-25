import re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import logfire
from sqlmodel import Session
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.entities.models.soccer.player_model import PlayerNumbers
from src.core.utils.video_utils import extract_player_torso
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.app.video_item import VideoItem
from src.entities.models.number_recognizer.number_classifier import NONE_TENS


class NumberRecognizer:
    MINICPM_MODEL_NAME = "openbmb/MiniCPM-V-4.6"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self):
        self.minicpm_model, self.minicpm_processor = self._load_minicpm()

    def _load_minicpm(self):
        """
        Loads MiniCPM-Llama3-V-2_5 following the officially recommended
        pattern: fp16 on CUDA, fp32 on CPU, trust_remote_code=True.
        """
        try:
            dtype = torch.float16 if self.DEVICE == "cuda" else torch.float32

            model = AutoModelForImageTextToText.from_pretrained(
                self.MINICPM_MODEL_NAME, torch_dtype=dtype, trust_remote_code=True
            )
            model = model.to(device=self.DEVICE)
            model.eval()

            processor = AutoProcessor.from_pretrained(
                self.MINICPM_MODEL_NAME, trust_remote_code=True
            )
            return model, processor
        except Exception as e:
            logfire.fatal(f"[NumberRecognizer] MiniCPM could not be initialized: {e}")
            raise e

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

            if crop is None or crop.size == 0:
                continue

            consensus_number, consensus_conf = self._minicpm_predict(crop)

            if consensus_number is None:
                logfire.info(
                    f"[NumberRecognizer] No number detected for player "
                    f"{state.player.id} at frame {video_item.frame_num}"
                )
                continue

            if not (0 < consensus_number <= 99):
                continue

            player_number = self._build_player_number(
                player_id=state.player.id,
                frame_number=video_item.frame_num,
                consensus_number=consensus_number,
                consensus_confidence=consensus_conf,
            )
            logfire.info(
                f"[NumberRecognizer] Number Data: {player_number.model_dump()}"
            )
            predicted_numbers.append(player_number)

        session.add_all(predicted_numbers)
        session.flush()

        return predicted_numbers

    def _preprocess_for_minicpm(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        Preprocesado para MiniCPM: corrección de iluminación (CLAHE) y filtro
        bilateral para suavizar sin perder color natural.
        """
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)

        lab_clahe = cv2.merge((l_clahe, a_channel, b_channel))
        illum_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        smoothed = cv2.bilateralFilter(
            illum_corrected, d=7, sigmaColor=50, sigmaSpace=50
        )
        return smoothed

    def _minicpm_predict(
        self, crop_bgr: np.ndarray
    ) -> Tuple[Optional[int], float]:
        """Ejecuta MiniCPM una sola vez por crop."""
        if self.minicpm_model is None or self.minicpm_processor is None:
            return None, 0.0
        try:
            processed = self._preprocess_for_minicpm(crop_bgr)
            h, w = processed.shape[:2]
            scale = max(1, 256 // min(h, w))

            processed = cv2.resize(
                processed,
                (w * scale, h * scale),
                interpolation=cv2.INTER_CUBIC,
            )

            rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": """You are an OCR system. Read the player's jersey number.
                            Return ONLY the jersey number.
                            Rules:
                            - Output digits only.
                            - Do not explain.
                            - Do not output spaces.
                            - If uncertain return None.""",
                        },
                    ],
                }
            ]

            prompt = self.minicpm_processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )

            inputs = self.minicpm_processor(
                images=[image],
                text=prompt,
                return_tensors="pt",
            )

            inputs = {
                k: v.to(self.minicpm_model.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

            output = self.minicpm_model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
            )

            generated_ids = output[:, inputs["input_ids"].shape[1]:]

            answer = self.minicpm_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            logfire.info(f"[NumberRecognizer] MiniCPM answer: {answer}")
            numb, conf = self._parse_response(answer)
            logfire.info(f"[NumberRecognizer] Parsed MiniCPM prediction: {numb} ({conf})")

            return numb, conf
        except Exception as e:
            logfire.warning(f"[NumberRecognizer] MiniCPM prediction failed: {e}")
            return None, 0.0

    def _parse_response(self, response: str) -> Tuple[Optional[int], float]:
        """
        MiniCPM devuelve texto libre, no una confianza numérica.
        Derivamos una confianza heurística basada en la limpieza de la respuesta.
        """
        if not response:
            return None, 0.0

        text = response.strip().upper()
        if "NONE" in text:
            return None, 0.0

        digits = re.sub(r"\D", "", text)
        if not digits:
            return None, 0.0

        clean_ratio = len(digits) / max(len(text.replace(" ", "")), 1)
        length_penalty = (
            1.0 if len(digits) <= 2 else max(0.3, 1.0 - 0.2 * (len(digits) - 2))
        )
        confidence = max(0.05, min(0.95, clean_ratio * length_penalty))

        if len(digits) > 2:
            digits = digits[-2:]

        return int(digits), float(confidence)

    def _build_player_number(
        self,
        player_id: int,
        frame_number: int,
        consensus_number: int,
        consensus_confidence: float,
    ) -> PlayerNumbers:
        """
        Construye la entidad PlayerNumbers a partir de la única predicción
        de MiniCPM. visible_prob se iguala a la confianza de la predicción.
        """
        if consensus_number < 10:
            tens_pred = NONE_TENS
            units_pred = consensus_number
        else:
            tens_pred = consensus_number // 10
            units_pred = consensus_number % 10

        return PlayerNumbers(
            player_id=player_id,
            frame_number=frame_number,
            number=consensus_number,
            confidence=consensus_confidence,
            visible_prob=consensus_confidence,
            tens_none_prob=1.0 if tens_pred == NONE_TENS else 0.0,
            tens_pred=tens_pred,
            tens_prob=1.0,
            units_pred=units_pred,
            units_prob=1.0,
        )


number_predictor = NumberRecognizer()