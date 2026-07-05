from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.entities.models.number_recognizer.number_net import JerseyNumberNet
from src.entities.models.soccer.player_model import PlayerNumbers

NONE_TENS = 10
NUM_TENS_CLASSES = 11
NUM_UNITS_CLASSES = 10

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class JerseyNumberClassifier:

    def __init__(
        self,
        checkpoint_path: Path,
        img_size: Tuple[int, int] = (128, 80),
        backbone: str = "resnet18",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_size = img_size

        self.model = JerseyNumberNet(backbone).to(self.device)
        ckpt = torch.load(
            checkpoint_path.as_posix(), map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model = self.model.to(memory_format=torch.channels_last)

        self._normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        # cv2.resize takes (width, height); self.img_size is (H, W) to match train.py's convention.
        resized = cv2.resize(
            crop_rgb,
            (self.img_size[1], self.img_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = self._normalize(tensor)
        return tensor.unsqueeze(0)

    @staticmethod
    def _illegible_result(player_id: int, frame_number: int) -> PlayerNumbers:
        return PlayerNumbers(
            player_id=player_id,
            frame_number=frame_number,
            confidence=0.0,
            tens_none_prob=0.0,
            visible_prob=0.0,
            tens_pred=NONE_TENS,
            tens_prob=0.0,
            units_pred=0,
            units_prob=0.0,
        )

    @torch.no_grad()
    def predict(self, player_id: int, frame_number: int, crop_bgr: Optional[np.ndarray]) -> PlayerNumbers:
        if crop_bgr is None or crop_bgr.size == 0:
            return self._illegible_result(player_id, frame_number)

        tensor = self._preprocess(crop_bgr).to(
            self.device, memory_format=torch.channels_last
        )
        visible_logits, tens_logits, units_logits = self.model(tensor)

        visible_p = F.softmax(visible_logits[0], dim=-1)
        tens_p = F.softmax(tens_logits[0], dim=-1)
        units_p = F.softmax(units_logits[0], dim=-1)

        visible_pred = int(visible_p.argmax())
        tens_pred = int(tens_p.argmax())
        units_pred = int(units_p.argmax())
        tens_none_prob = float(tens_p[NONE_TENS])

        if visible_pred == 0:
            return PlayerNumbers(
                player_id=player_id,
                frame_number=frame_number,
                confidence=float(visible_p[0]),
                visible_prob=float(visible_p[0]),
                tens_none_prob=tens_none_prob,
                tens_pred=tens_pred,
                tens_prob=float(tens_p[tens_pred]),
                units_pred=units_pred,
                units_prob=float(units_p[units_pred]),
            )

        number = units_pred if tens_pred == NONE_TENS else tens_pred * 10 + units_pred
        confidence = float(visible_p[1] * tens_p[tens_pred] * units_p[units_pred])
        return PlayerNumbers(
            player_id=player_id,
            frame_number=frame_number,
            number=number,
            confidence=confidence,
            tens_none_prob=tens_none_prob,
            visible_prob=float(visible_p[1]),
            tens_pred=tens_pred,
            tens_prob=float(tens_p[tens_pred]),
            units_pred=units_pred,
            units_prob=float(units_p[units_pred]),
        )
