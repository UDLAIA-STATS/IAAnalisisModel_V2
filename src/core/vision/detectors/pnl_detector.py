# src/core/vision/detectors/pnl_detector.py

import yaml
import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from src.config.routes import PNL_CALIB_DIR, WEIGHTS_DIR
from src.entities.models.homography.homography_models import FeatureDetectorBase

from addons.PnLCalib.model.cls_hrnet import get_cls_net

class PnLFeatureDetector(FeatureDetectorBase):
    """
    Detector de keypoints (58) y líneas (23) usando HRNet.
    """
    LINE_MAPPING = {
        1: (9, 8),    # Área penal izquierda - fondo
        2: (7, 8),    # Área penal izquierda - lateral
        3: (6, 7),    # Área penal izquierda - superior
        4: (13, 12),  # Área penal derecha - fondo
        5: (11, 12),  # Área penal derecha - lateral
        6: (10, 11),  # Área penal derecha - superior
        # 7-12: Omitir (porterías 3D)
        13: (4, 5),   # Línea de medio campo
        14: (3, 2),   # Línea de banda inferior
        15: (0, 3),   # Línea de fondo izquierda
        16: (1, 2),   # Línea de fondo derecha
        17: (0, 1),   # Línea de banda superior
        18: (17, 16), # Área pequeña izquierda - fondo
        19: (15, 16), # Área pequeña izquierda - lateral
        20: (14, 15), # Área pequeña izquierda - superior
        21: (21, 20), # Área pequeña derecha - fondo
        22: (19, 20), # Área pequeña derecha - lateral
        23: (18, 19), # Área pequeña derecha - superior
    }


    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        image_size: tuple = (960, 540),
        confidence_threshold: float = 0.3,
    ):
        self.device = device
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.kp_model: torch.nn.Module
        self.lines_model: torch.nn.Module
        self._load_models()

    def _load_models(self) -> None:
        """Carga los modelos y pesos."""

        config_kp_path = PNL_CALIB_DIR / "config" / "hrnetv2_w48.yaml"
        config_lines_path = PNL_CALIB_DIR / "config" / "hrnetv2_w48_l.yaml"

        with open(config_kp_path, 'r') as f:
            cfg_kp = yaml.safe_load(f)
        with open(config_lines_path, 'r') as f:
            cfg_lines = yaml.safe_load(f)

        # Pesos
        kp_weights = WEIGHTS_DIR / "SV_kp"
        lines_weights = WEIGHTS_DIR / "SV_lines"

        # Modelos
        self.kp_model = get_cls_net(cfg_kp, pretrained=kp_weights.as_posix())
        self.lines_model = get_cls_net(cfg_lines, pretrained=lines_weights.as_posix())

        self.kp_model.to(self.device)
        self.lines_model.to(self.device)
        self.kp_model.eval()
        self.lines_model.eval()

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocesa la imagen."""
        resized = cv2.resize(frame, self.image_size, interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype(np.float32) / 255.0
        tensor = torch.from_numpy(normalized).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    def _postprocess_keypoints(self, heatmaps: torch.Tensor, original_shape: tuple) -> Dict[int, Dict[str, float]]:
        """
        Convierte mapas de calor (1, 58, H, W) a coordenadas (x,y) en píxeles.
        Factor de escala = 4 (HRNet downsample 1/4).
        """
        h_orig, w_orig = original_shape[:2]
        heatmaps_np = heatmaps.detach().cpu().numpy()[0]  # (58, H, W)
        scale_x = w_orig / self.image_size[0]
        scale_y = h_orig / self.image_size[1]
        kp_dict = {}
        for kp_id in range(1, 59):  # IDs 1..58
            hm = heatmaps_np[kp_id - 1]
            max_val = np.max(hm)
            if max_val < self.confidence_threshold:
                # Si la confianza es baja, omitimos el keypoint (o lo dejamos con valor por defecto)
                # Logfire.debug(f"Keypoint {kp_id} descartado por baja confianza: {max_val:.3f}")
                continue
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            # Factor 4 porque HRNet downsample 1/4
            x_px = (x * 4) * scale_x
            y_px = (y * 4) * scale_y
            kp_dict[kp_id] = {'x': float(x_px), 'y': float(y_px)}
        return kp_dict

    def _postprocess_lines(self, heatmaps: torch.Tensor, original_shape: tuple) -> Dict[int, Dict[str, float]]:
        """
        Convierte mapas de calor de líneas (1, 24, H, W) a segmentos (x1,y1,x2,y2).
        Factor de escala = 4 (HRNet downsample 1/4).
        """
        h_orig, w_orig = original_shape[:2]
        heatmaps_np = heatmaps.detach().cpu().numpy()[0]  # (24, H, W)
        scale_x = w_orig / self.image_size[0]
        scale_y = h_orig / self.image_size[1]
        lines_dict = {}

        # 1. Extraer los 24 puntos
        points_24 = {}
        for idx in range(24):  # 0..23
            hm = heatmaps_np[idx]
            max_val = np.max(hm)
            if max_val < self.confidence_threshold:
                continue
            y, x = np.unravel_index(np.argmax(hm), hm.shape)
            x_px = (x * 4) * scale_x
            y_px = (y * 4) * scale_y
            points_24[idx] = (float(x_px), float(y_px))

        # 2. Mapear a líneas (excepto líneas 7-12)
        for line_id, (idx1, idx2) in self.LINE_MAPPING.items():
            if idx1 in points_24 and idx2 in points_24:
                x1, y1 = points_24[idx1]
                x2, y2 = points_24[idx2]
                lines_dict[line_id] = {
                    'x_1': x1,
                    'y_1': y1,
                    'x_2': x2,
                    'y_2': y2
                }
            else:
                # Si falta un punto, omitimos la línea
                # Logfire.debug(f"Línea {line_id} omitida por falta de puntos")
                pass

        return lines_dict

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detecta keypoints y líneas.
        """
        h, w = frame.shape[:2]
        input_tensor = self._preprocess(frame)

        with torch.no_grad():
            kp_heatmaps = self.kp_model(input_tensor)      # (1, 58, H, W)
            lines_heatmaps = self.lines_model(input_tensor)  # (1, 24, H, W)

        kp_dict = self._postprocess_keypoints(kp_heatmaps, (h, w))
        lines_dict = self._postprocess_lines(lines_heatmaps, (h, w))

        return {
            'kp_dict': kp_dict,
            'lines_dict': lines_dict,
            'image_width': w,
            'image_height': h,
            'confidence': 1.0  # se puede calcular como media de máximos
        }