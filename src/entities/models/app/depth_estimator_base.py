import math
import cv2
import logfire
import numpy as np
import torch

from typing import List, Optional
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.interpolate import RectBivariateSpline

from src.entities.types.depth_model_types import DepthModelsTypes
from src.config.routes import DEPTH_MODEL_PATH


class DepthEstimatorBase:
    """
    Calculadora de profundidad usando MiDaS + MediaPipe Pose (API Tasks).
    Calcula profundidad cada n_frames * 10 (cada ~10 segundos) o cuando cambia la escala de camara.
    """

    def __init__(
        self,
        device: str,
        frame_rate: float = 30.0,
        depth_model_type: DepthModelsTypes = DepthModelsTypes.DPT_Hybrid,
        alpha: float = 0.2,
        depth_scale: float = 1.0,
        camera_scale: float = 1.0,
        depth_calculation_interval_seconds: float = 10.0,
    ):

        if depth_model_type not in DepthModelsTypes.to_dict().values():
            raise ValueError(
                f"Tipo de modelo de profundidad no reconocido: {depth_model_type}"
            )

        self.alpha = alpha
        self.depth_scale = depth_scale
        self.camera_scale = camera_scale
        self.device = device
        self.frame_rate = frame_rate
        self.depth_calculation_interval_seconds = depth_calculation_interval_seconds

        self.frames_per_depth_calc = int(
            frame_rate * depth_calculation_interval_seconds
        )
        self.last_calculation_frame = -1
        self.last_camera_scale = camera_scale

        logfire.info(
            f"[DepthCalculator] Inicializado: calculo cada {self.frames_per_depth_calc} frames "
            f"({depth_calculation_interval_seconds}s), escala inicial={camera_scale}"
        )

        self.previous_depth = 0
        self.current_depth = 0

        logfire.info(f"[DepthCalculator] Cargando modelo MiDaS: {depth_model_type}")
        self.midas = torch.hub.load("intel-isl/MiDaS", depth_model_type.value)
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.midas.to(device)

        if (
            depth_model_type == DepthModelsTypes.DPT_Large
            or depth_model_type == DepthModelsTypes.DPT_Hybrid
        ):
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

        logfire.info("[DepthCalculator] Inicializando Pose Landmarker (API Tasks)")
        self._init_pose_landmarker()

        logfire.info("[DepthCalculator] Modelos cargados correctamente")

    def _init_pose_landmarker(self):
        """Inicializa el Pose Landmarker usando la API Tasks de MediaPipe."""
        model_path = DEPTH_MODEL_PATH

        base_options = python.BaseOptions(model_asset_path=model_path.as_posix())

        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    def should_calculate_depth(
        self, frame_num: int, current_camera_scale: float
    ) -> bool:
        """
        Determina si se debe calcular la profundidad.

        Se calcula cuando:
        1. Han pasado suficientes frames (cada n_frames * 10 segundos)
        2. La escala de la camara ha cambiado significativamente (>5%)
        """
        if self.last_calculation_frame < 0:
            logfire.debug(f"[DepthCalculator] Primer frame, calculando profundidad")
            return True

        scale_change = (
            abs(current_camera_scale - self.last_camera_scale) / self.last_camera_scale
            if self.last_camera_scale != 0
            else 0
        )
        if scale_change > 0.05:
            logfire.info(
                f"[DepthCalculator] Cambio de escala detectado: "
                f"{self.last_camera_scale:.3f} -> {current_camera_scale:.3f} "
                f"({scale_change*100:.1f}%), recalculando profundidad"
            )
            return True

        frames_since_last = frame_num - self.last_calculation_frame
        if frames_since_last >= self.frames_per_depth_calc:
            logfire.debug(
                f"[DepthCalculator] Intervalo alcanzado: {frames_since_last} frames "
                f"desde ultimo calculo"
            )
            return True

        return False

    def _get_waist_center(self, roi_frame: np.ndarray) -> Optional[tuple[float, float]]:
        """
        Detecta landmarks de pose y retorna el centro de la cintura (hips).

        Args:
            roi_frame: Frame de la region del jugador (BGR)

        Returns:
            (x_norm, y_norm) coordenadas normalizadas [0,1] del centro de cintura,
            o None si no se detecto pose.
        """
        rgb_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.pose_landmarker.detect(mp_image)

        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]

        LEFT_HIP = 23
        RIGHT_HIP = 24

        try:
            left_hip = landmarks[LEFT_HIP]
            right_hip = landmarks[RIGHT_HIP]

            mid_x = (left_hip.x + right_hip.x) / 2.0
            mid_y = (left_hip.y + right_hip.y) / 2.0

            return (mid_x, mid_y)
        except (IndexError, AttributeError):
            logfire.warning("[DepthCalculator] No se encontraron landmarks de cadera")
            return None

    def _calculate_depth_at_point(self, frame: np.ndarray, x: int, y: int) -> float:
        """Calcula profundidad en un punto especifico del frame usando MiDaS."""
        output_norm = self._process_frame_midas(frame)
        h, w = output_norm.shape

        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))

        x_grid = np.arange(w)
        y_grid = np.arange(h)

        spline = RectBivariateSpline(y_grid, x_grid, output_norm)
        depth_value = spline(y, x).item()
        logfire.debug(
            f"[DepthCalculator] Profundidad en ({x}, {y}): {depth_value}, tipo de valor {type(depth_value)}"
        )
        logfire.debug(
            f"[DepthCalculator] Escala de profundidad: {self.depth_scale}, tipo de valor {type(self.depth_scale)}"
        )

        depth_midas = self._depth_to_distance(depth_value, self.depth_scale)
        self.current_depth = depth_midas
        filtered_depth = self._apply_ema_filter()

        final_depth = (filtered_depth / 10.0) / self.camera_scale

        return float(final_depth)

    def _process_frame_midas(self, frame: np.ndarray) -> np.ndarray:
        """Procesa un frame completo con MiDaS."""
        input_batch = self.transform(frame).to(self.device)
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        output_norm = cv2.normalize(
            output, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        return output_norm

    def _apply_ema_filter(self) -> float:
        """Aplica filtro de media movil exponencial."""
        filtered_depth = (
            self.alpha * self.current_depth + (1 - self.alpha) * self.previous_depth
        )
        if self.previous_depth == filtered_depth:
            return self.previous_depth
        self.previous_depth = filtered_depth
        return filtered_depth

    def _depth_to_distance(self, depth_value: float, depth_scale: float) -> float:
        """Convierte valor de profundidad inversa a distancia en metros."""
        return math.fabs(1.0 / (depth_value * depth_scale))

    def process_player_depth(
        self,
        frame: np.ndarray,
        bbox: List[int],
        frame_num: int,
        current_camera_scale: float,
    ) -> Optional[float]:
        pass