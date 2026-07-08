"""
Calibrador basado en FramebyFrameCalib de PnLCalib.
Optimiza la calibración usando keypoints y líneas.
"""

import numpy as np
from typing import Dict, Optional


from addons.PnLCalib.utils.utils_calib_wp import FramebyFrameCalib, rotation_matrix_to_pan_tilt_roll

from src.entities.models.homography.homography_models import CalibratorBase, CalibrationResult
from src.entities.utils.homography_utils import _reprojection_error
import logfire


class PnLCalibrator(CalibratorBase):
    """
    Calibrador que utiliza FramebyFrameCalib para estimar la homografía.
    """

    def __init__(
        self,
        intrinsics: Optional[np.ndarray] = None,
        distortion: Optional[np.ndarray] = None,
        image_width: int = 1920,
        image_height: int = 1080,
        alpha: float = 0.7,  # peso para líneas vs puntos
        denormalize: bool = False,
    ):
        self.image_width = image_width
        self.image_height = image_height
        self.alpha = alpha
        self.denormalize = denormalize

        # Intrínsecos por defecto (estimación inicial)
        if intrinsics is None:
            self.intrinsics = np.array([
                [1000.0, 0.0, image_width / 2],
                [0.0, 1000.0, image_height / 2],
                [0.0, 0.0, 1.0]
            ], dtype=np.float32)
        else:
            self.intrinsics = intrinsics

        if distortion is None:
            self.distortion = np.zeros(5, dtype=np.float32)
        else:
            self.distortion = distortion

        self.calibrator = FramebyFrameCalib(
            iwidth=image_width,
            iheight=image_height,
            denormalize=denormalize
        )
        self.calibrator.alpha = alpha

    def calibrate(
        self,
        kp_dict: Dict[int, Dict[str, float]],
        lines_dict: Dict[int, Dict[str, float]],
        tilt: float,  # grados
        zoom: float,  # factor de escala
        image_width: int,
        image_height: int,
    ) -> Optional[CalibrationResult]:
        """
        Ejecuta la calibración con PnLCalib.
        """
        # Actualizar dimensiones de imagen (por si cambian)
        self.image_width = image_width
        self.image_height = image_height
        self.calibrator.image_width = image_width
        self.calibrator.image_height = image_height

        # Actualizar intrínsecos con zoom
        # El zoom escala la focal
        focal_scale = zoom if zoom > 0 else 1.0
        K = self.intrinsics.copy()
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale
        # El tilt afecta la orientación; se usará en la optimización inicial

        # Preparar diccionario de calibración
        calibration = {
            "intrinsics": K,
            "distortion": self.distortion,
        }

        # Actualizar el calibrador con keypoints y líneas
        try:
            self.calibrator.update(calibration, kp_dict, lines_dict)
        except Exception as e:
            logfire.error(f"Error al actualizar calibrador: {e}")
            return None

        # Ejecutar calibración con votación heurística
        # Usamos refine=True y refine_lines=True para mejor precisión
        result = self.calibrator.heuristic_voting(refine=True, refine_lines=True)

        if result is None:
            logfire.warning("No se encontró una calibración válida con heuristic_voting")
            # Intentar con get_homography_from_ground_plane como fallback
            H, rep_err = self.calibrator.get_homography_from_ground_plane(
                use_ransac=5.0, inverse=False, refine_lines=True
            )
            if H is not None:
                return self._build_result_from_homography(H, rep_err or 99, image_width, image_height, "ground_plane")
            return None

        # Extraer parámetros de la calibración
        cam_params = result['cam_params']
        rep_err = result['rep_err']
        method = result['mode']
        use_ransac = result['use_ransac']

        # Reconstruir la matriz de proyección P = K [R | t]
        R = np.array(cam_params['rotation_matrix'], dtype=np.float32)
        t = np.array(cam_params['position_meters'], dtype=np.float32).reshape(3, 1)
        K = self.intrinsics.copy()
        # Aplicar zoom a la focal
        focal_scale = zoom if zoom > 0 else 1.0
        K[0, 0] *= focal_scale
        K[1, 1] *= focal_scale

        P = K @ np.hstack((R, t))
        # Homografía para plano z=0 (el campo está en z=0)
        H = P[:, :3]  # 3x3
        # Normalizar
        H = H / H[2, 2]

        # Calcular error de reproyección con los puntos usados
        obj_pts, img_pts = self.calibrator.get_correspondences(method)
        if len(obj_pts) > 0:
            rep_err = _reprojection_error(H, img_pts, obj_pts[:, :2])
        else:
            rep_err = float(result['rep_err'])

        # Construir resultado
        calib_result = CalibrationResult(
            H=H,
            intrinsics=K,
            distortion=self.distortion,
            rotation_matrix=R,
            position_meters=t.flatten(),
            pan_deg=cam_params['pan_degrees'],
            tilt_deg=cam_params['tilt_degrees'],
            roll_deg=cam_params['roll_degrees'],
            reprojection_error=rep_err,
            inlier_count=len(obj_pts),
            line_inlier_count=len(self.calibrator.lines_dict_cons) if hasattr(self.calibrator, 'lines_dict_cons') else 0,
            method_used=f"pnl_{method}_ransac{use_ransac}",
            scale_estimate=self._estimate_scale(H),
            is_valid=rep_err < 10.0,  # umbral
            frame_num=0,  # se asignará en la capa superior
            match_id=0,
            keypoints_used=self.calibrator.subsets[method] if method in self.calibrator.subsets else {},
            lines_used=self.calibrator.lines_dict_cons if hasattr(self.calibrator, 'lines_dict_cons') else {}
        )

        logfire.info(f"Calibración PnL exitosa: error={rep_err:.2f}px, método={method}")
        return calib_result

    def _build_result_from_homography(
        self,
        H: np.ndarray,
        rep_err: float,
        image_width: int,
        image_height: int,
        method: str
    ) -> CalibrationResult:
        """Construye un CalibrationResult a partir de una homografía directa."""
        # Estimar parámetros de cámara desde la homografía (aproximación)
        # Usamos la función from_homography de FramebyFrameCalib
        self.calibrator.homography = H
        success = self.calibrator.from_homography()
        if not success:
            # Si falla, usar valores por defecto
            R = np.eye(3)
            t = np.zeros(3)
            pan = tilt = roll = 0.0
        else:
            R = self.calibrator.rotation
            t = self.calibrator.position
            pan, tilt, roll = rotation_matrix_to_pan_tilt_roll(R)
            pan = np.rad2deg(pan)
            tilt = np.rad2deg(tilt)
            roll = np.rad2deg(roll)

        return CalibrationResult(
            H=H,
            intrinsics=self.intrinsics,
            distortion=self.distortion,
            rotation_matrix=R,
            position_meters=t,
            pan_deg=pan,
            tilt_deg=tilt,
            roll_deg=roll,
            reprojection_error=rep_err,
            inlier_count=0,
            line_inlier_count=0,
            method_used=f"pnl_{method}_fallback",
            scale_estimate=self._estimate_scale(H),
            is_valid=rep_err < 15.0,
            frame_num=0,
            match_id=0,
            keypoints_used={},
            lines_used={}
        )

    def _estimate_scale(self, H: np.ndarray) -> float:
        """
        Estima la escala de la homografía usando la distancia entre dos puntos
        conocidos del campo (por ejemplo, el ancho del área penal: 16.5m).
        """
        # Puntos del área penal en coordenadas del campo (z=0)
        # Esquina inferior izquierda y superior izquierda del área penal
        p1 = np.array([0.0, 54.16, 1.0])   # (x, y, 1)
        p2 = np.array([0.0, 37.66, 1.0])   # diferencia en y = 16.5m
        # Proyectar a imagen
        proj1 = H @ p1
        proj1 /= proj1[2]
        proj2 = H @ p2
        proj2 /= proj2[2]
        # Distancia en píxeles
        dist_px = np.linalg.norm(proj1[:2] - proj2[:2])
        if dist_px > 0:
            return 16.5 / float(dist_px)  # metros por píxel
        return 1.0