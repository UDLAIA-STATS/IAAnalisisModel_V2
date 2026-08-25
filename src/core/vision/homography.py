from __future__ import annotations

import json

import cv2
import logfire
import numpy as np
from sqlmodel import Session

from src.entities.models.app.video_item import VideoItem
from src.entities.models.homography.homography_models import (
    HomographyResult,
    CalibrationResult,
)
from src.entities.homography.homography_base import HomographyBase

KEYPOINT_ID_TO_NAME = {
    1: "tl_corner",
    2: "mid_top",
    3: "tr_corner",
    4: "lpen_tl",
    5: "lpen_tr",
    6: "rpen_tr",
    7: "rpen_tl",
    8: "lsix_tl",
    9: "lsix_tr",
    10: "rsix_tr",
    11: "rsix_tl",
    12: "lgoal_crossbar_l",
}

def _get_keypoint_name(kp_id: int) -> str:
    """Devuelve el nombre de campo para un ID de keypoint."""
    return KEYPOINT_ID_TO_NAME.get(kp_id, f"kp_{kp_id}")

class PitchHomography(HomographyBase):
    """
    Compute and apply a homography from camera frame to a FIFA pitch template.
    Ahora integra detección ML y calibración PnL como primer intento.
    """

    def __init__(
        self,
        predetermined_points: dict[str, tuple[float, float]]
    ):
        super().__init__(predetermined_points)

    def calibrate(
        self,
        video_item: VideoItem,
        camera_scale: float,
        camera_tilt: float,
        session: Session,
        use_cache: bool = False,
    ) -> HomographyResult:
        h, w = video_item.frame.shape[:2]
        self.reference_frame_size = (int(w), int(h))

        if use_cache and self._cached_H is not None:
            # logfire.info(f"[Homography] Usando caché para frame {video_item.frame_num}")
            result = HomographyResult(
                H_json=json.dumps(self._cached_H.tolist()),
                reprojection_error=0.0,
                inlier_count=-1,
                is_valid=True,
                frame_num=video_item.frame_num,
                match_id=video_item.match_id,
                method_used="cache",
            )
            session.add(result)
            session.flush()
            return result

        ml_success = False
        if self.detector is not None and self.calibrator is not None:
            # logfire.info(f"[Homography] Intentando calibración ML+PnL para frame {video_item.frame_num}")
            try:
                features = self.detector.detect(video_item.frame)
                if features is not None:
                    calib_result = self.calibrator.calibrate(
                        kp_dict=features['kp_dict'],
                        lines_dict=features['lines_dict'],
                        tilt=camera_tilt,
                        zoom=camera_scale,
                        image_width=int(w),
                        image_height=int(h),
                    )
                    if calib_result is not None and calib_result.is_valid:
                        result = self._calibration_result_to_homography_result(
                            calib_result, video_item, session
                        )
                        session.add(result)
                        session.flush()
                        self._last_result = result
                        if result.is_valid:
                            self.cache_homography(result)
                        logfire.info(f"[Homography] Calibración ML+PnL exitosa (err={result.reprojection_error:.2f}px)")
                        return result
                    else:
                        logfire.warning("[Homography] ML+PnL falló o resultado inválido")
                else:
                    logfire.warning("[Homography] Detector ML no devolvió features")
            except Exception as e:
                logfire.error(f"[Homography] Error en ML+PnL: {e}")

        logfire.info(f"[Homography] Usando RANSAC clásico para frame {video_item.frame_num}")
        result = self._ransac_calibrate(video_item, camera_scale, camera_tilt, session)
        if result.is_valid:
            logfire.info(f"[Homography] RANSAC exitoso (err={result.reprojection_error:.2f}px)")
            self.cache_homography(result)
            return result

        logfire.warning(f"[Homography] RANSAC falló, intentando interpolación/extrapolación")
        H_interp = self._interpolate_homography(video_item.frame_num)

        if H_interp is not None:
            H_interp = H_interp / H_interp[2, 2]
            result = HomographyResult(
                H_json=json.dumps(H_interp.tolist()),
                reprojection_error=float("inf"),
                inlier_count=0,
                is_valid=True,
                frame_num=video_item.frame_num,
                match_id=video_item.match_id,
                method_used="interpolated",
            )
            session.add(result)
            session.flush()
            self._last_result = result
            self.cache_homography(result)
            logfire.info("[Homography] Interpolación exitosa (usando homografía anterior)")
            return result

        logfire.error("[Homography] Todas las estrategias fallaron, devolviendo identidad")
        result = HomographyResult(
            H_json=json.dumps(np.eye(3).tolist()),
            reprojection_error=float("inf"),
            inlier_count=0,
            is_valid=False,
            frame_num=video_item.frame_num,
            match_id=video_item.match_id,
            method_used="identity_fallback",
        )
        session.add(result)
        session.flush()
        return result

    def _calibration_result_to_homography_result(
        self,
        calib_result: CalibrationResult,
        video_item: VideoItem,
        session: Session,
    ) -> HomographyResult:
        """Convierte un CalibrationResult a HomographyResult y lo guarda en BD."""
        result = HomographyResult(
            H_json=json.dumps(calib_result.H.tolist()),
            reprojection_error=calib_result.reprojection_error,
            inlier_count=calib_result.inlier_count,
            is_valid=calib_result.is_valid,
            frame_num=video_item.frame_num,
            match_id=video_item.match_id,
            intrinsics_json=json.dumps(calib_result.intrinsics.tolist()),
            distortion_json=json.dumps(calib_result.distortion.tolist()),
            rotation_matrix_json=json.dumps(calib_result.rotation_matrix.tolist()),
            position_meters_json=json.dumps(calib_result.position_meters.tolist()),
            pan_tilt_roll_json=json.dumps({
                "pan": calib_result.pan_deg,
                "tilt": calib_result.tilt_deg,
                "roll": calib_result.roll_deg,
            }),
            method_used=calib_result.method_used,
            line_inlier_count=calib_result.line_inlier_count or 0,
            scale_estimate=calib_result.scale_estimate,
        )
        return result


_predetermined: dict[str, tuple[float, float]] = {
    "tl_corner": (142.0, 38.0),
    "tr_corner": (1138.0, 32.0),
    "bl_corner": (68.0, 698.0),
    "br_corner": (1212.0, 694.0),

    "mid_top": (640.0, 20.0),
    "mid_bottom": (640.0, 710.0),

    "centre_spot": (640.0, 365.0),

    "lpen_tl": (142.0, 182.0),
    "lpen_bl": (142.0, 518.0),
    "lpen_tr": (280.0, 182.0),
    "lpen_br": (280.0, 518.0),

    "rpen_tl": (1000.0, 178.0),
    "rpen_bl": (1000.0, 522.0),
    "rpen_tr": (1138.0, 178.0),
    "rpen_br": (1138.0, 522.0),

    "lsix_tl": (142.0, 275.0),
    "lsix_bl": (142.0, 425.0),
    "lsix_tr": (185.0, 275.0),
    "lsix_br": (185.0, 425.0),

    "rsix_tl": (1095.0, 275.0),
    "rsix_bl": (1095.0, 425.0),
    "rsix_tr": (1138.0, 275.0),
    "rsix_br": (1138.0, 425.0),
}

pitch_homography = PitchHomography(_predetermined)