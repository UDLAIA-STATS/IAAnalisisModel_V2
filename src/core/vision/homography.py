from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import logfire
import numpy as np
from sqlmodel import Session
from src.entities.models.app.video_item import VideoItem
from src.entities.utils.homography_utils import _line_intersection, _reprojection_error
from src.entities.models.homography.homography_models import HomographyResult
from src.entities.homography.homography_base import HomographyBase
from src.entities.models.homography import PITCH_WIDTH, FIELD_HEIGHT, FIELD_KEYPOINTS


class PitchHomography(HomographyBase):
    """
    Compute and apply a homography from camera frame to a FIFA pitch template.

    Parameters
    ----------
    predetermined_points : dict[str, tuple[float, float]]
        Manually calibrated fallback pixel positions for known field keypoints.
        Keys must match names in FIELD_KEYPOINTS.
        Example: {"tl_corner": (120.0, 45.0), "tr_corner": (1160.0, 38.0), ...}
    min_keypoints : int
        Minimum number of correspondences required to attempt homography.
    ransac_threshold : float
        RANSAC reprojection threshold in pixels.
    detection_confidence_threshold : float
        Minimum normalised confidence for a detected keypoint to be preferred
        over its predetermined fallback.
    max_reprojection_error : float
        Maximum acceptable mean reprojection error (pixels). Results above
        this threshold are flagged as invalid.
    canny_low, canny_high : int
        Thresholds for Canny edge detection.
    hough_threshold : int
        Accumulator threshold for HoughLinesP.
    hough_min_line_length : int
        Minimum line length (pixels) for HoughLinesP.
    hough_max_line_gap : int
        Maximum allowed gap (pixels) between line segments for HoughLinesP.
    line_cluster_angle_tol : float
        Angle tolerance (degrees) for grouping detected lines into
        horizontal / vertical / diagonal clusters.
    """

    def calibrate(
        self, video_item, camera_scale, camera_tilt, session, use_cache=False
    ):
        h, w = video_item.frame.shape[:2]
        self.reference_frame_size = (w, h)

        if use_cache and self._cached_H is not None:
            homography = HomographyResult(
                keypoints=[],
                match_id=video_item.match_id,
                reprojection_error=0.0,
                inlier_count=-1,
                is_valid=True,
                frame_num=video_item.frame_num,
            )
            homography.H = self._cached_H
            session.add(homography)
            session.flush()
            return homography

        segments = self._detect_lines(video_item.frame)
        if len(segments) < 6:
            fallback = self._wall_floor_fallback_line(video_item.frame)
            logfire.info(f"[Homography] Fallback line detected: {fallback}")
            if fallback is not None:
                segments = np.array([fallback], dtype=np.float32)

        segments = self._merge_segments(segments)
        clusters = self._cluster_lines(segments)
        logfire.info(f"[Homography] Line clusters detected: {len(clusters)}, segments detected: {len(segments)}")

        homography = HomographyResult(
            reprojection_error=float("inf"),
            inlier_count=0,
            is_valid=False,
            frame_num=video_item.frame_num,
            match_id=video_item.match_id,
        )
        session.add(homography)
        session.flush()

        candidate_image_pts = self._segments_to_candidates(clusters, w, h)
        candidate_image_pts = self._cluster_intersections(candidate_image_pts)
        logfire.info(f"[Homography] Candidate points detected: {len(candidate_image_pts)}")

        adapted_points = self._transform_predetermined_points(
            w, h, camera_scale, camera_tilt
        )

        keypoints = self._build_correspondences(
            candidate_image_pts, w, h, adapted_points, homography.id
        )
        session.add_all(keypoints)
        session.flush()
        homography.keypoints = (
            keypoints
        )

        logfire.info(f"[Homography] Adapted points detected: {len(keypoints)}: {adapted_points}")
        logfire.info(f"[Homography] Keypoints detectted: {len(keypoints)}, first 5: {keypoints[:5]}")


        if len(keypoints) < self.min_keypoints:
            if self._last_result is not None and self._last_result.is_valid:
                homography.H = self._last_result.H.copy()
                homography.inlier_count = self._last_result.inlier_count
                homography.reprojection_error = self._last_result.reprojection_error
                homography.is_valid = True
            elif self._cached_H is not None:
                homography.H = self._cached_H.copy()
                homography.is_valid = True
            else:
                homography.H = np.eye(3)
                homography.is_valid = False
            session.flush()
            return homography

        img_pts = np.array([kp.image_pt for kp in keypoints], dtype=np.float32)
        fld_pts = np.array([kp.field_pt for kp in keypoints], dtype=np.float32)

        H, mask = cv2.findHomography(
            img_pts, fld_pts, cv2.RANSAC, self.ransac_threshold
        )
        logfire.info(f"[Homography] Homography found: {H}m with mask: {mask}")
        if H is None:
            homography.H = np.eye(3)
            homography.inlier_count = 0
            session.flush()
            return homography

        inliers = int(mask.sum()) if mask is not None else 0
        repr_err = _reprojection_error(H, img_pts, fld_pts)
        geometry_ok = self.validate_homography(H)
        logfire.info(f"[Homography] Found homography with {inliers} inliers and reprojection error of {repr_err}, geometry ok: {geometry_ok}")

        homography.H = H
        homography.reprojection_error = repr_err
        homography.inlier_count = inliers
        homography.is_valid = inliers >= self.min_keypoints and geometry_ok

        session.flush()
        self._last_result = homography
        return homography


_predetermined: dict[str, tuple[float, float]] = {
    "tl_corner": (142.0, 38.0),
    "tr_corner": (1138.0, 32.0),
    "bl_corner": (68.0, 698.0),
    "br_corner": (1212.0, 694.0),
    "mid_top": (640.0, 20.0),
    "mid_bottom": (640.0, 710.0),
    "lpen_tl": (142.0, 182.0),
    "lpen_bl": (142.0, 518.0),
    "rpen_tr": (1138.0, 178.0),
    "rpen_br": (1138.0, 522.0),
}


pitch_homography = PitchHomography(_predetermined)
