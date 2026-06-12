from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import logfire
import numpy as np
from sqlmodel import Session
from src.entities.models.app.video_item import VideoItem
from src.entities.utils.homography_utils import _reprojection_error
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
        self,
        video_item: VideoItem,
        camera_scale: float,
        camera_tilt: float,
        session: Session,
        use_cache: bool = False,
    ) -> HomographyResult:
        h, w = video_item.frame.shape[:2]
 
        if self.reference_frame_size is None or w != self.reference_frame_size[0] or h != self.reference_frame_size[1]:
            self.reference_frame_size = (w, h)
 
        if use_cache and self._cached_H is not None:
            logfire.debug(f"frame {video_item.frame_num}: using cached homography")
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
 
        h, w = video_item.frame.shape[:2]
 
        segments = self._detect_lines(video_item.frame)
        segments = self._merge_segments(segments)
        logfire.debug(f"[Homography] frame {video_item.frame_num}: {len(segments)} line segments detected")
 
        clusters = self._cluster_lines(segments)
  
        adapted_points = self._transform_predetermined_points(w, h, camera_scale, camera_tilt)
 
        candidate_image_pts = self._segments_to_candidates(clusters, w, h)
        candidate_image_pts = self._cluster_intersections(candidate_image_pts)
        logfire.debug(f"[Homography] frame {video_item.frame_num}: {len(candidate_image_pts)} candidate intersections")
 
        homography = HomographyResult(
            reprojection_error=float("inf"),
            inlier_count=0,
            is_valid=False,
            frame_num=video_item.frame_num,
            match_id=video_item.match_id,
        )
        session.add(homography)
        session.flush()
 
        keypoints = self._build_correspondences(candidate_image_pts, w, h, adapted_points, homography.id)
        session.add_all(keypoints)
        session.flush()
        homography.keypoints = keypoints
 
        logfire.debug(f"[Homography] frame {video_item.frame_num}: {len(keypoints)} correspondences")
 
        if len(keypoints) < self.min_keypoints:
            logfire.warning(f"[Homography] frame {video_item.frame_num}: insufficient correspondences ({len(keypoints)})")
            homography.H = np.eye(3)
            session.add(homography)
            session.flush()
            return homography
 
        img_pts = np.array([kp.image_pt for kp in keypoints], dtype=np.float32)
        fld_pts = np.array([kp.field_pt for kp in keypoints], dtype=np.float32)
 
        H, mask = cv2.findHomography(img_pts, fld_pts, cv2.RANSAC, self.ransac_threshold)
 
        if H is None:
            homography.H = np.eye(3)
            homography.inlier_count = 0
            session.add(homography)
            session.flush()
            return homography
 
        inliers = int(mask.sum()) if mask is not None else 0
        repr_err = _reprojection_error(H, img_pts, fld_pts)
        geometry_ok = self.validate_homography(H)
 
        is_valid = (
            repr_err <= self.max_reprojection_error
            and inliers >= self.min_keypoints
            and geometry_ok
        )
 
        if not is_valid:
            logfire.warning(f"[Homography] frame {video_item.frame_num}: invalid homography (repr_err={repr_err})")
 
        result = HomographyResult(
            keypoints=keypoints,
            reprojection_error=repr_err,
            inlier_count=inliers,
            is_valid=is_valid,
            frame_num=video_item.frame_num,
            match_id=video_item.match_id,
        )
        result.H = H
        session.add(result)
        session.flush()
 
        self._last_result = result
        return result
 


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