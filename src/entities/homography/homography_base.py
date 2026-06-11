from typing import Optional

import cv2
import logfire
import numpy as np
from sqlmodel import Session

from src.entities.models.app.video_item import VideoItem
from src.entities.homography.homography_cluster import HomographyCluster
from src.entities.homography.homography_lines_operations import HomographyLinesOperation
from src.entities.utils.homography_utils import _line_intersection
from src.entities.models.homography.homography_constants import (
    FIELD_HEIGHT,
    FIELD_KEYPOINTS,
    PITCH_WIDTH,
)
from src.entities.models.homography.homography_models import (
    DetectedKeypoint,
    HomographyResult,
)


class HomographyBase(HomographyCluster, HomographyLinesOperation):
    def __init__(
        self,
        predetermined_points: dict[str, tuple[float, float]],
        reference_scale: float = 1.0,
        pitch_width: float = PITCH_WIDTH,
        pitch_height: float = FIELD_HEIGHT,
        min_keypoints: int = 4,
        ransac_threshold: float = 5.0,
        detection_confidence_threshold: float = 0.5,
        max_reprojection_error: float = 12.0,
        canny_low: int = 60,
        canny_high: int = 150,
        hough_threshold: int = 80,
        hough_min_line_length: int = 60,
        hough_max_line_gap: int = 25,
        line_cluster_angle_tol: float = 18.0,
    ) -> None:
        super().__init__(
            canny_high=canny_high,
            canny_low=canny_low,
            hough_max_line_gap=hough_max_line_gap,
            hough_min_line_length=hough_min_line_length,
            hough_threshold=hough_threshold,
        )
        self.predetermined_points = predetermined_points
        self.min_keypoints = min_keypoints
        self.ransac_threshold = ransac_threshold
        self.detection_confidence_threshold = detection_confidence_threshold
        self.max_reprojection_error = max_reprojection_error
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.hough_threshold = hough_threshold
        self.hough_min_line_length = hough_min_line_length
        self.hough_max_line_gap = hough_max_line_gap
        self.line_cluster_angle_tol = line_cluster_angle_tol

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.reference_scale = reference_scale

        self._last_result: Optional[HomographyResult] = None
        self._cached_H: Optional[np.ndarray] = None

    def calibrate(
        self,
        video_item: VideoItem,
        camera_scale: float,
        camera_tilt: float,
        session: Session,
        use_cache: bool = False,
    ) -> HomographyResult:
        """
        Compute the homography for a single frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame (HxWx3).
        frame_id : int, optional
            Frame index for logging.
        use_cache : bool
            If True and a cached H exists, skip detection and return a result
            using the cached matrix (useful for static cameras).

        Returns
        -------
        HomographyResult
        """
        raise NotImplementedError

    def set_reference_frame_size(self, frame_w: int, frame_h: int) -> None:
        self.reference_frame_size = frame_w, frame_h

    def _transform_predetermined_points(
        self, frame_w: int, frame_h: int, camera_scale: float, camera_tilt: float
    ) -> dict[str, tuple[float, float]]:
        """
        Reproject predetermined_points to match the current zoom and tilt.

        camera_scale: current zoom level (0.1–16). A value of 1.0 means the
                    frame matches reference_frame_size exactly.
        camera_tilt:  vertical tilt in degrees (-5 to +5). Positive = camera
                    tilts down (horizon rises in frame), negative = tilts up.

        Returns a new dict of pixel positions valid for *this* frame.
        """
        ref_w, ref_h = self.reference_frame_size

        if not ref_w or not ref_h:
            raise ValueError("reference_frame_size must be set first")

        norm_pts = {
            name: (px / ref_w, py / ref_h)
            for name, (px, py) in self.predetermined_points.items()
        }

        zoom_ratio = self.reference_scale / camera_scale
        scaled_pts = {}
        for name, (nx, ny) in norm_pts.items():
            sx = 0.5 + (nx - 0.5) * zoom_ratio
            sy = 0.5 + (ny - 0.5) * zoom_ratio
            scaled_pts[name] = (sx, sy)

        TILT_PX_PER_DEGREE = 0.03
        tilt_shift = -camera_tilt * TILT_PX_PER_DEGREE
        tilted_pts = {
            name: (sx, sy + tilt_shift) for name, (sx, sy) in scaled_pts.items()
        }

        result = {}
        for name, (nx, ny) in tilted_pts.items():
            if -0.05 <= nx <= 1.05 and -0.05 <= ny <= 1.05:  # 5% margin
                result[name] = (nx * frame_w, ny * frame_h)

        logfire.debug(
            f"[Homography] _transform_predetermined_points: "
            f"scale={camera_scale}, tilt={camera_tilt}, "
            f"visible={len(result)}/{len(self.predetermined_points)}"
        )
        return result

    def validate_homography(
        self,
        H: np.ndarray,
    ) -> bool:
        frame_w, frame_h = self.reference_frame_size

        if H is None or np.isnan(H).any() or np.isinf(H).any():
            return False

        det = np.linalg.det(H)

        if abs(det) < 1e-8:
            return False

        cond = np.linalg.cond(H)

        if cond > 1e6:
            return False

        corners = np.array(
            [
                [[0, 0]],
                [[frame_w, 0]],
                [[frame_w, frame_h]],
                [[0, frame_h]],
            ],
            dtype=np.float32,
        )

        proj = cv2.perspectiveTransform(
            corners,
            H,
        )

        proj = proj.reshape(-1, 2)

        area = cv2.contourArea(proj.astype(np.float32))
        min_area = (self.pitch_height * self.pitch_width) * 0.25

        return area >= min_area

    def cache_homography(self, result: HomographyResult) -> None:
        """
        Store a HomographyResult's matrix as the static cache.
        Use this for fixed cameras after the first good calibration frame.
        """
        if result.is_valid:
            self._cached_H = result.H.copy()
            logfire.info(f"[Homography] Cached homography: {self._cached_H}")
        else:
            logfire.warning(
                "[Homography] Attempted to cache an invalid homography — ignored"
            )

    def clear_cache(self) -> None:
        """Remove the cached homography (e.g. after a camera cut)."""
        self._cached_H = None

    def project_point(
        self,
        image_pt: tuple[float, float],
        H: Optional[np.ndarray] = None,
    ) -> Optional[tuple[float, float]]:
        """
        Project a single image pixel (x, y) → field (X_m, Y_m).

        Returns None if no valid H is available or the projected point
        falls outside the pitch bounds.
        """
        H = H or (
            self._last_result.H
            if self._last_result and self._last_result.is_valid
            else self._cached_H
        )
        if H is None:
            return None
        pt = np.array([[[image_pt[0], image_pt[1]]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H)[0][0]
        x, y = float(result[0]), float(result[1])
        margin = 5.0
        if not (
            -margin <= x <= self.pitch_width + margin
            and -margin <= y <= self.pitch_height + margin
        ):
            return None
        return x, y

    def project_bbox(
        self,
        bbox: tuple[float, float, float, float],
        H: Optional[np.ndarray] = None,
        use_feet: bool = True,
    ) -> Optional[tuple[float, float]]:
        """
        Project a bounding box to a field position.

        Parameters
        ----------
        bbox : (x1, y1, x2, y2) in pixel coordinates.
        use_feet : bool
            If True, project the bottom-center (feet contact point).
            If False, project the box center — useful for the ball.

        Returns
        -------
        (X_m, Y_m) or None
        """
        x1, y1, x2, y2 = bbox
        px = (x1 + x2) / 2.0
        py = y2 if use_feet else (y1 + y2) / 2.0
        return self.project_point((px, py), H=H)

    def project_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
        H: Optional[np.ndarray] = None,
        use_feet: bool = True,
    ) -> list[Optional[tuple[float, float]]]:
        """
        Vectorised projection for a list of bboxes.
        Returns a list of (X_m, Y_m) or None per bbox.
        """
        H_mat = H or (
            self._last_result.H
            if self._last_result and self._last_result.is_valid
            else self._cached_H
        )
        if H_mat is None or not bboxes:
            return [None] * len(bboxes)

        pts = []
        for x1, y1, x2, y2 in bboxes:
            px = (x1 + x2) / 2.0
            py = y2 if use_feet else (y1 + y2) / 2.0
            pts.append([px, py])

        pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(pts_arr, H_mat).reshape(-1, 2)

        margin = 5.0
        results: list[Optional[tuple[float, float]]] = []
        for x, y in projected:
            x, y = float(x), float(y)
            if (
                -margin <= x <= PITCH_WIDTH + margin
                and -margin <= y <= FIELD_HEIGHT + margin
            ):
                results.append((x, y))
            else:
                results.append(None)
        return results

    def draw_debug(
        self,
        frame: np.ndarray,
        result: Optional[HomographyResult] = None,
    ) -> np.ndarray:
        """
        Return a copy of the frame with detected keypoints, source labels,
        and reprojection error drawn.
        """
        annotated_frame = frame.copy()
        res = result or self._last_result
        if res is None:
            return annotated_frame

        for kp in res.keypoints:
            x, y = int(kp.image_pt[0]), int(kp.image_pt[1])
            color = (0, 255, 0) if kp.source == "detected" else (0, 165, 255)
            cv2.circle(annotated_frame, (x, y), 5, color, -1)
            cv2.putText(
                annotated_frame,
                f"{kp.name[:8]} ({kp.source[0].upper()})",
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

        status = "VALID" if res.is_valid else "INVALID"
        label = (
            f"H {status} | err={res.reprojection_error:.1f}px "
            f"| inliers={res.inlier_count} "
            f"| kpts={len(res.keypoints)}"
        )
        cv2.putText(
            annotated_frame,
            label,
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if res.is_valid else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return annotated_frame

    def _segments_to_candidates(
        self, clusters: list[list[np.ndarray]], frame_w: int, frame_h: int
    ) -> list[tuple[float, float]]:
        """
        Compute candidate intersection points from detected line segments.
        Cross-intersect horizontals × verticals and diagonals × others.
        """

        if len(clusters) < 2:
            return []

        candidates = []
        margin = 20

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for sa in clusters[i]:
                    for sb in clusters[j]:
                        pa1, pa2 = self._extend_segment(
                            sa,
                            frame_w,
                            frame_h,
                        )
                        pb1, pb2 = self._extend_segment(
                            sb,
                            frame_w,
                            frame_h,
                        )
                        pt = _line_intersection(
                            pa1,
                            pa2,
                            pb1,
                            pb2,
                        )

                        if pt is None:
                            continue

                        x, y = pt
                        if (
                            -margin <= x <= frame_w + margin
                            and -margin <= y <= frame_h + margin
                        ):
                            candidates.append(pt)

        return candidates

    def _snap_predetermined_points(
        self,
        adapted_points,
        fitted_lines,
    ):
        """
        Move predetermined points to the nearest line.
        """

        corrected = {}

        for name, pt in adapted_points.items():

            p = np.array(pt)

            best = p
            best_dist = np.inf

            for origin, direction in fitted_lines:

                t = np.dot(
                    p - origin,
                    direction,
                )

                proj = origin + t * direction

                d = np.linalg.norm(proj - p)

                if d < best_dist:
                    best_dist = d
                    best = proj

            corrected[name] = (
                float(best[0]),
                float(best[1]),
            )

        return corrected

    def _build_correspondences(
        self,
        candidate_pts: list[tuple[float, float]],
        frame_w: int,
        frame_h: int,
        adapted_predetermined: dict[str, tuple[float, float]],
        result_id: int,
    ) -> list[DetectedKeypoint]:
        """
        For each known field keypoint that has a predetermined pixel location:
        - If a detected candidate is close enough (within proximity_px), use it
          as "detected" with confidence proportional to closeness.
        - Otherwise fall back to the predetermined pixel position.
        """
        diag = np.sqrt(frame_w**2 + frame_h**2)
        proximity_px = diag * 0.05

        candidates_arr = (
            np.array(candidate_pts, dtype=np.float32) if candidate_pts else None
        )
        keypoints: list[DetectedKeypoint] = []

        for name, field_pt in FIELD_KEYPOINTS.items():
            if name not in adapted_predetermined:
                if candidates_arr is None:
                    continue
                pred_px = None
            else:
                pred_px = adapted_predetermined[name]

            best_pt = None
            best_conf = 0.0
            source = "predetermined"

            if pred_px is not None and candidates_arr is not None:
                anchor = np.array(pred_px)
                dists = np.linalg.norm(
                    candidates_arr - anchor,
                    axis=1,
                )
                nearby = np.where(dists < proximity_px)[0]
                if len(nearby):
                    best = nearby[np.argmin(dists[nearby])]
                    best_pt = tuple(candidates_arr[best])
                    best_conf = 1.0 - dists[best] / proximity_px

                    source = "detected"

            if best_pt is None:
                if pred_px is None:
                    continue
                best_pt = pred_px
                best_conf = 0.3
                source = "predetermined"

            keypoints.append(
                DetectedKeypoint(
                    homography_result_id=result_id,
                    name=name,
                    image_pt_x=float(best_pt[0]),
                    image_pt_y=float(best_pt[1]),
                    field_pt_x=float(field_pt[0]),
                    field_pt_y=float(field_pt[1]),
                    source=source,
                    confidence=float(best_conf),
                )
            )

        return keypoints
