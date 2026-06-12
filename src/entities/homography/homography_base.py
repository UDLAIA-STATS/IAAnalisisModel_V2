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
        raise NotImplementedError

    def set_reference_frame_size(self, frame_w: int, frame_h: int) -> None:
        self.reference_frame_size = frame_w, frame_h


    def _transform_predetermined_points(
        self, frame_w: int, frame_h: int, camera_scale: float, camera_tilt: float
    ) -> dict[str, tuple[float, float]]:
        ref_w, ref_h = self.reference_frame_size
        if not ref_w or not ref_h:
            raise ValueError("reference_frame_size must be set first")

        norm_pts = {
            name: (px / ref_w, py / ref_h)
            for name, (px, py) in self.predetermined_points.items()
        }
        zoom_ratio = self.reference_scale / camera_scale
        scaled_pts = {
            name: (0.5 + (nx - 0.5) * zoom_ratio, 0.5 + (ny - 0.5) * zoom_ratio)
            for name, (nx, ny) in norm_pts.items()
        }
        TILT_PX_PER_DEGREE = 0.03
        tilt_shift = -camera_tilt * TILT_PX_PER_DEGREE
        result = {}
        for name, (sx, sy) in scaled_pts.items():
            ny = sy + tilt_shift
            if -0.05 <= sx <= 1.05 and -0.05 <= ny <= 1.05:
                result[name] = (sx * frame_w, ny * frame_h)

        logfire.debug(
            f"[Homography] _transform_predetermined_points: "
            f"scale={camera_scale}, tilt={camera_tilt}, "
            f"visible={len(result)}/{len(self.predetermined_points)}"
        )
        return result

    def _segments_to_candidates(
        self, clusters: list[list[np.ndarray]], frame_w: int, frame_h: int
    ) -> list[tuple[float, float]]:
        """
        Cross-intersect segments from different angle families to find
        candidate pitch-corner pixel positions.

        Only segments from DIFFERENT clusters are paired.  Segments
        within the same angle family are parallel (or near-parallel) and
        their intersections are far off-frame — they add noise without
        any signal.
        """
        if len(clusters) < 2:
            return []

        candidates: list[tuple[float, float]] = []
        margin = 20

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                for sa in clusters[i]:
                    for sb in clusters[j]:
                        pa1, pa2 = self._extend_segment(sa, frame_w, frame_h)
                        pb1, pb2 = self._extend_segment(sb, frame_w, frame_h)
                        pt = _line_intersection(pa1, pa2, pb1, pb2)
                        if pt is None:
                            continue
                        x, y = pt
                        if (
                            -margin <= x <= frame_w + margin
                            and -margin <= y <= frame_h + margin
                        ):
                            candidates.append(pt)

        return candidates

    def _build_correspondences(
        self,
        candidate_pts: list[tuple[float, float]],
        frame_w: int,
        frame_h: int,
        adapted_predetermined: dict[str, tuple[float, float]],
        result_id: int,
    ) -> list[DetectedKeypoint]:
        """
        Match each known field keypoint to its best pixel position.

        Strategy
        --------
        For each keypoint that has a predetermined pixel anchor:

          Detected path  — a clustered intersection candidate lies within
            proximity_px of the anchor.  The nearest such candidate is
            used as the image point (source="detected").

          Fallback path  — no candidate is close enough.  The
            predetermined anchor itself is used (source="predetermined",
            confidence=0.3).

        Uniqueness constraint  (critical for RANSAC)
        --------------------------------------------
        Every candidate pixel may be claimed by at most ONE field keypoint.
        When two keypoints compete for the same candidate, the one with
        the shorter anchor-to-candidate distance wins.  The loser falls
        back to its predetermined position.

        Without this constraint, two keypoints that have identical or
        near-identical image coordinates give RANSAC a degenerate system:
        the pixel-space constraint they add is linearly dependent, so
        findHomography either returns None or produces a wildly incorrect H.

        This is the primary reason for the repeated "inliers=0" result:
        tl_corner and lpen_tl were both landing on the same detected
        candidate ~(85, 185) because their predetermined anchors had
        both been snapped to the same fitted-line projection.

        Proximity radius
        ----------------
        8 % of the frame diagonal (vs the original 5 %).  Angled cameras
        make predetermined anchors less accurate, so a slightly wider
        search radius is needed.  At 576×540 this is ~63 px; at 1280×720
        it is ~92 px.  The uniqueness constraint above prevents the wider
        radius from causing false matches.
        """
        diag = np.sqrt(frame_w**2 + frame_h**2)
        proximity_px = diag * 0.08  # 8% — wider than original 5% for angled views

        candidates_arr = (
            np.array(candidate_pts, dtype=np.float32) if candidate_pts else None
        )

        # Pass 1: greedily assign candidates to keypoints, closest-first.
        # claimed[cand_idx] = (best_dist, name_that_claimed_it)
        claimed: dict[int, tuple[float, str]] = {}
        assignments: dict[str, tuple[tuple[float, float], float]] = {}
        # assignments[name] = (image_pt, confidence)

        if candidates_arr is not None:
            # Process keypoints in anchor-distance order so the best-
            # anchored keypoint always wins ties deterministically.
            order = []
            for name in FIELD_KEYPOINTS:
                if name not in adapted_predetermined:
                    continue
                pred_px = adapted_predetermined[name]
                anchor = np.array(pred_px, dtype=np.float32)
                dists = np.linalg.norm(candidates_arr - anchor, axis=1)
                nearby = np.where(dists < proximity_px)[0]
                if not len(nearby):
                    continue
                best_idx = int(nearby[np.argmin(dists[nearby])])
                best_dist = float(dists[best_idx])
                order.append((best_dist, name, best_idx, pred_px))

            order.sort(key=lambda x: x[0])  # closest pair wins first

            for best_dist, name, best_idx, pred_px in order:
                if best_idx in claimed:
                    # already taken by a closer keypoint — fall through to
                    # predetermined fallback (handled in pass 2)
                    continue
                claimed[best_idx] = (best_dist, name)
                conf = 1.0 - best_dist / proximity_px
                assignments[name] = (tuple(candidates_arr[best_idx]), conf)

        # Pass 2: build the DetectedKeypoint list
        keypoints: list[DetectedKeypoint] = []

        for name, field_pt in FIELD_KEYPOINTS.items():
            if name in assignments:
                image_pt, conf = assignments[name]
                source = "detected"
            elif name in adapted_predetermined:
                image_pt = adapted_predetermined[name]
                conf = 0.3
                source = "predetermined"
            else:
                continue

            keypoints.append(
                DetectedKeypoint(
                    homography_result_id=result_id,
                    name=name,
                    image_pt_x=float(image_pt[0]),
                    image_pt_y=float(image_pt[1]),
                    field_pt_x=float(field_pt[0]),
                    field_pt_y=float(field_pt[1]),
                    source=source,
                    confidence=float(conf),
                )
            )

        return keypoints

    
    def validate_homography(self, H: np.ndarray) -> bool:
        frame_w, frame_h = self.reference_frame_size
        if H is None or np.isnan(H).any() or np.isinf(H).any():
            return False
        if abs(np.linalg.det(H)) < 1e-8:
            return False
        if np.linalg.cond(H) > 1e6:
            return False
        corners = np.array(
            [[[0, 0]], [[frame_w, 0]], [[frame_w, frame_h]], [[0, frame_h]]],
            dtype=np.float32,
        )
        proj = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        area = cv2.contourArea(proj.astype(np.float32))
        min_area = (self.pitch_height * self.pitch_width) * 0.25
        return area >= min_area

    def cache_homography(self, result: HomographyResult) -> None:
        if result.is_valid:
            self._cached_H = result.H.copy()
            logfire.info(f"[Homography] Cached homography: {self._cached_H}")
        else:
            logfire.warning(
                "[Homography] Attempted to cache an invalid homography — ignored"
            )

    def clear_cache(self) -> None:
        self._cached_H = None


    def project_batch(
        self,
        bboxes: list[tuple[float, float, float, float]],
        H: Optional[np.ndarray] = None,
        use_feet: bool = True,
    ) -> list[Optional[tuple[float, float]]]:
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
