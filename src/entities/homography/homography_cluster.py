import cv2
import numpy as np

from src.entities.utils.homography_utils import (
    _angle_deg,
    _angle_distance,
    _canonical_angle,
)


class HomographyCluster:
    def _merge_segments(self, segments: np.ndarray) -> np.ndarray:
        """
        Merge near-parallel, near-collinear fragments into single segments.

        Key differences from the original
        ----------------------------------
        Direction-flip fix
            Canonical angles are used for angular comparison, so two
            detections of the same physical line in opposite directions
            are always merged rather than treated as separate lines.

        Perpendicular-distance merging
            The original code compared Euclidean centroid distance
            (threshold 12 px).  Two fragments of the same long line can
            have centroids hundreds of pixels apart along the line, so
            they never merged.  Now we check the perpendicular distance
            between centroids (i.e. how far apart they are *across* the
            line direction).  Fragments of the same line have a near-zero
            perpendicular distance; genuinely parallel lines (like the
            alternating grass stripes) have a large one and stay separate.

        Threshold values
            angle:       5°  (unchanged — tight enough to keep stripe
                              families separate)
            perp distance: 10 px  (replaces the 12 px centroid threshold)
        """
        if len(segments) == 0:
            return segments

        used = np.zeros(len(segments), dtype=bool)
        merged: list[list[float]] = []

        ANGLE_THRESH = np.deg2rad(5)
        PERP_THRESH = 10.0

        for i, seg in enumerate(segments):
            if used[i]:
                continue

            used[i] = True
            group = [seg]

            a_i = _canonical_angle(np.arctan2(seg[3] - seg[1], seg[2] - seg[0]))
            # unit normal (perpendicular to the line direction)
            normal_i = np.array([-np.sin(a_i), np.cos(a_i)])
            mid_i = np.array([(seg[0] + seg[2]) / 2, (seg[1] + seg[3]) / 2])

            for j in range(i + 1, len(segments)):
                if used[j]:
                    continue

                seg_b = segments[j]
                a_j = _canonical_angle(
                    np.arctan2(seg_b[3] - seg_b[1], seg_b[2] - seg_b[0])
                )

                if _angle_distance(a_i, a_j) > ANGLE_THRESH:
                    continue

                mid_j = np.array([(seg_b[0] + seg_b[2]) / 2, (seg_b[1] + seg_b[3]) / 2])

                perp_dist = abs(float((mid_j - mid_i) @ normal_i))
                if perp_dist < PERP_THRESH:
                    used[j] = True
                    group.append(seg_b)

            pts = np.array(
                [pt for g in group for pt in ([g[0], g[1]], [g[2], g[3]])],
                dtype=np.float32,
            )
            vx, vy, x, y = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)

            direction = np.array([vx, vy], dtype=np.float32).reshape(-1)
            origin = np.array([x, y], dtype=np.float32).reshape(1, 2)
            t = (pts - origin) @ direction

            p1 = np.array([x, y]) + t.min() * np.array([vx, vy])
            p2 = np.array([x, y]) + t.max() * np.array([vx, vy])

            merged.append([p1[0][0], p1[1][0], p2[0][0], p2[1][0]])

        return (
            np.array(merged, dtype=np.float32)
            if merged
            else np.empty((0, 4), dtype=np.float32)
        )

    def _cluster_lines(
        self,
        segments: np.ndarray,
        min_cluster_size: int = 2,
    ) -> list[list[np.ndarray]]:
        """
        Partition segments into angle families, camera-angle agnostic.

        Key differences from the original
        ----------------------------------
        Direction-flip fix (the primary bug)
            The original used raw arctan2 angles.  A line detected as
            A→B gives +96°; the same line detected as B→A gives -84°.
            Their arctan2 distance is 180°, so they landed in different
            clusters even though they represent the same physical line.

            Fix: convert every angle to canonical [0, π) once before any
            comparison, and use _angle_distance() which wraps correctly
            at π.  Both detections now produce 96° and cluster together.

        Wider threshold (15° vs 8°)
            Perspective foreshortening makes parallel pitch lines appear
            at slightly different angles near the frame edges.  8° was
            too tight for angled camera views and split a single line
            family into many singleton clusters.  15° keeps all members
            of a family together while still separating the two dominant
            directions (touchlines ~90°, goal lines ~0°, midfield ~17°).

        Drop singleton clusters
            Single-segment clusters can never produce pitch-corner
            intersections.  Removing them (min_cluster_size=2) eliminates
            noise candidates that corrupt _build_correspondences.

        Sort by size, descending
            The two largest families are always the two dominant pitch-
            line directions.  Trying them first maximises the chance
            that the first few intersection candidates are real corners.
        """
        if len(segments) == 0:
            return []

        canon = np.array(
            [_canonical_angle(np.arctan2(s[3] - s[1], s[2] - s[0])) for s in segments]
        )

        used = np.zeros(len(canon), dtype=bool)
        clusters: list[list[np.ndarray]] = []
        THRESHOLD = np.deg2rad(15)

        for i in range(len(canon)):
            if used[i]:
                continue

            used[i] = True
            group = [segments[i]]

            for j in range(i + 1, len(canon)):
                if used[j]:
                    continue
                if _angle_distance(canon[i], canon[j]) < THRESHOLD:
                    used[j] = True
                    group.append(segments[j])

            if len(group) >= min_cluster_size:
                clusters.append(group)

        clusters.sort(key=len, reverse=True)
        return clusters

    def _fit_cluster_lines(
        self,
        clusters: list[list[np.ndarray]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Fit one infinite line per cluster.

        Returns
        -------
        [
            (point_on_line, direction),
            ...
        ]
        """

        output = []

        for cluster in clusters:

            pts = []

            for seg in cluster:
                pts.append([seg[0], seg[1]])
                pts.append([seg[2], seg[3]])

            pts = np.asarray(
                pts,
                dtype=np.float32,
            )

            vx, vy, x, y = cv2.fitLine(
                pts,
                cv2.DIST_L2,
                0,
                0.01,
                0.01,
            )

            output.append(
                (
                    np.array(
                        [x.item(), y.item()],
                        dtype=np.float32,
                    ),
                    np.array(
                        [vx.item(), vy.item()],
                        dtype=np.float32,
                    ),
                )
            )

        return output

    def _cluster_intersections(
        self,
        candidates: list[tuple[float, float]],
        radius: float = 20.0,
    ) -> list[tuple[float, float]]:
        """
        Collapse near-duplicate intersection candidates into one per corner.

        Key differences from the original
        ----------------------------------
        Larger radius (20 px vs 8 px)
            With angled camera geometry, many segment-pair combinations
            converge near the same physical corner but at slightly
            different pixel positions.  8 px was too tight — it left
            dozens of near-duplicate candidates, and
            _build_correspondences then matched whichever happened to be
            closest to the predetermined point, which was often wrong.

            20 px is still small enough not to merge corners from
            different pitch lines (the closest pair of real corners in
            this frame is >80 px apart).
        """
        if not candidates:
            return []

        pts = np.array(candidates, dtype=np.float32)
        used = np.zeros(len(pts), dtype=bool)
        output: list[tuple[float, float]] = []

        for i in range(len(pts)):
            if used[i]:
                continue

            cluster = [pts[i]]
            used[i] = True

            for j in range(i + 1, len(pts)):
                if used[j]:
                    continue
                if np.linalg.norm(pts[i] - pts[j]) < radius:
                    cluster.append(pts[j])
                    used[j] = True

            center = np.mean(cluster, axis=0)
            output.append((float(center[0]), float(center[1])))

        return output
