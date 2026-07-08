from typing import Optional

import cv2
import numpy as np


def _line_intersection(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
) -> Optional[tuple[float, float]]:
    """
    Compute the intersection of line (p1→p2) and (p3→p4).
    Returns None if lines are parallel.
    """
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-8:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / cross
    pt = p1 + t * d1
    return float(pt[0]), float(pt[1])


def _angle_deg(line: np.ndarray) -> float:
    """Return angle in [0, 180) degrees for a line segment (x1,y1,x2,y2)."""
    dx = line[2] - line[0]
    dy = line[3] - line[1]
    return float(np.degrees(np.arctan2(abs(dy), abs(dx)))) % 180


def _reprojection_error(
    H: np.ndarray,
    img_pts: np.ndarray,
    field_pts: np.ndarray,
) -> float:
    """Mean pixel reprojection error."""
    projected = cv2.perspectiveTransform(img_pts.reshape(-1, 1, 2), H)
    return float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - field_pts, axis=1)))


def _canonical_angle(angle_rad: float) -> float:
    """
    Fold any line angle into [0, π) so a segment and its 180°-flipped
    twin compare as identical.  A→B and B→A are the same physical line.
    """
    a = angle_rad % np.pi
    return a if a >= 0 else a + np.pi


def _angle_distance(a: float, b: float) -> float:
    """
    Smallest angular distance between two canonical angles in [0, π).
    Always returns a value in [0, π/2].
    """
    d = abs(a - b) % np.pi
    return min(d, np.pi - d)

def _snap_corner(
        lineA,
        lineB,
    ):
        return _line_intersection(
            lineA[0],
            lineA[0] + lineA[1] * 1000,
            lineB[0],
            lineB[0] + lineB[1] * 1000,
        )
