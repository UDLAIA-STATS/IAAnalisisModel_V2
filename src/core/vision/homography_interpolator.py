# src/core/vision/homography_interpolator.py

import numpy as np
from dataclasses import dataclass
from typing import Literal

from src.core.repository.homography_repository import HomographyRepository
from src.entities.models.homography import CleanHomographyFrame


@dataclass
class ProjectedPosition:
    x_meters: float
    y_meters: float
    source: Literal["keyframe", "interpolated", "extrapolated"]
    ref_frames: tuple[int, int]


class HomographyInterpolator:
    """
    Builds a linear interpolator over a sparse set of validated
    homography keyframes. For any frame number, returns the best
    available H matrix via:
      - exact match if the frame is a keyframe
      - linear blend between the two surrounding keyframes
      - nearest edge keyframe if outside the keyframe range
    """

    def __init__(self, clean_frames: list[CleanHomographyFrame]):
        if not clean_frames:
            raise ValueError("Cannot build interpolator with no keyframes")

        sorted_frames   = sorted(clean_frames, key=lambda f: f.frame_num)
        self._frames    = np.array([f.frame_num for f in sorted_frames])
        self._matrices  = np.array([f.H for f in sorted_frames])

    def get_H(self, frame: int) -> tuple[np.ndarray, str, tuple[int, int]]:
        frames   = self._frames
        matrices = self._matrices

        exact = np.where(frames == frame)[0]
        if len(exact):
            i = exact[0]
            return matrices[i], "keyframe", (int(frames[i]), int(frames[i]))

        if frame < frames[0]:
            return matrices[0], "extrapolated", (int(frames[0]), int(frames[0]))

        if frame > frames[-1]:
            return matrices[-1], "extrapolated", (int(frames[-1]), int(frames[-1]))

        right = int(np.searchsorted(frames, frame))
        left  = right - 1
        t     = (frame - frames[left]) / (frames[right] - frames[left])
        H     = (1 - t) * matrices[left] + t * matrices[right]

        return H, "interpolated", (int(frames[left]), int(frames[right]))

    def project(
        self,
        x1: float, y1: float, x2: float, y2: float,
        frame: int,
    ) -> ProjectedPosition:
        """
        Projects the bottom-centre of a bounding box to field coordinates.
        Uses y2 (feet) as the ground contact point.
        """
        H, source, ref = self.get_H(frame)

        x_px = (x1 + x2) / 2.0
        y_px = y2

        p    = H @ np.array([x_px, y_px, 1.0], dtype=np.float32)
        p   /= p[2]

        return ProjectedPosition(
            x_meters=float(p[0]),
            y_meters=float(p[1]),
            source=source,
            ref_frames=ref,
        )

    @classmethod
    def from_match(cls, match_id: int, session) -> "HomographyInterpolator":
        frames = HomographyRepository.get_clean_homographies(
            match_id=match_id, session=session
        )
        return cls(frames)