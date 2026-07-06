import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

from src.config.constants import PITCH_WIDTH, PITCH_LENGTH
from src.core.repository.homography_repository import HomographyRepository
from src.entities.models.homography import CleanHomographyFrame
from src.core.vision.correction.scale_corrector import ScaleCorrector


@dataclass
class ProjectedPosition:
    x_meters: float
    y_meters: float
    source: Literal["keyframe", "interpolated", "extrapolated", "clipped", "invalid", "scale_corrected"]
    ref_frames: tuple[int, int]
    is_valid: bool = True


class HomographyInterpolator:
    """
    Builds a linear interpolator over a sparse set of validated
    homography keyframes. For any frame number, returns the best
    available H matrix via:
      - exact match if the frame is a keyframe
      - linear blend between the two surrounding keyframes
      - nearest edge keyframe if outside the keyframe range
    """

    SOFT_MARGIN = 2.0  # Tolerancia para clamps suaves (ej. 2m fuera)
    HARD_LIMIT = 10.0

    def __init__(self, match_id: int, session, clean_frames: list[CleanHomographyFrame]):
        if not clean_frames:
            raise ValueError("Cannot build interpolator with no keyframes")

        sorted_frames = sorted(clean_frames, key=lambda f: f.frame_num)
        self._frames = np.array([f.frame_num for f in sorted_frames])
        self._matrices = np.array([f.H for f in sorted_frames])
        self._scale_corrector = ScaleCorrector(session)
        self.match_id = match_id

    def get_H(self, frame: int) -> tuple[np.ndarray, str, tuple[int, int]]:
        frames = self._frames
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
        left = right - 1
        t = (frame - frames[left]) / (frames[right] - frames[left])
        H = (1 - t) * matrices[left] + t * matrices[right]

        return H, "interpolated", (int(frames[left]), int(frames[right]))

    def project(
        self, x1: float, y1: float, x2: float, y2: float, frame: int
    ) -> ProjectedPosition:
        H, source, ref = self.get_H(frame)
        x_px = (x1 + x2) / 2.0
        y_px = y2

        p = H @ np.array([x_px, y_px, 1.0], dtype=np.float32)
        p /= p[2]
        x_m = float(p[0])
        y_m = float(p[1])

        inside_soft = (
            x_m >= -self.SOFT_MARGIN
            and x_m <= PITCH_LENGTH + self.SOFT_MARGIN
            and y_m >= -self.SOFT_MARGIN
            and y_m <= PITCH_WIDTH + self.SOFT_MARGIN
        )

        inside_hard = (
            x_m >= -self.HARD_LIMIT
            and x_m <= PITCH_LENGTH + self.HARD_LIMIT
            and y_m >= -self.HARD_LIMIT
            and y_m <= PITCH_WIDTH + self.HARD_LIMIT
        )

        if inside_soft:
            x_clip = np.clip(x_m, 0.0, PITCH_LENGTH)
            y_clip = np.clip(y_m, 0.0, PITCH_WIDTH)
            if x_clip != x_m or y_clip != y_m:
                source = "clipped"
            return ProjectedPosition(x_clip, y_clip, source, ref, is_valid=True)

        if inside_hard:
            x_clamped = np.clip(x_m, 0.0, PITCH_LENGTH)
            y_clamped = np.clip(y_m, 0.0, PITCH_WIDTH)
            if x_m < 0:
                x_clamped = 0.0
            elif x_m > PITCH_LENGTH:
                x_clamped = PITCH_LENGTH
            if y_m < 0:
                y_clamped = 0.0
            elif y_m > PITCH_WIDTH:
                y_clamped = PITCH_WIDTH
            return ProjectedPosition(x_clamped, y_clamped, "clipped", ref, is_valid=False)

        if self._scale_corrector is not None:
            x_corr, y_corr = self._scale_corrector.correct_point(x_m, y_m, self.match_id)

            if (
                -self.SOFT_MARGIN <= x_corr <= PITCH_LENGTH + self.SOFT_MARGIN
                and -self.SOFT_MARGIN <= y_corr <= PITCH_WIDTH + self.SOFT_MARGIN
            ):
                return ProjectedPosition(x_corr, y_corr, "scale_corrected", ref, is_valid=True)
            else:
                return ProjectedPosition(x_corr, y_corr, "scale_corrected", ref, is_valid=False)
        else:
            return ProjectedPosition(x_m, y_m, "invalid", ref, is_valid=False)

    @classmethod
    def from_match(cls, match_id: int, session) -> "HomographyInterpolator":
        frames = HomographyRepository.get_clean_homographies(
            match_id=match_id, session=session, max_reprojection_error=3.0
        )
        interpolator = cls(match_id, session, frames)
        return interpolator