import math
from typing import List, Optional, Tuple

import logfire
from sqlmodel import Session

from src.entities.models.soccer.ball_model import BallState
from src.entities.utils.spark_instance import spark


class PredictorBase:
    """Base class with constants and shared utilities for ball prediction."""

    # --- Filtering Thresholds ---
    MIN_BALL_CONFIDENCE: float = 0.35
    STATIC_SPEED_THRESHOLD: float = 0.5  # km/h
    STATIC_FRAME_THRESHOLD: int = 8      # consecutive static frames
    MAX_INTERPOLATION_GAP: int = 5       # max missing frames to interpolate
    MAX_POSITION_JUMP_PX: float = 180.0  # px — filter teleportation
    VELOCITY_SMOOTHING_ALPHA: float = 0.65

    # --- Ball Physics Constraints ---
    MAX_BALL_SPEED_KMH: float = 130.0    # FIFA record ~131 km/h
    MAX_BALL_ACCEL_MSS: float = 150.0    # rough upper bound for ball kick

    # --- Trajectory ---
    TRAJECTORY_WINDOW: int = 5
    FRAME_RATE: float = 30.0             # assumed FPS

    def __init__(self):
        self.spark = spark


    @staticmethod
    def _compute_centroid(state: BallState) -> Tuple[float, float]:
        """Compute bbox centroid from raw coordinates."""
        cx = (state.x1 + state.x2) / 2.0 if state.x1 is not None and state.x2 is not None else 0.0
        cy = (state.y1 + state.y2) / 2.0 if state.y1 is not None and state.y2 is not None else 0.0
        return cx, cy

    @staticmethod
    def _compute_raw_dx_dy(
        curr: BallState, prev: BallState
    ) -> Tuple[float, float]:
        """Raw pixel displacement between two consecutive states."""
        curr_cx, curr_cy = PredictorBase._compute_centroid(curr)
        prev_cx, prev_cy = PredictorBase._compute_centroid(prev)
        return curr_cx - prev_cx, curr_cy - prev_cy

    @staticmethod
    def _compute_meters_from_interpolator(
        interpolator,
        state: BallState,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Project bbox to meters via HomographyInterpolator.
        Returns (dx_meters, dy_meters) or (None, None) on failure.
        """
        try:
            pos = interpolator.project(
                state.x1,
                state.y1,
                state.x2,
                state.y2,
                state.frame_number,
            )
            return pos.x_meters, pos.y_meters
        except Exception:
            return None, None

    @classmethod
    def compute_movement_metrics(
        cls,
        states: List[BallState],
        match_id: int,
        session: Session,
    ) -> List[BallState]:
        """
        Compute dx, dy, dx_meters, dy_meters, speed, acceleration, velocity
        for every BallState in chronological order.

        Uses HomographyInterpolator when available; falls back to raw pixels.
        """
        if not states:
            return states


        interpolator = None
        try:
            from src.core.vision.homography_interpolator import HomographyInterpolator
            interpolator = HomographyInterpolator.from_match(match_id, session)
        except Exception as e:
            logfire.warning(
                f"[BallPredictor] Could not load HomographyInterpolator for match {match_id}: {e}"
            )

        states = sorted(states, key=lambda s: (s.frame_number, s.timestamp))

        for i, state in enumerate(states):
            if i == 0:
                continue

            prev = states[i - 1]

            dx_px, dy_px = cls._compute_raw_dx_dy(state, prev)
            state.dx = dx_px
            state.dy = dy_px

            if interpolator is not None:
                dm_x, dm_y = cls._compute_meters_from_interpolator(interpolator, state)
                prev_dm_x, prev_dm_y = cls._compute_meters_from_interpolator(
                    interpolator, prev
                )
                if dm_x is not None and dm_y is not None and prev_dm_x is not None and prev_dm_y is not None:    
                        state.dx_meters = dm_x - prev_dm_x
                        state.dy_meters = dm_y - prev_dm_y
                else:
                    state.dx_meters = dx_px
                    state.dy_meters = dy_px
            else:
                state.dx_meters = dx_px
                state.dy_meters = dy_px

            dt = state.timestamp - prev.timestamp

            if dt is None or dt == 0:
                dt = 0.0001

            state.delta_x = state.dx_meters
            state.delta_y = state.dy_meters
            state.distance_meters = math.hypot(state.dx_meters, state.dy_meters)

            state.vx = state.dx_meters / dt
            state.vy = state.dy_meters / dt
            speed_ms = math.hypot(state.vx, state.vy)
            state.speed_kmh = speed_ms * 3.6

            state.speed_kmh = min(state.speed_kmh, cls.MAX_BALL_SPEED_KMH)

            prev_vx = prev.vx if prev.vx is not None else 0.0
            prev_vy = prev.vy if prev.vy is not None else 0.0
            state.ax = (state.vx - prev_vx) / dt
            state.ay = (state.vy - prev_vy) / dt
            state.acceleration = math.hypot(state.ax, state.ay)
            state.acceleration = min(state.acceleration, cls.MAX_BALL_ACCEL_MSS)

        return states

