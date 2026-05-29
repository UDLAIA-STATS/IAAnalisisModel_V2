from typing import Dict, List, Tuple, Set, Optional
import math
import logfire
from collections import defaultdict

from sqlmodel import Session

from src.entities.services.player_validator_base import PlayerValidatorBase, UnionFind
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.soccer.player_model import PlayerState


# ── small data containers ────────────────────────────────────────────────────

class MotionSnapshot:
    """Lightweight summary of a track's physical motion profile."""

    __slots__ = (
        "player_id",
        "frame_start", "frame_end",
        "timestamp_start", "timestamp_end",
        "detection_count",
        # direction (unit vector averaged over all frames)
        "mean_dx", "mean_dy",
        # magnitudes
        "mean_speed_kmh",
        "mean_acceleration",
        # bounding positions for IOU / spatial matching
        "first_x1", "first_y1", "first_x2", "first_y2",
        "last_x1",  "last_y1",  "last_x2",  "last_y2",
    )

    def __init__(
        self,
        player_id: int,
        frame_start: int, frame_end: int,
        timestamp_start: float, timestamp_end: float,
        detection_count: int,
        mean_dx: float, mean_dy: float,
        mean_speed_kmh: float,
        mean_acceleration: float,
        first_x1: float, first_y1: float, first_x2: float, first_y2: float,
        last_x1: float,  last_y1: float,  last_x2: float,  last_y2: float,
    ) -> None:
        self.player_id = player_id
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.timestamp_start = timestamp_start
        self.timestamp_end = timestamp_end
        self.detection_count = detection_count
        self.mean_dx = mean_dx
        self.mean_dy = mean_dy
        self.mean_speed_kmh = mean_speed_kmh
        self.mean_acceleration = mean_acceleration
        self.first_x1 = first_x1
        self.first_y1 = first_y1
        self.first_x2 = first_x2
        self.first_y2 = first_y2
        self.last_x1 = last_x1
        self.last_y1 = last_y1
        self.last_x2 = last_x2
        self.last_y2 = last_y2

    @property
    def direction_angle_deg(self) -> float:
        """Predominant heading of the track in degrees [0, 360)."""
        return math.degrees(math.atan2(self.mean_dy, self.mean_dx)) % 360


class PlayerPhysicalValidator(PlayerValidatorBase):
    """
    Merge short-lived ('below-threshold') tracks into stable ('correct') ones
    by comparing physical motion features:

        • Frame number / timestamp continuity — no temporal overlap allowed.
        • Direction alignment — angle between the two average heading vectors.
        • Speed plausibility — the speed at the join point must not imply a
          physically impossible acceleration spike.
        • IoU of adjacent bounding boxes — spatial proximity gate.
        • Appearance count — only tracks below the dynamic threshold (¼ of
          the match maximum) are treated as merge candidates.

    Returns True if at least one merge was performed so the caller knows it
    must re-compute the downstream physical metrics (speed, acceleration, …)
    for the affected players.
    """

    # ── tuneable constants ────────────────────────────────────────────────
    MAX_DIRECTION_ANGLE_DEG: float = 60.0     # max heading divergence (°)
    MAX_TIMESTAMP_GAP_S: float = 3.0          # max gap between tracks (seconds)
    MIN_IOU: float = 0.0                      # IoU gate (0 = only bbox adjacency needed)
    MAX_SPEED_DELTA_KMH: float = 20.0         # max speed change at the join
    MERGE_SCORE_THRESHOLD: float = 1.0        # composite score upper bound

    # composite score weights (must sum to 1)
    W_DIRECTION: float = 0.35
    W_IOU: float = 0.30
    W_SPEED: float = 0.20
    W_TIMESTAMP: float = 0.15

    # ── public entry point ────────────────────────────────────────────────

    def validate(self, match_id: int, session: Session) -> bool:
        """
        Run the physical merge pass for *match_id*.

        Returns
        -------
        bool
            True  → at least one merge was committed; caller should
                    reset and re-compute physical metrics for affected IDs.
            False → nothing changed.
        """
        logfire.info(f"[PlayerPhysicalValidator] Starting for match {match_id}")

        limit = PlayerStatesRepository.get_max_aparitions(match_id, session)
        logfire.info(f"[PlayerPhysicalValidator] Appearance threshold = {limit}")

        below_states, correct_states = PlayerStatesRepository.get_states_appearances(
            match_id=match_id,
            limit_appearance=limit,
            session=session,
        )

        if not below_states or not correct_states:
            logfire.info("[PlayerPhysicalValidator] Nothing to validate")
            return False

        below_summaries = self._build_motion_summaries(below_states)
        correct_summaries = self._build_motion_summaries(correct_states)

        logfire.info(
            f"[PlayerPhysicalValidator] below={len(below_summaries)} "
            f"correct={len(correct_summaries)}"
        )

        scored: List[Tuple[float, int, int]] = []

        for below_id, below_summary in below_summaries.items():
            for correct_id, correct_summary in correct_summaries.items():

                if self._has_temporal_overlap(below_summary, correct_summary):
                    continue

                timestamp_gap = self._timestamp_gap(below_summary, correct_summary)
                if timestamp_gap > self.MAX_TIMESTAMP_GAP_S:
                    continue

                dir_angle = self._direction_delta(below_summary, correct_summary)
                if dir_angle > self.MAX_DIRECTION_ANGLE_DEG:
                    continue

                speed_delta = abs(below_summary.mean_speed_kmh - correct_summary.mean_speed_kmh)
                if speed_delta > self.MAX_SPEED_DELTA_KMH:
                    continue

                if correct_summary.frame_end < below_summary.frame_start:
                    early, late = correct_summary, below_summary
                else:
                    early, late = below_summary, correct_summary

                iou = self._bbox_iou(
                    (early.last_x1,  early.last_y1,  early.last_x2,  early.last_y2),
                    (late.first_x1, late.first_y1, late.first_x2, late.first_y2),
                )

                if iou < self.MIN_IOU:
                    continue

                score = self._physical_score(dir_angle, iou, speed_delta, timestamp_gap)
                scored.append((score, below_id, correct_id))

                logfire.info(
                    f"[PlayerPhysicalValidator] Candidate {below_id}↔{correct_id}: "
                    f"score={score:.3f}  dir={dir_angle:.1f}°  "
                    f"iou={iou:.3f}  Δspeed={speed_delta:.1f}  ts_gap={timestamp_gap:.2f}s"
                )

        if not scored:
            logfire.info("[PlayerPhysicalValidator] No valid candidates found")
            return False

        scored.sort(key=lambda x: x[0])

        uf = UnionFind()
        merged_below: Set[int] = set()

        for score, below_id, correct_id in scored:
            if score > self.MERGE_SCORE_THRESHOLD:
                break
            if below_id in merged_below:
                continue

            uf.union(correct_id, below_id)
            merged_below.add(below_id)

            logfire.info(
                f"[PlayerPhysicalValidator] Scheduling merge "
                f"{below_id} → {correct_id} (score={score:.3f})"
            )

        # ── execute merges ────────────────────────────────────────────────
        groups = uf.groups()
        total_merged = 0

        for root, members in groups.items():
            for member in members:
                if member == root:
                    continue
                PlayerStatesRepository.merge_states(root, member, session)
                total_merged += 1

        if total_merged == 0:
            return False

        session.commit()

        logfire.info(
            f"[PlayerPhysicalValidator] Done. "
            f"Merged {total_merged} tracks across {len(groups)} groups."
        )

        return True

    # ── motion summary builder ────────────────────────────────────────────

    def _build_motion_summaries(
        self, states: List[PlayerState]
    ) -> Dict[int, MotionSnapshot]:
        """Collapse a flat list of PlayerState rows into one MotionSnapshot each."""

        per_player: Dict[int, List[PlayerState]] = defaultdict(list)
        for s in states:
            per_player[s.player_id].append(s)

        summaries: Dict[int, MotionSnapshot] = {}

        for player_id, detections in per_player.items():
            detections.sort(key=lambda s: s.frame_number)

            first = detections[0]
            last  = detections[-1]

            # average motion vectors — skip rows where dx/dy are None
            dx_vals = [s.dx for s in detections if s.dx is not None]
            dy_vals = [s.dy for s in detections if s.dy is not None]
            speeds  = [s.speed_kmh for s in detections if s.speed_kmh is not None]
            accels  = [s.acceleration for s in detections if s.acceleration is not None]

            mean_dx = sum(dx_vals) / len(dx_vals) if dx_vals else 0.0
            mean_dy = sum(dy_vals) / len(dy_vals) if dy_vals else 0.0
            mean_speed = sum(speeds) / len(speeds) if speeds else 0.0
            mean_accel = sum(accels) / len(accels) if accels else 0.0

            summaries[player_id] = MotionSnapshot(
                player_id=player_id,
                frame_start=first.frame_number,
                frame_end=last.frame_number,
                timestamp_start=first.timestamp,
                timestamp_end=last.timestamp,
                detection_count=len(detections),
                mean_dx=mean_dx,
                mean_dy=mean_dy,
                mean_speed_kmh=mean_speed,
                mean_acceleration=mean_accel,
                first_x1=first.x1, first_y1=first.y1,
                first_x2=first.x2, first_y2=first.y2,
                last_x1=last.x1,   last_y1=last.y1,
                last_x2=last.x2,   last_y2=last.y2,
            )

        return summaries

    # ── geometric / physical helpers ──────────────────────────────────────

    @staticmethod
    def _has_temporal_overlap(a: MotionSnapshot, b: MotionSnapshot) -> bool:
        """True when the two tracks are active at the same frame."""
        return (
            max(a.frame_start, b.frame_start)
            <= min(a.frame_end, b.frame_end)
        )

    @staticmethod
    def _timestamp_gap(a: MotionSnapshot, b: MotionSnapshot) -> float:
        """
        Absolute gap in seconds between the end of the earlier track
        and the start of the later one.
        """
        if a.timestamp_end <= b.timestamp_start:
            return b.timestamp_start - a.timestamp_end
        return a.timestamp_start - b.timestamp_end

    @staticmethod
    def _direction_delta(a: MotionSnapshot, b: MotionSnapshot) -> float:
        """
        Angle (degrees) between the two average heading vectors.
        Returns 0 when both tracks are effectively stationary.
        """
        mag_a = math.hypot(a.mean_dx, a.mean_dy)
        mag_b = math.hypot(b.mean_dx, b.mean_dy)

        if mag_a < 1e-6 or mag_b < 1e-6:
            return 0.0

        dot = a.mean_dx * b.mean_dx + a.mean_dy * b.mean_dy
        cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def _bbox_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        """
        Intersection-over-Union for axis-aligned bounding boxes.
        Each box is (x1, y1, x2, y2).  Returns 0.0 when boxes do not overlap.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter   = inter_w * inter_h

        if inter == 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union  = area_a + area_b - inter

        return inter / union if union > 0.0 else 0.0

    def _physical_score(
        self,
        dir_angle: float,
        iou: float,
        speed_delta: float,
        ts_gap: float,
    ) -> float:
        """
        Composite score in [0, 1] — lower is better.

        Each component is normalised to [0, 1] against its hard-limit ceiling:
          • direction  → 0 = perfectly aligned, 1 = at the angular limit
          • iou        → 0 = perfect spatial overlap, 1 = no overlap
          • speed      → 0 = identical speed, 1 = at the speed delta limit
          • timestamp  → 0 = consecutive frames, 1 = at the time-gap limit
        """
        dir_score   = dir_angle   / self.MAX_DIRECTION_ANGLE_DEG
        iou_score   = 1.0 - iou                                    # higher IoU → lower cost
        speed_score = speed_delta / self.MAX_SPEED_DELTA_KMH
        ts_score    = ts_gap      / self.MAX_TIMESTAMP_GAP_S

        return (
            self.W_DIRECTION  * dir_score
            + self.W_IOU      * iou_score
            + self.W_SPEED    * speed_score
            + self.W_TIMESTAMP * ts_score
        )

physical_validator = PlayerPhysicalValidator()