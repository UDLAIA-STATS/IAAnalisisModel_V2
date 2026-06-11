from typing import Dict, List, Set
import logfire
from collections import defaultdict

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    greatest,
    least,
    lit,
    when,
    abs as sql_abs,
)
from pyspark.sql.types import FloatType, IntegerType, StructType, StructField

from sqlmodel import Session

from src.core.repository.player_repository import PlayerRepository
from src.entities.models.pyspark.udfs import direction_delta_udf, bbox_iou_udf
from src.entities.models.app.union_model import UnionFind, MotionSnapshot
from src.entities.services.player_validator_base_spark import (
    PlayerValidatorBase,
)
from src.core.repository.player_states_repository import PlayerStatesRepository
from src.entities.models.soccer.player_model import PlayerState


class PlayerPhysicalValidator(PlayerValidatorBase):
    """
    Merge short-lived ('below-threshold') tracks into stable ('correct') ones
    by comparing physical motion features using Spark for distributed computation.

    Uses Spark SQL and DataFrames to replace nested loops for significantly
    faster processing on large datasets.
    """

    MAX_DIRECTION_ANGLE_DEG: float = 60.0
    MAX_TIMESTAMP_GAP_S: float = 3.0
    MIN_IOU: float = 0.0
    MAX_SPEED_DELTA_KMH: float = 14.0
    MERGE_SCORE_THRESHOLD: float = 1.0

    W_DIRECTION: float = 0.35
    W_IOU: float = 0.30
    W_SPEED: float = 0.20
    W_TIMESTAMP: float = 0.15

    def __init__(self):
        """Initialize with Spark Session."""
        super().__init__()

    def validate(self, match_id: int, session: Session) -> bool:
        """
        Run the physical merge pass for *match_id* using Spark.

        Returns True if at least one merge was committed.
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

        # ── convert to Spark DataFrames for cross-join and filtering ───────
        below_df = self._summaries_to_dataframe(below_summaries, "below")
        correct_df = self._summaries_to_dataframe(correct_summaries, "correct")

        below_df = (
            below_df.withColumnRenamed("early_cx", "below_early_cx")
            .withColumnRenamed("early_cy", "below_early_cy")
            .withColumnRenamed("late_cx", "below_late_cx")
            .withColumnRenamed("late_cy", "below_late_cy")
        )

        correct_df = (
            correct_df.withColumnRenamed("early_cx", "correct_early_cx")
            .withColumnRenamed("early_cy", "correct_early_cy")
            .withColumnRenamed("late_cx", "correct_late_cx")
            .withColumnRenamed("late_cy", "correct_late_cy")
        )

        below_df = below_df.withColumnRenamed("frame_gap", "below_frame_gap")
        correct_df = correct_df.withColumnRenamed("frame_gap", "correct_frame_gap")

        candidates_df = below_df.crossJoin(correct_df)

        ts_gap = self._timestamp_gap_expr(
            col("below_timestamp_end"),
            col("below_timestamp_start"),
            col("correct_timestamp_end"),
            col("correct_timestamp_start"),
        )

        candidates_df = (
            candidates_df.withColumn(
                "early_x1",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("below_last_x1"),
                ).otherwise(col("correct_last_x1")),
            )
            .withColumn(
                "early_y1",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("below_last_y1"),
                ).otherwise(col("correct_last_y1")),
            )
            .withColumn(
                "early_x2",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("below_last_x2"),
                ).otherwise(col("correct_last_x2")),
            )
            .withColumn(
                "early_y2",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("below_last_y2"),
                ).otherwise(col("correct_last_y2")),
            )
            .withColumn(
                "late_x1",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("correct_first_x1"),
                ).otherwise(col("below_first_x1")),
            )
            .withColumn(
                "late_y1",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("correct_first_y1"),
                ).otherwise(col("below_first_y1")),
            )
            .withColumn(
                "late_x2",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("correct_first_x2"),
                ).otherwise(col("below_first_x2")),
            )
            .withColumn(
                "late_y2",
                when(
                    col("below_timestamp_end") <= col("correct_timestamp_start"),
                    col("correct_first_y2"),
                ).otherwise(col("below_first_y2")),
            )
        )

        filtered = (
            candidates_df.filter(
                ~self._has_temporal_overlap_expr(
                    col("below_frame_start"),
                    col("below_frame_end"),
                    col("correct_frame_start"),
                    col("correct_frame_end"),
                )
            )
            .filter(
                self._timestamp_gap_expr(
                    col("below_timestamp_end"),
                    col("below_timestamp_start"),
                    col("correct_timestamp_end"),
                    col("correct_timestamp_start"),
                )
                <= self.MAX_TIMESTAMP_GAP_S
            )
            .filter(
                self._direction_delta_expr(
                    col("below_mean_dx"),
                    col("below_mean_dy"),
                    col("correct_mean_dx"),
                    col("correct_mean_dy"),
                )
                <= self.MAX_DIRECTION_ANGLE_DEG
            )
            .filter(
                sql_abs(col("below_mean_speed_kmh") - col("correct_mean_speed_kmh"))
                <= self.MAX_SPEED_DELTA_KMH
            )
        )

        # Calculate composite scores
        scored = (
            filtered.withColumn(
                "score",
                self._physical_score_expr(
                    self._direction_delta_expr(
                        col("below_mean_dx"),
                        col("below_mean_dy"),
                        col("correct_mean_dx"),
                        col("correct_mean_dy"),
                    ),
                    self._iou_expr(
                        col("early_x1"),
                        col("early_y1"),
                        col("early_x2"),
                        col("early_y2"),
                        col("late_x1"),
                        col("late_y1"),
                        col("late_x2"),
                        col("late_y2"),
                    ),
                    sql_abs(
                        col("below_mean_speed_kmh") - col("correct_mean_speed_kmh")
                    ),
                    self._timestamp_gap_expr(
                        col("below_timestamp_end"),
                        col("below_timestamp_start"),
                        col("correct_timestamp_end"),
                        col("correct_timestamp_start"),
                    ),
                ),
            )
            .orderBy("score")
            .select(col("score"), col("below_player_id"), col("correct_player_id"))
        )

        scored_list = [
            (row.score, row.below_player_id, row.correct_player_id)
            for row in scored.collect()
        ]

        if not scored_list:
            logfire.info("[PlayerPhysicalValidator] No valid candidates found")
            return False

        uf = UnionFind()
        merged_below: Set[int] = set()

        for score, below_id, correct_id in scored_list:
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

        below_states, _ = PlayerStatesRepository.get_states_appearances(
            match_id=match_id,
            limit_appearance=limit,
            session=session,
        )

        if not below_states:
            logfire.info("[PlayerPhysicalValidator] Nothing to validate")
            return True

        player_ids = self._get_ghost_player_ids(below_states, appearance_threshold=160)

        for player_id in player_ids:
            PlayerRepository.delete_player(player_id, session)

        return True

    def _build_motion_summaries(
        self, states: List[PlayerState]
    ) -> Dict[int, MotionSnapshot]:
        """Collapse player states to MotionSnapshot using optimized logic."""

        per_player: Dict[int, List[PlayerState]] = defaultdict(list)
        for state in states:
            per_player[state.player_id].append(state)

        summaries: Dict[int, MotionSnapshot] = {}

        for player_id, detections in per_player.items():
            detections.sort(key=lambda s: s.frame_number)

            first = detections[0]
            last = detections[-1]

            dx_vals = [s.dx for s in detections if s.dx is not None]
            dy_vals = [s.dy for s in detections if s.dy is not None]
            speeds = [s.speed_kmh for s in detections if s.speed_kmh is not None]
            accels = [s.acceleration for s in detections if s.acceleration is not None]

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
                first_x1=first.x1,
                first_y1=first.y1,
                first_x2=first.x2,
                first_y2=first.y2,
                last_x1=last.x1,
                last_y1=last.y1,
                last_x2=last.x2,
                last_y2=last.y2,
            )

        return summaries

    def _summaries_to_dataframe(
        self, summaries: Dict[int, MotionSnapshot], prefix: str
    ) -> DataFrame:
        """Convert MotionSnapshot dict to Spark DataFrame."""
        rows = []
        for player_id, snapshot in summaries.items():
            rows.append(
                {
                    f"{prefix}_player_id": player_id,
                    f"{prefix}_frame_start": snapshot.frame_start,
                    f"{prefix}_frame_end": snapshot.frame_end,
                    f"{prefix}_timestamp_start": snapshot.timestamp_start,
                    f"{prefix}_timestamp_end": snapshot.timestamp_end,
                    f"{prefix}_detection_count": snapshot.detection_count,
                    f"{prefix}_mean_dx": snapshot.mean_dx,
                    f"{prefix}_mean_dy": snapshot.mean_dy,
                    f"{prefix}_mean_speed_kmh": snapshot.mean_speed_kmh,
                    f"{prefix}_mean_acceleration": snapshot.mean_acceleration,
                    f"{prefix}_first_x1": snapshot.first_x1,
                    f"{prefix}_first_y1": snapshot.first_y1,
                    f"{prefix}_first_x2": snapshot.first_x2,
                    f"{prefix}_first_y2": snapshot.first_y2,
                    f"{prefix}_last_x1": snapshot.last_x1,
                    f"{prefix}_last_y1": snapshot.last_y1,
                    f"{prefix}_last_x2": snapshot.last_x2,
                    f"{prefix}_last_y2": snapshot.last_y2,
                }
            )

        schema = StructType(
            [
                StructField(f"{prefix}_player_id", IntegerType(), True),
                StructField(f"{prefix}_frame_start", IntegerType(), True),
                StructField(f"{prefix}_frame_end", IntegerType(), True),
                StructField(f"{prefix}_timestamp_start", FloatType(), True),
                StructField(f"{prefix}_timestamp_end", FloatType(), True),
                StructField(f"{prefix}_detection_count", IntegerType(), True),
                StructField(f"{prefix}_mean_dx", FloatType(), True),
                StructField(f"{prefix}_mean_dy", FloatType(), True),
                StructField(f"{prefix}_mean_speed_kmh", FloatType(), True),
                StructField(f"{prefix}_mean_acceleration", FloatType(), True),
                StructField(f"{prefix}_first_x1", FloatType(), True),
                StructField(f"{prefix}_first_y1", FloatType(), True),
                StructField(f"{prefix}_first_x2", FloatType(), True),
                StructField(f"{prefix}_first_y2", FloatType(), True),
                StructField(f"{prefix}_last_x1", FloatType(), True),
                StructField(f"{prefix}_last_y1", FloatType(), True),
                StructField(f"{prefix}_last_x2", FloatType(), True),
                StructField(f"{prefix}_last_y2", FloatType(), True),
            ]
        )

        return self.spark.createDataFrame(rows, schema=schema)
    
    def _get_ghost_player_ids(
    self, states: List[PlayerState], appearance_threshold: int = 160
    ) -> Set[int]:
        """
        Return player_ids that are likely ghost/artifact tracks:
        A) All appearances have zero total movement (dx == 0 and dy == 0)
        B) Total appearance count is below `appearance_threshold`
        """
        per_player: Dict[int, List[PlayerState]] = defaultdict(list)
        for state in states:
            per_player[state.player_id].append(state)

        ghost_ids: Set[int] = set()

        for player_id, detections in per_player.items():
            # Condition B: fewer than 160 appearances
            if len(detections) < appearance_threshold:
                ghost_ids.add(player_id)
                continue

            # Condition A: every detection has zero movement
            has_any_movement = any(
                (s.dx is not None and s.dx != 0.0)
                or (s.dy is not None and s.dy != 0.0)
                for s in detections
            )
            if not has_any_movement:
                ghost_ids.add(player_id)

        return ghost_ids

    def _has_temporal_overlap_expr(self, fs_a, fe_a, fs_b, fe_b):
        """Spark expression for temporal overlap check."""
        return greatest(fs_a, fs_b) <= least(fe_a, fe_b)

    def _timestamp_gap_expr(self, ts_end_a, ts_start_a, ts_end_b, ts_start_b):
        """Spark expression to calculate timestamp gap."""
        return when(ts_end_a <= ts_start_b, ts_start_b - ts_end_a).otherwise(
            ts_start_a - ts_end_b
        )

    def _direction_delta_expr(self, dx_a, dy_a, dx_b, dy_b):
        """Spark UDF call for direction delta."""
        return direction_delta_udf(dx_a, dy_a, dx_b, dy_b)

    def _iou_expr(self, ax1, ay1, ax2, ay2, bx1, by1, bx2, by2):
        """Spark UDF call for IoU calculation."""
        return bbox_iou_udf(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)

    def _physical_score_expr(self, dir_angle, iou, speed_delta, ts_gap):
        """Calculate composite physical score."""
        dir_score = dir_angle / lit(self.MAX_DIRECTION_ANGLE_DEG)
        iou_score = lit(1.0) - iou
        speed_score = speed_delta / lit(self.MAX_SPEED_DELTA_KMH)
        ts_score = ts_gap / lit(self.MAX_TIMESTAMP_GAP_S)

        return (
            lit(self.W_DIRECTION) * dir_score
            + lit(self.W_IOU) * iou_score
            + lit(self.W_SPEED) * speed_score
            + lit(self.W_TIMESTAMP) * ts_score
        )


physical_validator = PlayerPhysicalValidator()
