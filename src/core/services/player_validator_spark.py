from typing import Dict, Set
import logfire

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, greatest, least, abs as sql_abs
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructType,
    StructField,
)

from sqlmodel import Session

from src.entities.models.app.player_validator_models import TrackSummary
from src.entities.models.app.union_model import UnionFind
from src.entities.models.pyspark.udfs import (
    bbox_iou_udf,
    color_distance_udf,
    euclidean_distance_udf,
)
from src.entities.services.player_validator_base_spark import (
    PlayerValidatorBase,
)
from src.core.repository.player_states_repository import PlayerStatesRepository


class PlayerValidator(PlayerValidatorBase):
    """
    Main validator using Spark for distributed track merging.
    Replaces nested loops with Spark SQL operations for significant speedup.
    """

    def validate(self, match_id: int, total_frames: int, session: Session) -> None:
        logfire.info(f"[PlayerValidator] Starting validation for match {match_id}")

        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=total_frames,
            session=session,
        )

        states_df = self._build_centroids(list(states), session)

        if states_df.count() == 0:
            logfire.info("[PlayerValidator] No data to validate")
            return

        correct_centroids = states_df.filter(col("tracker_id") <= 22)
        incorrect_centroids = states_df.filter(col("tracker_id") > 22)

        if correct_centroids.count() == 0 or incorrect_centroids.count() == 0:
            logfire.info("[PlayerValidator] No data to validate")
            return

        correct_summaries = self._build_track_summaries(correct_centroids)
        incorrect_summaries = self._build_track_summaries(incorrect_centroids)

        ghost_ids: Set[int] = {
            pid
            for pid, s in incorrect_summaries.items()
            if s.detection_count <= self.GHOST_TRACK_THRESHOLD
        }

        logfire.info(
            f"[PlayerValidator] Correct tracks: {len(correct_summaries)} | "
            f"Incorrect tracks: {len(incorrect_summaries)} | "
            f"Ghost tracks: {ghost_ids}"
        )

        correct_df = self._summaries_to_dataframe(correct_summaries, "correct")
        incorrect_df = self._summaries_to_dataframe(incorrect_summaries, "incorrect")

        incorrect_df = incorrect_df.withColumn(
            "is_ghost", col("incorrect_player_id").isin(list(ghost_ids))
        )

        # incorrect_df = (
        #     incorrect_df.withColumnRenamed("early_cx", "incorrect_first_cx")
        #     .withColumnRenamed("early_cy", "incorrect_first_cy")
        #     .withColumnRenamed("late_cx", "incorrect_last_cx")
        #     .withColumnRenamed("late_cy", "incorrect_last_cy")
        # )

        # correct_df = (
        #     correct_df.withColumnRenamed("early_cx", "correct_first_cx")
        #     .withColumnRenamed("early_cy", "correct_first_cy")
        #     .withColumnRenamed("late_cx", "correct_last_cx")
        #     .withColumnRenamed("late_cy", "correct_last_cy")
        # )

        candidates = incorrect_df.crossJoin(correct_df)

        candidates = candidates.withColumn(
            "frame_gap",
            col("incorrect_frame_start") - col("correct_frame_end"),
        ).withColumn(
            "timestamp_gap",
            col("incorrect_timestamp_start") - col("correct_timestamp_end"),
        )

        filtered_candidates = (
            candidates.filter(col("incorrect_frame_start") > col("correct_frame_end"))
            .filter(
                ~self._has_frame_overlap_expr(
                    col("incorrect_frame_start"),
                    col("incorrect_frame_end"),
                    col("correct_frame_start"),
                    col("correct_frame_end"),
                )
            )
            .filter(col("frame_gap") > 0)
            .filter(col("frame_gap") <= self.MAX_FRAME_GAP)
            .filter(col("timestamp_gap") > 0)
            .filter(
                sql_abs(col("correct_avg_width") - col("incorrect_avg_width"))
                < self.MAX_WIDTH_DIFF
            )
            .filter(
                sql_abs(col("correct_avg_height") - col("incorrect_avg_height"))
                < self.MAX_HEIGHT_DIFF
            )
            .filter(
                self._position_dist_expr(
                    col("incorrect_first_cx"),
                    col("incorrect_first_cy"),
                    col("correct_last_cx"),
                    col("correct_last_cy"),
                )
                < (self.BASE_DISTANCE + self.PIXELS_PER_FRAME * col("frame_gap"))
            )
            .withColumn(
                "positional_iou",
                self._calculate_iou_diff(
                    col("incorrect_first_x1"),
                    col("incorrect_first_y1"),
                    col("incorrect_first_x2"),
                    col("incorrect_first_y2"),
                    col("correct_last_x1"),
                    col("correct_last_y1"),
                    col("correct_last_x2"),
                    col("correct_last_y2"),
                ),
            )
            .filter(col("positional_iou") > self.IOU_THRESHOLD)
            .filter(
                self._color_dist_expr(
                    col("incorrect_avg_color"), col("correct_avg_color")
                )
                < self.MAX_COLOR_DISTANCE
            )
        )

        scored = filtered_candidates.withColumn(
            "base_score",
            self._composite_score_expr(
                self._position_dist_expr(
                    col("incorrect_first_cx"),
                    col("incorrect_first_cy"),
                    col("correct_last_cx"),
                    col("correct_last_cy"),
                ),
                self._position_dist_expr(
                    col("incorrect_first_cx"),
                    col("incorrect_first_cy"),
                    col("correct_mid_cx"),
                    col("correct_mid_cy"),
                ),
                col("frame_gap")
            ),
        )

        scored_list = [
            (row.base_score, row.incorrect_player_id, row.correct_player_id)
            for row in scored.orderBy("base_score").collect()
        ]

        if not scored_list:
            logfire.info("[PlayerValidator] No valid candidates")
            session.commit()
            return

        uf = UnionFind()
        merged_incorrect: Set[int] = set()

        for score, incorrect_id, correct_id in scored_list:
            if score > self.MERGE_SCORE_THRESHOLD:
                break
            if incorrect_id in merged_incorrect:
                continue

            uf.union(correct_id, incorrect_id)
            merged_incorrect.add(incorrect_id)

            logfire.info(
                f"[PlayerValidator] Scheduling merge {incorrect_id} → {correct_id} "
                f"(score={score:.2f})"
            )

        merge_groups = uf.groups()
        total_merged = 0

        for root, members in merge_groups.items():
            for member in members:
                if member == root:
                    continue
                PlayerStatesRepository.merge_states(root, member, session)
                total_merged += 1

        session.commit()

        logfire.info(
            f"[PlayerValidator] Validation completed. "
            f"Merged {total_merged} tracks across {len(merge_groups)} groups."
        )

    def _build_centroids_df(self, states_df: DataFrame) -> DataFrame:
        """Build centroids DataFrame from states (already have cx, cy)."""
        return states_df.select(
            "player_id", "tracker_id", "cx", "cy", "color", "confidence", "frame_number"
        )

    def _summaries_to_dataframe(
        self, summaries: Dict[int, TrackSummary], prefix: str
    ) -> DataFrame:
        """Convert summary dict to Spark DataFrame with early/late positions."""
        rows = []
        for player_id, summary in summaries.items():
            rows.append(
                {
                    f"{prefix}_player_id": player_id,
                    f"{prefix}_frame_start": summary.frame_start,
                    f"{prefix}_frame_end": summary.frame_end,
                    f"{prefix}_detection_count": summary.detection_count,
                    f"{prefix}_avg_color": summary.avg_color,
                    f"{prefix}_avg_width": summary.avg_width,
                    f"{prefix}_avg_height": summary.avg_height,
                    f"{prefix}_timestamp_start": summary.timestamp_start,
                    f"{prefix}_timestamp_end": summary.timestamp_end,
                    f"{prefix}_first_x1": summary.first_x1,
                    f"{prefix}_first_y1": summary.first_y1,
                    f"{prefix}_first_x2": summary.first_x2,
                    f"{prefix}_first_y2": summary.first_y2,
                    f"{prefix}_first_cx": summary.first_cx,
                    f"{prefix}_first_cy": summary.first_cy,
                    f"{prefix}_mid_x1": summary.mid_x1,
                    f"{prefix}_mid_y1": summary.mid_y1,
                    f"{prefix}_mid_x2": summary.mid_x2,
                    f"{prefix}_mid_y2": summary.mid_y2,
                    f"{prefix}_mid_cx": summary.mid_cx,
                    f"{prefix}_mid_cy": summary.mid_cy,
                    f"{prefix}_last_x1": summary.last_x1,
                    f"{prefix}_last_y1": summary.last_y1,
                    f"{prefix}_last_x2": summary.last_x2,
                    f"{prefix}_last_y2": summary.last_y2,
                    f"{prefix}_last_cx": summary.last_cx,
                    f"{prefix}_last_cy": summary.last_cy,
                }
            )

        schema = StructType(
            [
                StructField(f"{prefix}_player_id", IntegerType(), True),
                StructField(f"{prefix}_frame_start", IntegerType(), True),
                StructField(f"{prefix}_frame_end", IntegerType(), True),
                StructField(f"{prefix}_detection_count", IntegerType(), True),
                StructField(f"{prefix}_avg_color", StringType(), True),
                StructField(f"{prefix}_timestamp_start", FloatType(), True),
                StructField(f"{prefix}_timestamp_end", FloatType(), True),
                StructField(f"{prefix}_avg_width", FloatType(), True),
                StructField(f"{prefix}_avg_height", FloatType(), True),
                StructField(f"{prefix}_first_x1", FloatType(), True),
                StructField(f"{prefix}_first_y1", FloatType(), True),
                StructField(f"{prefix}_first_x2", FloatType(), True),
                StructField(f"{prefix}_first_y2", FloatType(), True),
                StructField(f"{prefix}_first_cx", FloatType(), True),
                StructField(f"{prefix}_first_cy", FloatType(), True),
                StructField(f"{prefix}_mid_x1", FloatType(), True),
                StructField(f"{prefix}_mid_y1", FloatType(), True),
                StructField(f"{prefix}_mid_x2", FloatType(), True),
                StructField(f"{prefix}_mid_y2", FloatType(), True),
                StructField(f"{prefix}_mid_cx", FloatType(), True),
                StructField(f"{prefix}_mid_cy", FloatType(), True),
                StructField(f"{prefix}_last_x1", FloatType(), True),
                StructField(f"{prefix}_last_y1", FloatType(), True),
                StructField(f"{prefix}_last_x2", FloatType(), True),
                StructField(f"{prefix}_last_y2", FloatType(), True),
                StructField(f"{prefix}_last_cx", FloatType(), True),
                StructField(f"{prefix}_last_cy", FloatType(), True),
            ]
        )

        return self.spark.createDataFrame(rows, schema=schema)

    def _has_frame_overlap_expr(self, fs_a, fe_a, fs_b, fe_b):
        """Spark expression for frame overlap check."""
        return greatest(fs_a, fs_b) <= least(fe_a, fe_b)

    def _color_dist_expr(self, color_a, color_b):
        """Spark UDF call for color distance."""
        return color_distance_udf(color_a, color_b)

    def _position_dist_expr(self, cx1, cy1, cx2, cy2):
        """Spark UDF call for position distance."""
        return euclidean_distance_udf(cx1, cy1, cx2, cy2)

    def _composite_score_expr(self, first_pos_dist, mid_pos_dist, gap):
        """Calculate composite score as Spark expression."""
        position_score = first_pos_dist / self.MAX_DISTANCE
        mid_score = mid_pos_dist / self.MAX_DISTANCE
        gap_score = gap / self.MAX_FRAME_GAP

        return (
            0.45 * position_score
            + 0.45 * mid_score
            + 0.1 * gap_score
        )

    def _calculate_iou_diff(
        self,
        ax1,
        ay1,
        ax2,
        ay2,
        bx1,
        by1,
        bx2,
        by2,
    ):
        return bbox_iou_udf(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)


player_validator_cls = PlayerValidator()
