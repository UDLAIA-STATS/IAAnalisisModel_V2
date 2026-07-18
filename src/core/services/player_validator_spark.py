from typing import Dict, Set
import logfire

from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    broadcast,
    col,
    greatest,
    least,
    abs as sql_abs,
    sqrt,
    pow,
    struct,
    min as sql_min,
    udf,
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

        correct_centroids = states_df.filter(col("tracker_id") <= 22).cache()
        incorrect_centroids = states_df.filter(col("tracker_id") > 22).cache()

        if correct_centroids.count() == 0 or incorrect_centroids.count() == 0:
            logfire.info("[PlayerValidator] No data to validate")
            correct_centroids.unpersist()
            incorrect_centroids.unpersist()
            return

        correct_df = self._build_track_summaries(correct_centroids)
        incorrect_df = self._build_track_summaries(incorrect_centroids)

        correct_df = correct_df.select(
            [col(c).alias(f"correct_{c}") for c in correct_df.columns]
        )
        incorrect_df = incorrect_df.select(
            [col(c).alias(f"incorrect_{c}") for c in incorrect_df.columns]
        )

        correct_df.cache()
        incorrect_df.cache()

        ghost_ids_df = (
            incorrect_df.filter(
                col("incorrect_detection_count") <= self.GHOST_TRACK_THRESHOLD
            )
            .select("incorrect_player_id")
            .distinct()
        )
        ghost_ids = set(
            [int(row.incorrect_player_id) for row in ghost_ids_df.collect()]
        )

        logfire.info(
            f"[PlayerValidator] Correct tracks: {correct_df.count()} | "
            f"Incorrect tracks: {incorrect_df.count()} | "
            f"Ghost tracks: {ghost_ids}"
        )

        incorrect_df = incorrect_df.withColumn(
            "is_ghost", col("incorrect_player_id").isin(list(ghost_ids))
        )

        candidates = incorrect_df.crossJoin(broadcast(correct_df))

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
                bbox_iou_udf(
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
                color_distance_udf(
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
                col("frame_gap"),
            ),
        )

        scored_filtered = scored.filter(col("base_score") <= self.MERGE_SCORE_THRESHOLD)
        best_candidates_df = (
            scored_filtered.groupBy("incorrect_player_id")
            .agg(sql_min(struct("base_score", "correct_player_id")).alias("best"))
            .select(
                col("incorrect_player_id"),
                col("best.base_score").alias("best_score"),
                col("best.correct_player_id").alias("best_correct_id"),
            )
        )

        best_candidates_list = [
            (row.best_score, row.incorrect_player_id, row.best_correct_id)
            for row in best_candidates_df.orderBy("best_score").collect()
        ]

        if not best_candidates_list:
            logfire.info("[PlayerValidator] No valid candidates found")
            correct_df.unpersist()
            incorrect_df.unpersist()
            correct_centroids.unpersist()
            incorrect_centroids.unpersist()
            return

        uf = UnionFind()
        merged_incorrect: Set[int] = set()

        for score, incorrect_id, correct_id in best_candidates_list:
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
        correct_df.unpersist()
        incorrect_df.unpersist()
        correct_centroids.unpersist()
        incorrect_centroids.unpersist()

        logfire.info(
            f"[PlayerValidator] Validation completed. "
            f"Merged {total_merged} tracks across {len(merge_groups)} groups."
        )

    def _build_centroids_df(self, states_df: DataFrame) -> DataFrame:
        """Build centroids DataFrame from states (already have cx, cy)."""
        return states_df.select(
            "player_id", "tracker_id", "cx", "cy", "color", "confidence", "frame_number"
        )

    def _has_frame_overlap_expr(self, fs_a, fe_a, fs_b, fe_b):
        """Spark expression for frame overlap check."""
        return greatest(fs_a, fs_b) <= least(fe_a, fe_b)

    def _position_dist_expr(self, cx1, cy1, cx2, cy2):
        """Spark call for position distance."""
        return sqrt(pow(cx1 - cx2, 2) + pow(cy1 - cy2, 2))

    def _composite_score_expr(self, first_pos_dist, mid_pos_dist, gap):
        """Calculate composite score as Spark expression."""
        position_score = first_pos_dist / self.MAX_DISTANCE
        mid_score = mid_pos_dist / self.MAX_DISTANCE
        gap_score = gap / self.MAX_FRAME_GAP

        return 0.45 * position_score + 0.45 * mid_score + 0.1 * gap_score


player_validator_cls = PlayerValidator()
