from typing import List, Set
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    row_number,
    lit,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StructType,
    StructField,
)
from sqlmodel import Session, col as modelcol

from src.entities.predictors.goal_predictor_base import GoalValidatorBase
from src.entities.models.soccer.goal_model import GoalModel
from src.entities.models.app.union_model import UnionFind
from src.core.repository.goal_repository import GoalRepository

class GoalValidator(GoalValidatorBase):
    """
    Main goal validator using PySpark for distributed duplicate detection.

    Pipeline:
        1. Load all GoalModel records for a match
        2. Filter out invalid goals (null bbox, low confidence, tiny area)
        3. Group goals by temporal proximity (frame/timestamp windows)
        4. Within each temporal group, compute pairwise BBox IoU
        5. Cluster duplicates using Union-Find (IoU >= threshold)
        6. Select the most probable real goal per cluster
        7. Delete all non-winner goals from the database
    """

    def validate(self, match_id: int, session: Session) -> None:
        logfire.info(f"[GoalValidator] Starting validation for match {match_id}")

        goals = list(GoalRepository.get_goals_by_match_id(match_id, session))
        if not goals:
            logfire.info("[GoalValidator] No goals found for match")
            return

        logfire.info(f"[GoalValidator] Loaded {len(goals)} raw goals")

        goals_df = self._build_goals_dataframe(goals)

        valid_df = self._filter_invalid_goals(goals_df)
        valid_count = valid_df.count()
        logfire.info(f"[GoalValidator] {valid_count} valid goals after filtering")

        if valid_count == 0:
            logfire.warning("[GoalValidator] No valid goals remain after filtering")
            return

        temporal_groups = self._build_temporal_groups(valid_df)

        clusters = self._cluster_by_iou(temporal_groups)

        winners = self._select_winners(clusters)
        winner_ids = {row.id for row in winners.select("id").collect()}
        logfire.info(f"[GoalValidator] Selected {len(winner_ids)} real goals")

        self._delete_non_winners(match_id, winner_ids, session)

        logfire.info(f"[GoalValidator] Validation completed for match {match_id}")

    def _build_goals_dataframe(self, goals: List[GoalModel]) -> DataFrame:
        """Convert GoalModel objects to a Spark DataFrame with derived features."""
        rows = []
        for g in goals:
            area = self._compute_area(g.x1, g.y1, g.x2, g.y2)
            cx, cy = self._compute_centroid(g.x1, g.y1, g.x2, g.y2)
            logfire.info(f"[GoalValidator] Goal {g.id}: area={area}, cx={cx}, cy={cy}")
            rows.append({
                "id": g.id,
                "match_id": g.match_id,
                "frame_number": g.frame_number,
                "timestamp": g.timestamp,
                "confidence": g.confidence,
                "x1": g.x1,
                "y1": g.y1,
                "x2": g.x2,
                "y2": g.y2,
                "area": area,
                "cx": cx,
                "cy": cy,
            })

        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("match_id", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("x1", FloatType(), True),
            StructField("y1", FloatType(), True),
            StructField("x2", FloatType(), True),
            StructField("y2", FloatType(), True),
            StructField("area", FloatType(), True),
            StructField("cx", FloatType(), True),
            StructField("cy", FloatType(), True),
        ])

        return self.spark.createDataFrame(rows, schema=schema)

    def _filter_invalid_goals(self, goals_df: DataFrame) -> DataFrame:
        """
        Remove goals with:
            - Null or zero-area bbox
            - Confidence below MIN_CONFIDENCE
            - Any null coordinate
        """
        return (
            goals_df.filter(col("x1").isNotNull())
            .filter(col("y1").isNotNull())
            .filter(col("x2").isNotNull())
            .filter(col("y2").isNotNull())
            .filter(col("area") >= self.MIN_BBOX_AREA)
            .filter(col("confidence") >= self.MIN_CONFIDENCE)
        )


    def _build_temporal_groups(self, valid_df: DataFrame) -> DataFrame:
        """
        Assign a temporal group ID to each goal.
        Goals within MAX_FRAME_GAP frames and MAX_TIMESTAMP_GAP seconds
        of any other goal in the group belong to the same event.

        Uses a simple approach: sort by frame, then greedily group.
        """
        rows = valid_df.orderBy("frame_number", "timestamp").collect()
        if not rows:
            return valid_df.withColumn("temporal_group", lit(-1))

        groups = []
        current_group = 0
        last_frame = rows[0].frame_number
        last_ts = rows[0].timestamp

        for row in rows:
            frame_gap = abs(row.frame_number - last_frame)
            ts_gap = abs(row.timestamp - last_ts)

            if frame_gap > self.MAX_FRAME_GAP or ts_gap > self.MAX_TIMESTAMP_GAP:
                current_group += 1
                last_frame = row.frame_number
                last_ts = row.timestamp

            groups.append({
                "id": row.id,
                "temporal_group": current_group,
                "frame_number": row.frame_number,
                "timestamp": row.timestamp,
                "confidence": row.confidence,
                "x1": row.x1,
                "y1": row.y1,
                "x2": row.x2,
                "y2": row.y2,
                "area": row.area,
                "cx": row.cx,
                "cy": row.cy,
            })

        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("temporal_group", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("x1", FloatType(), True),
            StructField("y1", FloatType(), True),
            StructField("x2", FloatType(), True),
            StructField("y2", FloatType(), True),
            StructField("area", FloatType(), True),
            StructField("cx", FloatType(), True),
            StructField("cy", FloatType(), True),
        ])

        return self.spark.createDataFrame(groups, schema=schema)


    def _cluster_by_iou(self, grouped_df: DataFrame) -> DataFrame:
        """
        For each temporal group, compute pairwise IoU between all goals.
        Goals with IoU >= GOAL_IOU_THRESHOLD are clustered as the same physical goal.
        Returns DataFrame with cluster_id column.
        """
        groups = [row.temporal_group for row in grouped_df.select("temporal_group").distinct().collect()]

        all_clustered = []
        global_cluster_id = 0

        for group_id in groups:
            group_df = grouped_df.filter(col("temporal_group") == group_id)
            group_rows = group_df.collect()

            if len(group_rows) <= 1:
                for row in group_rows:
                    all_clustered.append({
                        "id": row.id,
                        "cluster_id": global_cluster_id,
                        "frame_number": row.frame_number,
                        "timestamp": row.timestamp,
                        "confidence": row.confidence,
                        "x1": row.x1,
                        "y1": row.y1,
                        "x2": row.x2,
                        "y2": row.y2,
                        "area": row.area,
                        "cx": row.cx,
                        "cy": row.cy,
                    })
                    global_cluster_id += 1
                continue

            uf = UnionFind()
            id_to_idx = {row.id: i for i, row in enumerate(group_rows)}

            for i, row_a in enumerate(group_rows):
                for j, row_b in enumerate(group_rows):
                    if i >= j:
                        continue
                    iou = self._compute_iou(
                        row_a.x1, row_a.y1, row_a.x2, row_a.y2,
                        row_b.x1, row_b.y1, row_b.x2, row_b.y2,
                    )
                    if iou >= self.GOAL_IOU_THRESHOLD:
                        uf.union(row_a.id, row_b.id)

            clusters = uf.groups()
            for root, members in clusters.items():
                for member in members:
                    row = group_rows[id_to_idx[member]]
                    all_clustered.append({
                        "id": row.id,
                        "cluster_id": global_cluster_id,
                        "frame_number": row.frame_number,
                        "timestamp": row.timestamp,
                        "confidence": row.confidence,
                        "x1": row.x1,
                        "y1": row.y1,
                        "x2": row.x2,
                        "y2": row.y2,
                        "area": row.area,
                        "cx": row.cx,
                        "cy": row.cy,
                    })
                global_cluster_id += 1

        schema = StructType([
            StructField("id", IntegerType(), True),
            StructField("cluster_id", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("x1", FloatType(), True),
            StructField("y1", FloatType(), True),
            StructField("x2", FloatType(), True),
            StructField("y2", FloatType(), True),
            StructField("area", FloatType(), True),
            StructField("cx", FloatType(), True),
            StructField("cy", FloatType(), True),
        ])

        return self.spark.createDataFrame(all_clustered, schema=schema)

    @staticmethod
    def _compute_iou(
        ax1: float, ay1: float, ax2: float, ay2: float,
        bx1: float, by1: float, bx2: float, by2: float,
    ) -> float:
        """Compute IoU between two bboxes (pure Python fallback)."""
        ix1 = max(ax1 or 0, bx1 or 0)
        iy1 = max(ay1 or 0, by1 or 0)
        ix2 = min(ax2 or 0, bx2 or 0)
        iy2 = min(ay2 or 0, by2 or 0)

        inter_w = max(0.0, ix2 - ix1)
        inter_h = max(0.0, iy2 - iy1)
        inter = inter_w * inter_h

        if inter == 0.0:
            return 0.0

        area_a = max(0.0, (ax2 or 0) - (ax1 or 0)) * max(0.0, (ay2 or 0) - (ay1 or 0))
        area_b = max(0.0, (bx2 or 0) - (bx1 or 0)) * max(0.0, (by2 or 0) - (by1 or 0))
        union = area_a + area_b - inter

        return inter / union if union > 0.0 else 0.0


    def _select_winners(self, clustered_df: DataFrame) -> DataFrame:
        """
        For each cluster, select the most probable real goal.
        Ranking: highest confidence > largest area > earliest timestamp
        """
        w_cluster = Window.partitionBy("cluster_id").orderBy(
            col("confidence").desc(),
            col("area").desc(),
            col("timestamp").asc(),
        )

        ranked = clustered_df.withColumn("rn", row_number().over(w_cluster))
        winners = ranked.filter(col("rn") == 1).drop("rn")

        cluster_count = clustered_df.select("cluster_id").distinct().count()
        winner_count = winners.count()
        logfire.info(
            f"[GoalValidator] {cluster_count} clusters, {winner_count} winners selected"
        )

        return winners

    def _delete_non_winners(
        self, match_id: int, winner_ids: Set[int], session: Session
    ) -> None:
        """Delete all goals that are not winners."""
        from sqlmodel import select
        stmt = select(GoalModel).where(modelcol(GoalModel.match_id) == match_id)
        all_goals = list(session.exec(stmt).all())

        deleted = 0
        for goal in all_goals:
            if goal.id not in winner_ids:
                session.delete(goal)
                deleted += 1

        session.commit()
        logfire.info(f"[GoalValidator] Deleted {deleted} duplicate goals, kept {len(winner_ids)}")


goal_validator_cls = GoalValidator()