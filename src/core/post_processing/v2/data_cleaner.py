import math
from typing import List, Optional, Tuple

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    avg,
    col,
    row_number,
    count,
    lit,
    lag,
    sqrt,
    pow,
    coalesce,
    expr,
    max as spark_max,
)
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
import logfire

from src.entities.utils.spark_instance import spark
from src.core.post_processing.v2.processing_config import PostProcessingConfig

class DataCleaner:
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.spark = spark

    def clean_ball_states(
            self, ball_df: DataFrame, goals_df: Optional[DataFrame] = None
        ) -> DataFrame:
        if ball_df is None or ball_df.isEmpty():
            return ball_df

        df = ball_df.filter(
            (col("confidence") >= self.config.min_ball_confidence)
            & (col("area") >= self.config.min_bbox_area)
            & (col("speed_kmh") <= self.config.max_ball_speed_kmh)
            & (col("acceleration") <= self.config.max_ball_accel_mss)
        )

        w = Window.partitionBy("match_id").orderBy("frame_number")
        df = df.withColumn("prev_cx", lag("cx", 1).over(w))
        df = df.withColumn("prev_cy", lag("cy", 1).over(w))
        df = df.withColumn(
            "jump",
            sqrt(
                pow(col("cx") - col("prev_cx"), 2) + pow(col("cy") - col("prev_cy"), 2)
            ),
        )
        df = df.filter(
            (col("jump") <= self.config.max_position_jump_px) | col("jump").isNull()
        )

        w_frame = Window.partitionBy("match_id", "frame_number").orderBy(
            col("confidence").desc()
        )
        df = (
            df.withColumn("rn", row_number().over(w_frame))
            .filter(col("rn") == 1)
            .drop("rn")
        )

        goal_zones = self._compute_goal_zones(goals_df)

        df = df.withColumn("speed_kmh", coalesce(col("speed_kmh"), lit(0.0)))
        df = df.withColumn(
            "is_static", col("speed_kmh") < self.config.static_speed_threshold
        )

        if goal_zones:
            near_goal_cond = None
            for gx, gy, radius in goal_zones:
                cond = sqrt(
                    pow(col("cx") - lit(gx), 2) + pow(col("cy") - lit(gy), 2)
                ) <= lit(radius)
                near_goal_cond = cond if near_goal_cond is None else (near_goal_cond | cond)
            df = df.withColumn("near_goal", near_goal_cond)
        else:
            df = df.withColumn("near_goal", lit(False))

        df = df.withColumn(
            "static_group",
            expr(
                "sum(case when (is_static and not near_goal) then 0 else 1 end) "
                "over (partition by match_id order by frame_number)"
            ),
        )
        df = df.withColumn(
            "static_count",
            count("*").over(Window.partitionBy("match_id", "static_group")),
        )
        df = df.filter(
            ~(
                (col("is_static") == True)
                & (col("near_goal") == False)
                & (col("static_count") >= self.config.static_frame_threshold)
            )
        )
        df = df.drop(
            "prev_cx", "prev_cy", "jump", "is_static",
            "static_group", "static_count", "near_goal",
        )

        logfire.info(f"[DataCleaner] Ball states cleaned: {df.count()} remaining")
        return df

    def _compute_goal_zones(
        self, goals_df: Optional[DataFrame]
    ) -> List[Tuple[float, float, float]]:
        """Returns (center_x, center_y, radius_px) per detected goal, expanded
        by goal_zone_margin_px, used to exempt a resting ball near/inside the
        net from the static-run pruning filter above."""
        if goals_df is None or goals_df.isEmpty():
            return []

        group_cols = ["cluster"] if "cluster" in goals_df.columns else []
        agg_exprs = [
            avg("cx").alias("center_cx"),
            avg("cy").alias("center_cy"),
            spark_max(col("x2") - col("x1")).alias("max_w"),
            spark_max(col("y2") - col("y1")).alias("max_h"),
        ]
        stats = (
            goals_df.groupBy(*group_cols).agg(*agg_exprs).collect()
            if group_cols
            else goals_df.agg(*agg_exprs).collect()
        )

        zones = []
        for row in stats:
            if row.center_cx is None or row.center_cy is None:
                continue
            half_diag = 0.5 * math.sqrt((row.max_w or 0.0) ** 2 + (row.max_h or 0.0) ** 2)
            zones.append((row.center_cx, row.center_cy, half_diag + self.config.goal_zone_margin_px))
        return zones

    def clean_goal_posts(self, posts_df: DataFrame) -> DataFrame:
        if posts_df is None or posts_df.isEmpty():
            return posts_df

        df = posts_df.filter(
            (col("confidence") >= self.config.min_post_confidence)
            & (col("area") >= self.config.min_post_area)
        )
        if df.isEmpty():
            return posts_df

        assembler = VectorAssembler(inputCols=["cx", "cy"], outputCol="features")
        k = 2
        if df.count() < 2:
            k = 1

        kmeans = KMeans(featuresCol="features", k=k, seed=42)
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(df)
        centers = model.stages[-1].clusterCenters()
        if len(centers) == 2:
            dist = math.sqrt(
                (centers[0][0] - centers[1][0]) ** 2
                + (centers[0][1] - centers[1][1]) ** 2
            )
            if dist < self.config.cluster_distance_threshold:
                k = 2
                if df.count() < 2:
                    k = 1

                kmeans = KMeans(featuresCol="features", k=k, seed=42)
                pipeline = Pipeline(stages=[assembler, kmeans])
                model = pipeline.fit(df)

        clustered = model.transform(df)
        clustered = clustered.withColumnRenamed("prediction", "cluster")

        cluster_counts = clustered.groupBy("cluster").agg(count("*").alias("cnt"))
        valid_clusters = [
            row.cluster
            for row in cluster_counts.filter(
                col("cnt") >= self.config.min_consecutive_frames
            ).collect()
        ]
        if not valid_clusters:
            logfire.warning("[DataCleaner] No valid goal post clusters found.")
            return posts_df
        df = clustered.filter(col("cluster").isin(valid_clusters))

        centroids = df.groupBy("cluster").agg(
            avg("cx").alias("center_cx"), avg("cy").alias("center_cy")
        )
        df = df.join(centroids, on="cluster")
        df = df.withColumn(
            "dist_to_center",
            sqrt(
                pow(col("cx") - col("center_cx"), 2)
                + pow(col("cy") - col("center_cy"), 2)
            ),
        )
        w = Window.partitionBy("cluster", "frame_number").orderBy(
            col("dist_to_center").asc()
        )
        df = (
            df.withColumn("rn", row_number().over(w))
            .filter(col("rn") == 1)
            .drop("rn", "features", "center_cx", "center_cy", "dist_to_center")
        )

        logfire.info(f"[DataCleaner] Goal posts cleaned: {df.count()} remaining")
        return df

    def clean_player_states(
        self, player_df: DataFrame
    ) -> DataFrame:
        if player_df is None:
            return player_df
        return player_df.filter(col("confidence") >= 0.2)
