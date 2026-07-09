from typing import List, Set, Optional, Tuple
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    when,
    row_number,
    count,
    avg,
    sqrt,
    pow,
)
from pyspark.sql.types import FloatType, IntegerType, StructType, StructField
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline, PipelineModel

from sqlmodel import Session, select, col as modelcol

from src.entities.models.soccer.goal_model import GoalModel
from src.entities.predictors.goal_predictor_base import GoalValidatorBase
from src.core.repository.goal_repository import GoalRepository


class GoalPostValidator(GoalValidatorBase):
    """
    Cleans goal post detections using spatial clustering and temporal consistency.
    This assumes the GoalModel table contains post detections (or we filter by some flag).
    """

    MIN_CONFIDENCE: float = 0.25  # promedio de detecciones de arcos
    MIN_BBOX_AREA: float = 50.0  # área mínima en píxeles²
    MAX_NUM_POSTS: int = 2  # máximo de arcos en la escena
    MIN_CONSECUTIVE_FRAMES: int = (
        3  # mínimo de detecciones para considerar un arco real
    )
    CLUSTER_DISTANCE_THRESHOLD: float = 150.0  # px, para separar clusters

    def validate(self, match_id: int, session: Session) -> None:
        logfire.info(f"[GoalPostValidator] Starting validation for match {match_id}")

        posts = list(GoalRepository.get_goals_by_match_id(match_id, session))
        if not posts:
            logfire.info("[GoalPostValidator] No posts found for match")
            return

        logfire.info(f"[GoalPostValidator] Loaded {len(posts)} raw post detections")

        posts_df = self._build_posts_dataframe(posts)

        valid_df = self._filter_invalid_posts(posts_df)
        valid_count = valid_df.count()
        logfire.info(f"[GoalPostValidator] {valid_count} valid posts after filtering")
        if valid_count == 0:
            logfire.warning("[GoalPostValidator] No valid posts remain")
            return

        featured_df = self._add_features(valid_df)

        cluster_model = self._train_spatial_cluster(featured_df)
        if cluster_model is None:
            logfire.warning(
                "[GoalPostValidator] Clustering failed, keeping all detections"
            )
            return

        clustered_df = cluster_model.transform(featured_df)
        clustered_df = clustered_df.withColumnRenamed("prediction", "cluster")

        cluster_counts = clustered_df.groupBy("cluster").agg(count("*").alias("cnt"))
        valid_clusters = (
            cluster_counts.filter(col("cnt") >= self.MIN_CONSECUTIVE_FRAMES)
            .select("cluster")
            .collect()
        )
        valid_cluster_ids = [row.cluster for row in valid_clusters]

        if not valid_cluster_ids:
            logfire.warning("[GoalPostValidator] No cluster meets minimum frame count.")
            return

        final_df = clustered_df.filter(col("cluster").isin(valid_cluster_ids))

        centroids = self._get_cluster_centroids(final_df, valid_cluster_ids)
        centroids_df = self.spark.createDataFrame(
            [(cid, cx, cy) for cid, cx, cy in centroids],
            StructType(
                [
                    StructField("cluster", IntegerType()),
                    StructField("center_cx", FloatType()),
                    StructField("center_cy", FloatType()),
                ]
            ),
        )
        final_df = final_df.join(centroids_df, on="cluster", how="left")
        final_df = final_df.withColumn(
            "dist_to_center",
            sqrt(
                pow(col("cx") - col("center_cx"), 2)
                + pow(col("cy") - col("center_cy"), 2)
            ),
        )

        w = Window.partitionBy("cluster", "frame_number").orderBy(
            col("dist_to_center").asc()
        )
        ranked = final_df.withColumn("rn", row_number().over(w))
        winners = ranked.filter(col("rn") == 1).drop("rn")

        winner_ids = {row.id for row in winners.select("id").collect()}
        logfire.info(
            f"[GoalPostValidator] Selected {len(winner_ids)} real post detections"
        )

        self._delete_non_winners(match_id, winner_ids, session)

        logfire.info(f"[GoalPostValidator] Validation completed for match {match_id}")


    def _build_posts_dataframe(self, posts: List[GoalModel]) -> DataFrame:
        rows = []
        for p in posts:
            area = self._compute_area(p.x1, p.y1, p.x2, p.y2)
            cx, cy = self._compute_centroid(p.x1, p.y1, p.x2, p.y2)
            rows.append(
                {
                    "id": p.id,
                    "match_id": p.match_id,
                    "frame_number": p.frame_number,
                    "timestamp": p.timestamp,
                    "confidence": p.confidence,
                    "x1": p.x1,
                    "y1": p.y1,
                    "x2": p.x2,
                    "y2": p.y2,
                    "area": area,
                    "cx": cx,
                    "cy": cy,
                }
            )
        schema = StructType(
            [
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
            ]
        )
        return self.spark.createDataFrame(rows, schema=schema)

    def _filter_invalid_posts(self, df: DataFrame) -> DataFrame:
        return (
            df.filter(col("x1").isNotNull())
            .filter(col("y1").isNotNull())
            .filter(col("x2").isNotNull())
            .filter(col("y2").isNotNull())
            .filter(col("area") >= self.MIN_BBOX_AREA)
            .filter(col("confidence") >= self.MIN_CONFIDENCE)
        )

    def _add_features(self, df: DataFrame) -> DataFrame:
        df = df.withColumn(
            "aspect_ratio",
            when(
                col("y2") - col("y1") > 0,
                (col("x2") - col("x1")) / (col("y2") - col("y1")),
            ).otherwise(0.0),
        )
        return df

    def _train_spatial_cluster(self, df: DataFrame) -> Optional[PipelineModel]:
        """
        Entrena KMeans con k=2; si los centros están muy juntos, usa k=1.
        """
        if df.count() < 10:
            return None

        assembler = VectorAssembler(inputCols=["cx", "cy"], outputCol="features")
        kmeans = KMeans(featuresCol="features", k=2, seed=42)
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(df)

        centers = model.stages[-1].clusterCenters()
        if len(centers) == 2:
            dist = (
                (centers[0][0] - centers[1][0]) ** 2
                + (centers[0][1] - centers[1][1]) ** 2
            ) ** 0.5
            if dist < self.CLUSTER_DISTANCE_THRESHOLD:
                logfire.info("[GoalPostValidator] Clusters too close, using k=1.")
                kmeans = KMeans(featuresCol="features", k=1, seed=42)
                pipeline = Pipeline(stages=[assembler, kmeans])
                model = pipeline.fit(df)

        logfire.info("[GoalPostValidator] Spatial clustering model trained.")
        return model

    def _get_cluster_centroids(
        self, df: DataFrame, cluster_ids: List[int]
    ) -> List[Tuple[int, float, float]]:
        """Calcula el centroide promedio (cx, cy) para cada cluster."""
        centroids = (
            df.filter(col("cluster").isin(cluster_ids))
            .groupBy("cluster")
            .agg(avg("cx").alias("center_cx"), avg("cy").alias("center_cy"))
            .collect()
        )
        return [(row.cluster, row.center_cx, row.center_cy) for row in centroids]

    def _delete_non_winners(
        self, match_id: int, winner_ids: Set[int], session: Session
    ) -> None:
        stmt = select(GoalModel).where(modelcol(GoalModel.match_id) == match_id)
        all_posts = list(session.exec(stmt).all())
        deleted = 0
        for post in all_posts:
            if post.id not in winner_ids:
                session.delete(post)
                deleted += 1
        session.commit()
        logfire.info(
            f"[GoalPostValidator] Deleted {deleted} non‑valid post detections, kept {len(winner_ids)}"
        )


goal_validator_cls = GoalPostValidator()
