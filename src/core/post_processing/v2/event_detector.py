from networkx import is_empty
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    count,
    row_number,
    when,
    lit,
    expr,
    explode,
    lead,
    lag,
    sum as spark_sum,
    max as spark_max,
    min as spark_min,
)

from pyspark.sql.types import DoubleType, IntegerType, StructType, StructField
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from src.entities.utils.spark_instance import spark
from src.core.post_processing.v2.processing_config import PostProcessingConfig
import logfire
from typing import Optional, List, Dict, Tuple

from src.core.services.global_value_store import value_store

class EventDetector:
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.spark = spark

    def predict_possession(
        self, features_df: DataFrame, model: Optional[PipelineModel] = None
    ) -> DataFrame:
        if features_df.isEmpty():
            return features_df

        if model is not None:
            pred = model.transform(features_df)
            prob = vector_to_array(col("probability"))
            pred = pred.withColumn("possession_prob", prob.getItem(1))
            pred = pred.withColumn(
                "possession_pred",
                when(
                    col("possession_prob") >= self.config.possession_threshold, 1
                ).otherwise(0),
            )
            # Seleccionar por frame el jugador con mayor probabilidad
            w = Window.partitionBy("frame_number").orderBy(
                col("possession_prob").desc()
            )
            final = pred.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)
            return final.select(
                "frame_number",
                "player_state_id",
                "player_id",
                "possession_pred",
                "possession_prob",
            )
        else:
            # Heurística: jugador más cercano al balón
            w = Window.partitionBy("frame_number").orderBy(col("dist_m").asc())
            heuristic = features_df.withColumn("rn", row_number().over(w)).filter(
                col("rn") == 1
            )
            heuristic = heuristic.withColumn("possession_pred", lit(1))
            heuristic = heuristic.withColumn("possession_prob", lit(1.0))
            return heuristic.select(
                "frame_number",
                "player_state_id",
                "player_id",
                "possession_pred",
                "possession_prob",
            )

    def predict_goal_scorer(
        self, candidates_df: Optional[DataFrame], model: Optional[PipelineModel] = None
    ) -> Tuple[List[Dict], Dict]:
        if candidates_df is None or candidates_df.isEmpty():
            return [], {}

        if model is not None:
            # Aplicar modelo
            scored = model.transform(candidates_df)
            prob = vector_to_array(col("probability"))
            scored = scored.withColumn("ml_score", prob.getItem(1))
            w = Window.partitionBy("goal_id").orderBy(col("ml_score").desc())
            best = scored.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)
            rows = best.collect()
        else:
            # Heurística: usar score más bajo (cercanía a gol, etc.)
            w = Window.partitionBy("goal_id").orderBy(col("score").asc())
            best = candidates_df.withColumn("rn", row_number().over(w)).filter(
                col("rn") == 1
            )
            rows = best.collect()

        results = []
        links = {}
        for row in rows:
            links[row.player_id] = links.get(row.player_id, 0) + 1
            results.append(
                {
                    "goal_id": row.goal_id,
                    "player_id": row.player_id,
                    "track_id": row.track_id,
                    "team_id": row.team_id,
                    "shirt_number": row.shirt_number,
                    "possession_frame": row.frame_number,
                    "goal_frame": row.goal_frame,
                    "frame_gap": row.frame_gap,
                    "dist_to_goal_line": (
                        row.dist_to_goal_line_m
                        if hasattr(row, "dist_to_goal_line_m")
                        else 0.0
                    ),
                    "score": row.score if hasattr(row, "score") else 0.0,
                    "is_goal": True,
                    "event_type": "goal",
                }
            )
        logfire.info(f"[EventDetector] Linked {len(results)} goals")
        return results, links

    def predict_shots(
        self, features_df: DataFrame, model: Optional[PipelineModel] = None
    ) -> DataFrame:
        if features_df.isEmpty():
            return features_df

        if model is not None:
            pred = model.transform(features_df)
            prob = vector_to_array(col("probability"))
            pred = pred.withColumn("shot_prob", prob.getItem(1))
            pred = pred.withColumn(
                "shot_pred",
                when(col("shot_prob") >= self.config.shot_threshold, 1).otherwise(0),
            )
            # Seleccionar mejores por jugador (el de mayor probabilidad)
            w = Window.partitionBy("player_id").orderBy(col("shot_prob").desc())
            final = pred.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)
            return final.select(
                "player_id",
                "track_id",
                "team_id",
                "shirt_number",
                "frame_number",
                "shot_pred",
                "shot_prob",
                "dist_to_goal_line",
                "ball_speed",
            )
        else:
            # Heurística: balón cerca de gol y con velocidad alta
            threshold_m = self.config.near_goal_cm_threshold / 100.0
            heuristic = (
                features_df.filter(
                    (col("dist_to_goal_line") <= threshold_m)
                    & (col("ball_speed") >= self.config.min_shot_speed_kmh)
                    & (col("angle_to_goal") <= 60)  # umbral de ángulo
                )
                .withColumn("shot_pred", lit(1))
                .withColumn("shot_prob", lit(1.0))
            )
            # Seleccionar por jugador
            w = Window.partitionBy("player_id").orderBy(col("dist_to_goal_line").asc())
            final = heuristic.withColumn("rn", row_number().over(w)).filter(
                col("rn") == 1
            )

            logfire.info("[EventDetector] Predicted shots: " + str(final.count()))
            return final.select(
                "player_id",
                "track_id",
                "team_id",
                "shirt_number",
                "frame_number",
                "shot_pred",
                "shot_prob",
                "dist_to_goal_line",
                "ball_speed",
            )

    def predict_trajectory(
        self,
        model_cx: Optional[PipelineModel],
        model_cy: Optional[PipelineModel],
        total_frames: int,
        match_id: int,
    ) -> Optional[DataFrame]:
        if model_cx is None or model_cy is None:
            return None
        # Generar DataFrame con todos los frames
        frames = [(match_id, i, i * i) for i in range(total_frames)]
        schema = StructType(
            [
                StructField("match_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("frame2", DoubleType(), True),
            ]
        )
        all_frames = self.spark.createDataFrame(frames, schema)
        pred_cx = model_cx.transform(all_frames).select(
            "frame_number", col("prediction").alias("pred_cx")
        )
        pred_cy = model_cy.transform(all_frames).select(
            "frame_number", col("prediction").alias("pred_cy")
        )
        traj = pred_cx.join(pred_cy, on="frame_number")
        return traj

    def interpolate_ball_states(
        self, winners_df: DataFrame, traj_df: Optional[DataFrame], total_frames: int
    ) -> Optional[DataFrame]:
        if winners_df.isEmpty():
            return None

        w = Window.partitionBy("match_id").orderBy("frame_number")
        df = (
            winners_df.withColumn("next_frame", lead("frame_number", 1).over(w))
            .withColumn("next_cx", lead("cx", 1).over(w))
            .withColumn("next_cy", lead("cy", 1).over(w))
        )
        df = df.withColumn("gap_size", col("next_frame") - col("frame_number") - 1)

        # Pequeños gaps: interpolación lineal
        small_gap = df.filter(
            (col("gap_size") <= self.config.max_interpolation_gap)
            & (col("gap_size") > 0)
        )
        small_gap = small_gap.withColumn(
            "interp_frames", expr(f"sequence(frame_number + 1, next_frame - 1)")
        )
        exploded = small_gap.withColumn("interp_frame", explode("interp_frames"))
        exploded = exploded.withColumn(
            "ratio",
            (col("interp_frame") - col("frame_number"))
            / (col("next_frame") - col("frame_number")),
        )
        interp_small = (
            exploded.withColumn(
                "interp_cx", col("cx") + (col("next_cx") - col("cx")) * col("ratio")
            )
            .withColumn(
                "interp_cy", col("cy") + (col("next_cy") - col("cy")) * col("ratio")
            )
            .select(
                "match_id",
                col("interp_frame").alias("frame_number"),
                "interp_cx",
                "interp_cy",
            )
        )

        # Grandes gaps: usar trayectoria si está disponible
        interp_large = None
        if traj_df is not None:
            large_gap = df.filter(
                (col("gap_size") > self.config.max_interpolation_gap)
                & (col("gap_size") > 0)
            )
            large_gap = large_gap.withColumn(
                "interp_frames", expr(f"sequence(frame_number + 1, next_frame - 1)")
            )
            exploded_large = large_gap.withColumn(
                "interp_frame", explode("interp_frames")
            )
            interp_large = exploded_large.select(
                "match_id", col("interp_frame").alias("frame_number")
            )
            interp_large = interp_large.join(
                traj_df, on="frame_number", how="inner"
            ).select(
                "match_id",
                "frame_number",
                col("pred_cx").alias("interp_cx"),
                col("pred_cy").alias("interp_cy"),
            )

        # Unir interpolados
        if interp_small is not None and interp_large is not None:
            interp_all = interp_small.union(interp_large)
        elif interp_small is not None:
            interp_all = interp_small
        elif interp_large is not None:
            interp_all = interp_large
        else:
            return None

        # Construir filas completas con valores por defecto (bbox de 20x20 alrededor del centro)
        interp_all = (
            interp_all.withColumn("x1", col("interp_cx") - 10)
            .withColumn("y1", col("interp_cy") - 10)
            .withColumn("x2", col("interp_cx") + 10)
            .withColumn("y2", col("interp_cy") + 10)
            .withColumn("confidence", lit(0.5))
            .withColumn("timestamp", col("frame_number") / int(value_store.get("frame_rate", 40)))
            .withColumn("area", lit(400.0))
            .withColumn("dx", lit(0.0))
            .withColumn("dy", lit(0.0))
            .withColumn("dx_meters", lit(0.0))
            .withColumn("dy_meters", lit(0.0))
            .withColumn("distance_meters", lit(0.0))
            .withColumn("speed_kmh", lit(0.0))
            .withColumn("vx", lit(0.0))
            .withColumn("vy", lit(0.0))
            .withColumn("ax", lit(0.0))
            .withColumn("ay", lit(0.0))
            .withColumn("acceleration", lit(0.0))
        )

        return interp_all

    def compute_possession_time(
        self, possession_df: DataFrame
    ) -> DataFrame:
        if possession_df.isEmpty():
            return possession_df
        # Asumimos que possession_df tiene frame_number, player_id, possession_pred (1/0)
        w = Window.partitionBy("player_id").orderBy("frame_number")
        changed = when(
            lag("possession_pred", 1).over(w).isNull()
            | (col("possession_pred") != lag("possession_pred", 1).over(w)),
            lit(1),
        ).otherwise(0)
        df = possession_df.withColumn("group", spark_sum(changed).over(w))
        groups = (
            df.filter(col("possession_pred") == 1)
            .groupBy("player_id", "group")
            .agg(
                (spark_max("frame_number") - spark_min("frame_number") + 2).alias(
                    "duration_frames"
                )
            )
        )
        # Convertir frames a segundos (asumiendo frame_rate)
        groups = groups.withColumn(
            "duration_sec", col("duration_frames") / self.config.frame_rate
        )
        groups = groups.filter(
            col("duration_sec") >= self.config.min_possession_seconds
        )
        result = groups.groupBy("player_id").agg(
            spark_sum("duration_sec").alias("possession_time")
        )
        return result

    def compute_shot_counts(self, shot_df: DataFrame) -> DataFrame:
        if shot_df.isEmpty():
            return shot_df
        return (
            shot_df.filter(col("shot_pred") == 1)
            .groupBy("player_id")
            .agg(count("*").alias("shots"))
        )
