import math

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    acos,
    col,
    degrees,
    sqrt,
    pow,
    when,
    lit,
    coalesce,
    avg,
    expr,
    abs as sql_abs,
    broadcast,
)
from src.entities.utils.spark_instance import spark
from src.core.post_processing.v2.processing_config import PostProcessingConfig
import logfire
from typing import Optional, Dict, Tuple
import numpy as np


class FeatureEngineer:
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.spark = spark
        self.meter_per_pixel = 0.01  # valor por defecto, podría venir de configuración

    def build_possession_features(
        self, ball_df: DataFrame, player_df: DataFrame
    ) -> DataFrame:
        if ball_df is None or player_df is None:
            return self.spark.createDataFrame([], player_df.schema)

        ball_df = ball_df.select(
            col("frame_number"),
            col("cx").alias("ball_cx"),
            col("cy").alias("ball_cy"),
            col("vx").alias("vx_ball"),
            col("vy").alias("vy_ball"),
            col("confidence").alias("ball_confidence"),
        )
        joined = player_df.join(broadcast(ball_df), on="frame_number", how="inner")

        joined = joined.withColumn(
            "dist_m",
            sqrt(
                pow(col("cx") - col("ball_cx"), 2) + pow(col("cy") - col("ball_cy"), 2)
            )
            * self.meter_per_pixel,
        )
        joined = joined.withColumn("to_ball_x", col("ball_cx") - col("cx")).withColumn(
            "to_ball_y", col("ball_cy") - col("cy")
        )
        joined = joined.withColumn(
            "player_speed_norm", sqrt(pow(col("vx"), 2) + pow(col("vy"), 2))
        )
        joined = joined.withColumn(
            "ball_speed_norm", sqrt(pow(col("vx_ball"), 2) + pow(col("vy_ball"), 2))
        )
        joined = joined.withColumn(
            "dot_to_ball",
            col("vx") * col("to_ball_x") * self.meter_per_pixel
            + col("vy") * col("to_ball_y") * self.meter_per_pixel,
        )
        joined = joined.withColumn(
            "angle_cos",
            when(
                (col("player_speed_norm") > 0) & (col("dist_m") > 0),
                col("dot_to_ball") / (col("player_speed_norm") * col("dist_m")),
            ).otherwise(0.0),
        )
        joined = joined.withColumn("area_norm", col("area") / 10000.0)
        joined = joined.withColumn("player_conf", col("confidence"))
        joined = joined.withColumn("ball_conf", col("ball_confidence"))

        features = joined.select(
            "frame_number",
            "player_state_id",
            "player_id",
            "track_id",
            "team_id",
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "player_conf",
            "ball_conf",
            "area_norm",
            "has_ball",
        ).na.fill(9999.0)

        logfire.info(
            f"[FeatureEngineer] Built possession features: {features.count()} rows"
        )
        return features

    def compute_goal_line(
        self, goals_df: Optional[DataFrame], ball_df: Optional[DataFrame]
    ) -> Optional[Dict]:
        """Estima la recta de gol a partir de los centroides de los goles."""
        if goals_df is None or goals_df.isEmpty() or ball_df is None:
            return {"slope": 0.0, "intercept": 68.0, "is_vertical": False}

        # Unir con ball para obtener coordenadas en metros (si es posible)
        goals_with_ball = goals_df.join(
            ball_df.select(
                "frame_number",
                (col("cx") * self.meter_per_pixel).alias("gx_m"),
                (col("cy") * self.meter_per_pixel).alias("gy_m"),
            ),
            on="frame_number",
            how="left",
        )
        goals_with_ball = goals_with_ball.withColumn(
            "gx_m", coalesce(col("gx_m"), col("cx") * self.meter_per_pixel)
        ).withColumn("gy_m", coalesce(col("gy_m"), col("cy") * self.meter_per_pixel))

        points = goals_with_ball.select("gx_m", "gy_m").dropna().collect()
        if len(points) < 2:
            return {"slope": 0.0, "intercept": 68.0, "is_vertical": False}

        xs = np.array([p.gx_m for p in points])
        ys = np.array([p.gy_m for p in points])
        if np.var(xs) < 0.1:
            return {
                "slope": float("inf"),
                "intercept": float(np.mean(xs)),
                "is_vertical": True,
            }
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        logfire.info(f"[FeatureEngineer] Goal line: y = {m:.3f}x + {b:.3f}")
        return {"slope": float(m), "intercept": float(b), "is_vertical": False}

    def build_goal_linking_features(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        player_df: DataFrame,
        goal_line: Optional[Dict],
    ) -> Optional[DataFrame]:
        if any(df is None for df in [goals_df, ball_df, player_df]) or not goal_line:
            return None

        # Generar candidatos: posesiones que ocurren antes del gol
        g = goals_df.alias("g")
        p = player_df.alias("p")
        b = ball_df.alias("b")

        candidates = (
            p.join(broadcast(g), how="inner")
            .where(col("p.frame_number") < col("g.frame_number"))
            .where(
                (col("g.frame_number") - col("p.frame_number"))
                <= self.config.possession_lookback_frames
            )
            .where(
                (col("g.timestamp") - col("p.timestamp"))
                <= self.config.possession_lookback_seconds
            )
        )

        # Distancia del balón al centro del gol
        candidates = candidates.withColumn(
            "ball_to_goal_dist_px",
            sqrt(
                pow(col("p.ball_x") - col("g.cx"), 2)
                + pow(col("p.ball_y") - col("g.cy"), 2)
            ),
        )

        # Distancia a la línea de gol (usando coordenadas en metros)
        ball_at_goal = ball_df.select(
            col("frame_number").alias("g_ball_frame"),
            col("cx").alias("ball_at_goal_cx"),
            col("cy").alias("ball_at_goal_cy"),
            col("confidence").alias("ball_conf_at_goal"),
        )
        candidates = candidates.join(
            broadcast(ball_at_goal),
            col("g.frame_number") == col("g_ball_frame"),
            "left",
        )

        # Área de gol reducida
        candidates = candidates.withColumn(
            "inside_goal_area",
            when(
                (col("ball_at_goal_cx").isNotNull())
                & (
                    col("ball_at_goal_cx")
                    >= (col("g.x1") + col("g.x2")) / 2
                    - (col("g.x2") - col("g.x1"))
                    / 2
                    * math.sqrt(self.config.goal_area_reduction_factor)
                )
                & (
                    col("ball_at_goal_cx")
                    <= (col("g.x1") + col("g.x2")) / 2
                    + (col("g.x2") - col("g.x1"))
                    / 2
                    * math.sqrt(self.config.goal_area_reduction_factor)
                )
                & (
                    col("ball_at_goal_cy")
                    >= (col("g.y1") + col("g.y2")) / 2
                    - (col("g.y2") - col("g.y1"))
                    / 2
                    * math.sqrt(self.config.goal_area_reduction_factor)
                )
                & (
                    col("ball_at_goal_cy")
                    <= (col("g.y1") + col("g.y2")) / 2
                    + (col("g.y2") - col("g.y1"))
                    / 2
                    * math.sqrt(self.config.goal_area_reduction_factor)
                ),
                lit(True),
            ).otherwise(False),
        )

        # Distancia a la línea de gol en metros
        # Convertir a metros
        candidates = candidates.withColumn(
            "ball_at_goal_mx", col("ball_at_goal_cx") * self.meter_per_pixel
        ).withColumn("ball_at_goal_my", col("ball_at_goal_cy") * self.meter_per_pixel)
        if goal_line["is_vertical"]:
            candidates = candidates.withColumn(
                "dist_to_goal_line_m",
                sql_abs(col("ball_at_goal_mx") - lit(goal_line["intercept"])),
            )
        else:
            m = goal_line["slope"]
            b = goal_line["intercept"]
            candidates = candidates.withColumn(
                "dist_to_goal_line_m",
                sql_abs(m * col("ball_at_goal_mx") - col("ball_at_goal_my") + b)
                / sqrt(lit(m * m + 1)),
            )

        # Velocidad del balón en el frame del gol
        ball_speed_at_goal = ball_df.select(
            col("frame_number").alias("g_ball_speed_frame"),
            col("speed_kmh").alias("ball_speed"),
        )
        candidates = candidates.join(
            broadcast(ball_speed_at_goal),
            col("g.frame_number") == col("g_ball_speed_frame"),
            "left",
        )

        candidates = candidates.withColumn(
            "frame_gap", col("g.frame_number") - col("p.frame_number")
        )

        # Score heurístico para fallback
        candidates = candidates.withColumn(
            "score",
            when(
                col("inside_goal_area"),
                (col("frame_gap") / self.config.possession_lookback_frames) * 0.10
                + (col("ball_to_goal_dist_px") / 1000.0) * 0.10
                + (col("dist_to_goal_line_m") / 10.0) * 0.10
                + (1.0 - col("p.confidence")) * 0.20
                + (1.0 - col("ball_conf_at_goal")) * 0.20,
            ).otherwise(
                (col("frame_gap") / self.config.possession_lookback_frames) * 0.30
                + (col("ball_to_goal_dist_px") / 1000.0) * 0.25
                + (col("dist_to_goal_line_m") / 10.0) * 0.25
                + (1.0 - col("p.confidence")) * 0.20
                + 10.0
            ),
        )

        features = candidates.select(
            col("g.id").alias("goal_id"),
            col("p.frame_number"),
            col("p.player_id"),
            col("p.track_id"),
            col("p.team_id"),
            col("p.shirt_number"),
            col("p.confidence"),
            col("p.speed_kmh").alias("player_speed"),
            col("g.frame_number").alias("goal_frame"),
            col("g.timestamp").alias("goal_timestamp"),
            col("frame_gap"),
            col("ball_to_goal_dist_px"),
            col("dist_to_goal_line_m"),
            col("ball_speed"),
            col("inside_goal_area"),
            col("score"),
        ).na.fill(9999.0)

        logfire.info(
            f"[FeatureEngineer] Built goal linking candidates: {features.count()}"
        )
        return features

    def build_shot_features(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        player_df: DataFrame,
        goal_line: Optional[Dict],
    ) -> Optional[DataFrame]:
        if any(df is None for df in [goals_df, ball_df, player_df]) or not goal_line:
            return None

        p = player_df.alias("p")
        b = ball_df.alias("b")

        candidates = (
            p.join(broadcast(b), on="frame_number", how="inner")
            .where(col("b.speed_kmh") >= self.config.min_shot_speed_kmh)
            .where(col("b.speed_kmh") <= self.config.max_shot_speed_kmh)
        )

        # Distancia a la línea de gol (en metros)
        if goal_line["is_vertical"]:
            candidates = candidates.withColumn(
                "dist_to_goal_line",
                sql_abs(
                    col("b.cx") * self.meter_per_pixel - lit(goal_line["intercept"])
                ),
            )
        else:
            m = goal_line["slope"]
            b_intercept = goal_line["intercept"]
            candidates = candidates.withColumn(
                "dist_to_goal_line",
                sql_abs(
                    m * (col("b.cx") * self.meter_per_pixel)
                    - (col("b.cy") * self.meter_per_pixel)
                    + b_intercept
                )
                / sqrt(lit(m * m + 1)),
            )

        # Ángulo hacia el gol (usando centroide promedio)
        goal_center = self.compute_goal_centroid(goals_df)
        if goal_center:
            gx, gy = goal_center
            candidates = candidates.withColumn("goal_center_x", lit(gx))
            candidates = candidates.withColumn("goal_center_y", lit(gy))
            candidates = candidates.withColumn(
                "to_goal_x", col("goal_center_x") - (col("b.cx") * self.meter_per_pixel)
            ).withColumn(
                "to_goal_y", col("goal_center_y") - (col("b.cy") * self.meter_per_pixel)
            )
            candidates = (
                candidates.withColumn(
                    "dot_product",
                    col("b.vx") * col("to_goal_x") + col("b.vy") * col("to_goal_y"),
                )
                .withColumn("norm_v", sqrt(col("b.vx") ** 2 + col("b.vy") ** 2))
                .withColumn(
                    "norm_to_goal", sqrt(col("to_goal_x") ** 2 + col("to_goal_y") ** 2)
                )
                .withColumn(
                    "angle_to_goal",
                    when(
                        (col("norm_v") > 0) & (col("norm_to_goal") > 0),
                        sql_abs(
                            col("dot_product") / (col("norm_v") * col("norm_to_goal"))
                        ),
                    ).otherwise(0.0),
                )
            )
        else:
            candidates = candidates.withColumn("angle_to_goal", lit(0.0))

        candidates = candidates.withColumn(
            "angle_to_goal", degrees(acos(col("angle_to_goal")))
        )

        # Si hay goles, marcar si el balón está dentro del área de gol original
        if not goals_df.isEmpty():
            avg_rect = goals_df.select(
                avg("x1").alias("avg_x1"),
                avg("y1").alias("avg_y1"),
                avg("x2").alias("avg_x2"),
                avg("y2").alias("avg_y2"),
            ).collect()[0]
            if avg_rect.avg_x1 is not None:
                candidates = candidates.withColumn(
                    "inside_shot_area",
                    when(
                        (col("b.cx") >= avg_rect.avg_x1)
                        & (col("b.cx") <= avg_rect.avg_x2)
                        & (col("b.cy") >= avg_rect.avg_y1)
                        & (col("b.cy") <= avg_rect.avg_y2),
                        lit(True),
                    ).otherwise(False),
                )
            else:
                candidates = candidates.withColumn("inside_shot_area", lit(False))
        else:
            candidates = candidates.withColumn("inside_shot_area", lit(False))

        features = candidates.select(
            col("p.player_id"),
            col("p.track_id"),
            col("p.team_id"),
            col("p.shirt_number"),
            col("frame_number"),
            col("p.timestamp"),
            col("b.timestamp").alias("ball_timestamp"),
            col("b.speed_kmh").alias("ball_speed"),
            col("p.speed_kmh").alias("player_speed"),
            col("p.confidence"),
            col("dist_to_goal_line"),
            col("angle_to_goal"),
            col("inside_shot_area"),
        ).na.fill(99999.0)

        logfire.info(f"[FeatureEngineer] Built shot candidates: {features.count()}")
        return features

    def compute_goal_centroid(
        self, goals_df: Optional[DataFrame]
    ) -> Optional[Tuple[float, float]]:
        if goals_df is None or goals_df.isEmpty():
            return None
        avg_row = goals_df.select(
            avg("cx").alias("cx"), avg("cy").alias("cy")
        ).collect()[0]
        if avg_row.cx is not None and avg_row.cy is not None:
            return (
                avg_row.cx * self.meter_per_pixel,
                avg_row.cy * self.meter_per_pixel,
            )
        return None

    def build_trajectory_features(
        self, ball_df: Optional[DataFrame]
    ) -> Optional[DataFrame]:
        if ball_df is None:
            return None
        return ball_df.select("frame_number", "cx", "cy").withColumn(
            "frame2", col("frame_number") ** 2
        )
