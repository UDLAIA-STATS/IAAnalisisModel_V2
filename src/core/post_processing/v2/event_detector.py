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
    coalesce,
    get as array_get,
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
            pred = pred.withColumn(
                "possession_prob", coalesce(array_get(prob, lit(1.0)), lit(0.0))
            )
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

    def _consolidate_goal_winners(
        self, best: DataFrame, order_by_cols: list
    ) -> DataFrame:
        """
        `best` has one row per raw goal_id (one candidate per raw goal-post
        detection row). Multiple raw goal_ids commonly correspond to the SAME
        real scoring event, since the goal structure/ball stays detectable
        across a stretch of nearby frames, not just one exact instant.
        Cluster those by frame-gap and keep only the single best-ranked row
        per cluster - "the real goal" / winner - so one physical goal isn't
        counted (and doesn't inflate team_goals) once per raw detection.
        """
        w_seq = Window.orderBy("goal_frame")
        prev_frame = lag("goal_frame", 1).over(w_seq)
        new_event = when(
            prev_frame.isNull()
            | ((col("goal_frame") - prev_frame) > self.config.goal_event_gap_frames),
            1,
        ).otherwise(0)
        clustered = best.withColumn(
            "real_goal_event_id", spark_sum(new_event).over(w_seq)
        )

        w_rep = Window.partitionBy("real_goal_event_id").orderBy(*order_by_cols)
        return (
            clustered.withColumn("rn2", row_number().over(w_rep))
            .filter(col("rn2") == 1)
            .drop("rn2", "real_goal_event_id")
        )

    def consolidate_goal_frames(self, goals_df: Optional[DataFrame]) -> List[int]:
        if goals_df is None or goals_df.isEmpty():
            return []
        frames_df = goals_df.select("frame_number").distinct()
        w = Window.orderBy("frame_number")
        prev_frame = lag("frame_number", 1).over(w)
        new_event = when(
            prev_frame.isNull()
            | ((col("frame_number") - prev_frame) > self.config.goal_event_gap_frames),
            1,
        ).otherwise(0)
        clustered = frames_df.withColumn("event_group", spark_sum(new_event).over(w))
        reps = clustered.groupBy("event_group").agg(
            spark_min("frame_number").alias("frame_number")
        )
        return [row.frame_number for row in reps.select("frame_number").collect()]

    def predict_goal_scorer(
        self,
        candidates_df: Optional[DataFrame],
        model: Optional[PipelineModel] = None,
        shot_features: Optional[DataFrame] = None,
        shot_model: Optional[PipelineModel] = None,
        goal_frames: Optional[List[int]] = None,
    ) -> Dict[str, List[Dict]]:
        """
        Predice el anotador de cada gol y también detecta tiros cercanos.
        Devuelve un diccionario con dos listas: 'goals' y 'shots'.
        """
        result = {"goals": [], "shots": []}

        if candidates_df is not None and not candidates_df.isEmpty():
            candidates_df = candidates_df.filter(col("score") < 1000)

            if candidates_df.isEmpty():
                return result

            if model is not None:
                scored = model.transform(candidates_df)
                prob = vector_to_array(col("probability"))
                scored = scored.withColumn(
                    "ml_score", coalesce(array_get(prob, lit(1.0)), lit(0.0))
                )
                w = Window.partitionBy("goal_id").orderBy(col("ml_score").desc())
                best = scored.withColumn("rn", row_number().over(w)).filter(
                    col("rn") == 1
                )
                best = self._consolidate_goal_winners(best, [col("ml_score").desc()])
                rows = best.collect()
            else:
                w = Window.partitionBy("goal_id").orderBy(col("score").asc())
                best = candidates_df.withColumn("rn", row_number().over(w)).filter(
                    col("rn") == 1
                )
                best = self._consolidate_goal_winners(best, [col("score").asc()])
                rows = best.collect()
            

            avg_dist = sum(
                x.dist_to_goal_line_m
                for x in sorted(rows, key=lambda x: x.dist_to_goal_line_m)
            ) / len(rows)
            logfire.info(f"[EventDetector] Average distance to goal line: {avg_dist}")
            rows_added = set()

            for row in rows:
                if avg_dist == 0:
                    break
                
                if row.dist_to_goal_line_m > avg_dist * 1.25 or row.dist_to_goal_line_m <= 0.5:
                    result["goals"].append(
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
                            "event_type": "real_goal",
                        }
                    )
                    rows_added.add(row.goal_id)

            for row in rows:
                if row.goal_id in rows_added:
                    continue

                result["goals"].append(
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

        if shot_features is not None and not shot_features.isEmpty():
            if goal_frames is None:
                goal_frames = [g["goal_frame"] for g in result["goals"]]
            shots = self._detect_shots_from_features(
                shot_features, shot_model, goal_frames
            )
            result["shots"] = shots

        logfire.info(
            f"[EventDetector] Detected {len(result['goals'])} goals and {len(result['shots'])} shots"
        )
        return result

    def predict_shots(
        self,
        features_df: DataFrame,
        model: Optional[PipelineModel] = None,
        goal_frames: Optional[List[int]] = None,
    ) -> DataFrame:
        if features_df.isEmpty():
            return features_df

        if model is not None:
            pred = model.transform(features_df)
            prob = vector_to_array(col("probability"))
            pred = pred.withColumn(
                "shot_prob", coalesce(array_get(prob, lit(1.0)), lit(0.0))
            )
            pred = pred.withColumn(
                "shot_pred",
                when(col("shot_prob") >= self.config.shot_threshold, 1).otherwise(0),
            )
            qualifying = pred.filter(col("shot_pred") == 1)
            final = self._cluster_shot_events(qualifying, [col("shot_prob").desc()])
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
            threshold_m = self.config.near_goal_cm_threshold / 100.0
            heuristic = (
                features_df.filter(
                    (col("inside_shot_area") == False)
                    & (col("dist_to_goal_line") > threshold_m)
                )
                .withColumn("shot_pred", lit(1))
                .withColumn("shot_prob", lit(1.0))
            )
            if goal_frames:
                heuristic = heuristic.filter(~col("frame_number").isin(goal_frames))

            final = self._cluster_shot_events(
                heuristic,
                [col("dist_to_goal_line").desc()],
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
        frames = [(match_id, i, float(i * i)) for i in range(total_frames)]
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
            .withColumn(
                "timestamp",
                col("frame_number") / int(value_store.get("frame_rate", 40)),
            )
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

    def compute_possession_time(self, possession_df: DataFrame) -> DataFrame:
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

    def _detect_shots_from_features(
        self,
        shot_features: DataFrame,
        shot_model: Optional[PipelineModel] = None,
        goal_frames: Optional[List[int]] = None,
    ) -> List[Dict]:
        if shot_features.isEmpty():
            return []

        if shot_model is not None:
            pred = shot_model.transform(shot_features)
            prob = vector_to_array(col("probability"))
            pred = pred.withColumn(
                "shot_prob", coalesce(array_get(prob, lit(1.0)), lit(0.0))
            )
            pred = pred.withColumn(
                "shot_pred",
                when(col("shot_prob") >= self.config.shot_threshold, 1).otherwise(0),
            )
            qualifying = pred.filter(col("shot_pred") == 1)
            final = self._cluster_shot_events(qualifying, [col("shot_prob").desc()])
            rows = final.collect()
            return [
                {
                    "player_id": row.player_id,
                    "track_id": row.track_id,
                    "team_id": row.team_id,
                    "shirt_number": row.shirt_number,
                    "possession_frame": row.frame_number,
                    "shot_frame": row.frame_number,
                    "dist_to_goal_line": row.dist_to_goal_line,
                    "speed_kmh": row.ball_speed,
                    "is_goal": False,
                    "event_type": "shot",
                    "ml_score": row.shot_prob,
                }
                for row in rows
            ]
        else:
            threshold_m = self.config.near_goal_cm_threshold / 100.0
            filtered = shot_features.filter(
                (col("inside_shot_area") == False)
                & (col("dist_to_goal_line") > threshold_m)
            )
            if goal_frames:
                filtered = filtered.filter(~col("frame_number").isin(goal_frames))

            best = self._cluster_shot_events(
                filtered,
                [col("dist_to_goal_line").desc()],
            )
            rows = best.collect()
            return [
                {
                    "player_id": row.player_id,
                    "track_id": row.track_id,
                    "team_id": row.team_id,
                    "shirt_number": row.shirt_number,
                    "possession_frame": row.frame_number,
                    "shot_frame": row.frame_number,
                    "dist_to_goal_line": row.dist_to_goal_line,
                    "speed_kmh": row.ball_speed,
                    "is_goal": False,
                    "event_type": "shot",
                }
                for row in rows
            ]

    def _cluster_shot_events(self, df: DataFrame, order_by_cols: list) -> DataFrame:
        """
        Groups a player's qualifying shot frames into discrete attempts based on
        frame-gap, instead of collapsing to a single best row per player for the
        whole match. Keeps one representative row per (player, event) using
        order_by_cols to pick the best frame within each burst.
        """
        w_seq = Window.partitionBy("player_id").orderBy("frame_number")
        prev_frame = lag("frame_number", 1).over(w_seq)
        new_event = when(
            prev_frame.isNull()
            | ((col("frame_number") - prev_frame) > self.config.shot_event_gap_frames),
            1,
        ).otherwise(0)
        df = df.withColumn("shot_event_id", spark_sum(new_event).over(w_seq))

        w_rep = Window.partitionBy("player_id", "shot_event_id").orderBy(*order_by_cols)
        return (
            df.withColumn("rn", row_number().over(w_rep))
            .filter(col("rn") == 1)
            .drop("rn", "shot_event_id")
        )
