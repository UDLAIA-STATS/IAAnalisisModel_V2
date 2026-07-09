from typing import Dict, Optional, Tuple
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    avg,
    col,
    expr,
    lit,
    when,
    row_number,
    count,
    sum as spark_sum,
    max as spark_max,
    min as spark_min,
    lag,
    sqrt,
    pow,
    acos,
    degrees,
    broadcast,
)
from pyspark.sql.types import FloatType, IntegerType, StructType, StructField
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.functions import vector_to_array

from sqlmodel import Session, select, col as modelcol

from src.entities.predictors.ball_possession_analyzer_base import (
    BallPossessionAnalyzerBase,
)
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.goal_model import GoalModel
from src.core.repository.player_repository import PlayerRepository


class BallPossessionMLAnalyzer(BallPossessionAnalyzerBase):
    """
    On-the-fly ML analyzer using PySpark ML to infer possession and shots
    from the current match data. Models are trained per match.
    """

    SHOT_DISTANCE_THRESHOLD: float = 25.0
    SHOT_ANGLE_THRESHOLD: float = 45.0
    GOAL_ASSOCIATION_WINDOW: float = 1.0
    MIN_POSSESSION_SECONDS: float = 0.1
    POSSESSION_PROB_THRESHOLD: float = 0.5
    SHOT_PROB_THRESHOLD: float = 0.4

    def analyze(self, match_id: int, session: Session) -> None:
        logfire.info(
            f"[BallPossessionMLAnalyzer] Starting per‑match ML training for {match_id}"
        )

        player_states_df = self._load_player_states(match_id, session)
        ball_states_df = self._load_ball_states(match_id, session)
        goals_df = self._load_goals(match_id, session)

        if player_states_df.isEmpty():
            logfire.warning("No player states found.")
            return

        # Build feature matrix (joins player and ball)
        feature_df = self._build_feature_matrix(player_states_df, ball_states_df)
        logfire.info(f"[BallPossessionMLAnalyzer] timestamp column count: {feature_df.columns.count('timestamp')}")

        # Compute goal centroid (if any goals)
        goal_centroid = (
            self._compute_goal_centroid(goals_df) if not goals_df.isEmpty() else None
        )
        if goal_centroid:
            feature_df = self._add_goal_features(feature_df, goal_centroid)

        # ------------------------------------------------------------
        # 1. Train a possession classifier (RandomForest)
        #    using explicit `has_ball` as label.
        # ------------------------------------------------------------
        possession_model = self._train_possession_model(feature_df)
        if possession_model:
            feature_df = self._predict_possession(feature_df, possession_model)
        else:
            # Fallback: use explicit flag
            feature_df = feature_df.withColumn("possession_pred", col("has_ball"))

        # ------------------------------------------------------------
        # 2. Train a shot classifier if goals exist
        #    Use frames near goals as positive, random others as negative.
        # ------------------------------------------------------------
        if goal_centroid and not goals_df.isEmpty():
            shot_model = self._train_shot_model(feature_df, goals_df)
            if shot_model:
                feature_df = self._predict_shots(feature_df, shot_model)
            else:
                # Fallback: rule‑based shot detection
                feature_df = self._rule_based_shots(feature_df)
        else:
            feature_df = feature_df.withColumn("shot_pred", lit(0))

        # 3. Compute possession time from predicted possession
        possession_times = self._compute_possession_time(feature_df)

        # 4. Count shots from predicted shots
        shot_counts = self._aggregate_shots(feature_df)

        # 5. Assign goals to players using predicted possession
        goal_assignments = self._assign_goals_to_players(feature_df, goals_df)

        # 6. Update database
        self._update_players(
            match_id, possession_times, shot_counts, goal_assignments, session
        )

        logfire.info(
            f"[BallPossessionMLAnalyzer] Analysis complete for match {match_id}"
        )

    # --------------------------------------------------------------------
    # ML Model Training
    # --------------------------------------------------------------------

    def _train_possession_model(self, df: DataFrame):
        """
        Train a RandomForestClassifier to predict has_ball.
        Use only rows where has_ball is not null.
        """
        # Filter rows with explicit label
        train_df = df.filter(col("has_ball").isNotNull())
        if train_df.count() < 10:
            logfire.warning(
                "Too few labelled frames for possession training; skipping ML."
            )
            return None
        
        distinct_labels = train_df.select("has_ball").distinct().count()
        if distinct_labels < 2:
            logfire.warning("[BallPossessionMLAnalyzer] Only one class present for possession training; skipping ML.")
            return None

        # Feature columns (must match those present in df)
        feature_cols = ["dist_pb", "player_speed", "ball_speed"]
        # If goal features exist, we could include them, but they are not needed for possession.
        # For possession, distance to goal may help, so add if present.
        if "dist_ball_goal" in df.columns:
            feature_cols.append("dist_ball_goal")
        if "angle_ball_goal" in df.columns:
            feature_cols.append("angle_ball_goal")

        # Assemble
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="has_ball",
            numTrees=50,
            maxDepth=10,
            seed=42,
        )

        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(train_df)
        logfire.info("Possession classifier trained successfully.")
        return model

    def _predict_possession(self, df: DataFrame, model: PipelineModel) -> DataFrame:
        """Apply possession model to all frames, add probability and prediction."""
        result = model.transform(df)
        prob_array = vector_to_array(col("probability"))
        result = result.withColumn(
            "possession_prob",
            prob_array.getItem(1)
        )
        result = result.withColumn(
            "possession_pred",
            when(col("possession_prob") >= self.POSSESSION_PROB_THRESHOLD, 1).otherwise(
                0
            ),
        )
        return result

    def _train_shot_model(self, df: DataFrame, goals_df: DataFrame):
        """
        Create training data for shots:
        - Positive: frames within GOAL_ASSOCIATION_WINDOW before a goal.
        - Negative: random frames far from goal.
        """
        df_with_ts = df.withColumnRenamed("timestamp", "player_ts")
        goals = goals_df.withColumnRenamed("timestamp", "goal_ts")

        # Usar crossJoin + filter en lugar de join con expr para evitar ambigüedad
        pos_df = df_with_ts.crossJoin(broadcast(goals)) \
            .where(
                expr("goal_ts - player_ts >= 0 AND goal_ts - player_ts <= {}".format(self.GOAL_ASSOCIATION_WINDOW))
            ) \
            .withColumn("time_diff", expr("goal_ts - player_ts")) \
            .withColumn("label", lit(1)) \
            .select("label", "dist_pb", "player_speed", "ball_speed",
                    "dist_ball_goal", "angle_ball_goal")
        # ... resto del código (negativos y entrenamiento) igual

        # Negativos: frames lejos del gol (usamos df original)
        neg_df = (
            df.filter(
                (col("dist_ball_goal") > 2 * self.SHOT_DISTANCE_THRESHOLD)
                | (col("angle_ball_goal") > 60)
            )
            .withColumn("label", lit(0))
            .select(
                "label",
                "dist_pb",
                "player_speed",
                "ball_speed",
                "dist_ball_goal",
                "angle_ball_goal",
            )
        )

        pos_count = pos_df.count()
        if pos_count < 5:
            logfire.warning("Too few positive shot examples; skipping shot model.")
            return None

        neg_count = neg_df.count()
        sample_frac = min(1.0, pos_count / neg_count) if neg_count > 0 else 1.0
        neg_sampled = neg_df.sample(
            withReplacement=False, fraction=sample_frac, seed=42
        )

        train_df = pos_df.union(neg_sampled)
        train_df.cache()

        feature_cols = [
            "dist_pb",
            "player_speed",
            "ball_speed",
            "dist_ball_goal",
            "angle_ball_goal",
        ]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol="label",
            numTrees=50,
            maxDepth=10,
            seed=42,
        )
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(train_df)
        logfire.info("Shot classifier trained successfully.")
        return model

    def _predict_shots(self, df: DataFrame, model: PipelineModel) -> DataFrame:
        df = df.drop("features", "scaled_features", "rawPrediction", "probability", "prediction")
        result = model.transform(df)
        prob_array = vector_to_array(col("probability"))
        result = result.withColumn(
            "shot_prob",
            prob_array.getItem(1)
        )
        result = result.withColumn(
            "shot_pred",
            when(col("shot_prob") >= self.SHOT_PROB_THRESHOLD, 1).otherwise(0)
        )
        return result

    def _rule_based_shots(self, df: DataFrame) -> DataFrame:
        return df.withColumn(
            "shot_pred",
            when(
                (col("has_ball") == 1)
                & (col("dist_ball_goal") <= self.SHOT_DISTANCE_THRESHOLD)
                & (col("angle_ball_goal") <= self.SHOT_ANGLE_THRESHOLD),
                1,
            ).otherwise(0),
        )

    # --------------------------------------------------------------------
    # Possession time and shot aggregation (using predicted labels)
    # --------------------------------------------------------------------

    def _compute_possession_time(self, df: DataFrame) -> DataFrame:
        # Use possession_pred as the label
        w = Window.partitionBy("player_id").orderBy("timestamp")
        changed = when(
            lag("possession_pred", 1).over(w).isNull()
            | (col("possession_pred") != lag("possession_pred", 1).over(w)),
            lit(1),
        ).otherwise(lit(0))
        df_grouped = df.withColumn("possession_group", spark_sum(changed).over(w))

        possession_groups = (
            df_grouped.filter(col("possession_pred") == 1)
            .groupBy("player_id", "possession_group")
            .agg((spark_max("timestamp") - spark_min("timestamp")).alias("duration"))
        )
        possession_groups = possession_groups.filter(
            col("duration") >= self.MIN_POSSESSION_SECONDS
        )
        result = possession_groups.groupBy("player_id").agg(
            spark_sum("duration").alias("possession_time")
        )
        return result

    def _aggregate_shots(self, df: DataFrame) -> DataFrame:
        return (
            df.filter(col("shot_pred") == 1)
            .groupBy("player_id")
            .agg(count("*").alias("shots"))
        )

    def _assign_goals_to_players(self, df: DataFrame, goals_df: DataFrame) -> Dict[int, int]:
        if goals_df.isEmpty():
            return {}

        ball_players = df.filter(col("possession_pred") == 1) \
            .select("player_id", "timestamp", "frame_number") \
            .withColumnRenamed("timestamp", "player_ts")

        goals = goals_df.withColumnRenamed("timestamp", "goal_ts") \
            .withColumnRenamed("frame_number", "goal_frame")

        joined = ball_players.crossJoin(broadcast(goals)) \
            .where(
                expr("goal_ts - player_ts >= 0 AND goal_ts - player_ts <= {}".format(self.GOAL_ASSOCIATION_WINDOW))
            ) \
            .withColumn("time_diff", expr("goal_ts - player_ts"))

        w = Window.partitionBy("id").orderBy(col("time_diff").asc())
        ranked = joined.withColumn("rn", row_number().over(w))
        best = ranked.filter(col("rn") == 1)

        goal_counts = best.groupBy("player_id").agg(count("*").alias("goal_count")).collect()
        return {row.player_id: row.goal_count for row in goal_counts}

    def _load_player_states(self, match_id: int, session: Session) -> DataFrame:
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(PlayerModel.match_id == match_id)
            .order_by(modelcol(PlayerState.timestamp))
        )
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._player_state_schema())
        data = [
            (
                pm.id,
                pm.track_id,
                ps.frame_number,
                ps.timestamp,
                1 if ps.has_ball else 0,
                ps.ball_x,
                ps.ball_y,
                ps.x1,
                ps.y1,
                ps.x2,
                ps.y2,
                ps.dx_meters,
                ps.dy_meters,
                ps.vx,
                ps.vy,
            )
            for ps, pm in rows
        ]
        return self.spark.createDataFrame(data, self._player_state_schema())

    def _player_state_schema(self) -> StructType:
        return StructType(
            [
                StructField("player_id", IntegerType(), True),
                StructField("track_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("has_ball", IntegerType(), True),
                StructField("ball_x", FloatType(), True),
                StructField("ball_y", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("dx_meters", FloatType(), True),
                StructField("dy_meters", FloatType(), True),
                StructField("vx", FloatType(), True),
                StructField("vy", FloatType(), True),
            ]
        )

    def _load_ball_states(self, match_id: int, session: Session) -> DataFrame:
        stmt = (
            select(BallState)
            .where(BallState.match_id == match_id)
            .order_by(modelcol(BallState.timestamp))
        )
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._ball_state_schema())
        data = [
            (
                bs.frame_number,
                bs.timestamp,
                bs.x1,
                bs.y1,
                bs.x2,
                bs.y2,
                bs.dx_meters,
                bs.dy_meters,
                bs.confidence,
            )
            for bs in rows
        ]
        return self.spark.createDataFrame(data, self._ball_state_schema())

    def _ball_state_schema(self) -> StructType:
        return StructType(
            [
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("dx_meters", FloatType(), True),
                StructField("dy_meters", FloatType(), True),
                StructField("confidence", FloatType(), True),
            ]
        )

    def _load_goals(self, match_id: int, session: Session) -> DataFrame:
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._goal_schema())
        data = [
            (g.id, g.frame_number, g.timestamp, g.x1, g.y1, g.x2, g.y2, g.confidence)
            for g in rows
        ]
        return self.spark.createDataFrame(data, self._goal_schema())

    def _goal_schema(self) -> StructType:
        return StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("confidence", FloatType(), True),
            ]
        )

    def _build_feature_matrix(
        self, player_df: DataFrame, ball_df: DataFrame
    ) -> DataFrame:
        player_df = player_df.withColumn("px", (col("x1") + col("x2")) / 2).withColumn(
            "py", (col("y1") + col("y2")) / 2
        )
        ball_df = (
            ball_df.withColumn("bx", (col("x1") + col("x2")) / 2)
            .withColumn("by", (col("y1") + col("y2")) / 2)
            .withColumnRenamed("timestamp", "ball_timestamp")
            .withColumnRenamed("dx_meters", "dx_meters_ball")
            .withColumnRenamed("dy_meters", "dy_meters_ball")
            .withColumnRenamed("x1", "ball_x1")
            .withColumnRenamed("y1", "ball_y1")
            .withColumnRenamed("x2", "ball_x2")
            .withColumnRenamed("y2", "ball_y2")
            .withColumnRenamed("confidence", "ball_confidence")
        )

        joined = player_df.join(ball_df, on="frame_number", how="left").fillna(
            {"bx": 0.0, "by": 0.0, "dx_meters_ball": 0.0, "dy_meters_ball": 0.0}
        )

        joined = joined.withColumn(
            "dist_pb",
            sqrt(pow(col("px") - col("bx"), 2) + pow(col("py") - col("by"), 2)),
        )
        joined = joined.withColumn(
            "player_speed", sqrt(pow(col("vx"), 2) + pow(col("vy"), 2))
        )
        joined = joined.withColumn(
            "ball_speed",
            sqrt(pow(col("dx_meters_ball"), 2) + pow(col("dy_meters_ball"), 2)),
        )
        return joined

    def _add_goal_features(
        self, df: DataFrame, centroid: Tuple[float, float]
    ) -> DataFrame:
        gx, gy = centroid
        df = df.withColumn(
            "dist_ball_goal",
            sqrt(pow(col("bx") - lit(gx), 2) + pow(col("by") - lit(gy), 2)),
        )
        df = df.withColumn(
            "angle_ball_goal",
            degrees(
                acos(
                    when(
                        (col("ball_speed") > 0) & (col("dist_ball_goal") > 0),
                        (
                            col("dx_meters_ball") * (lit(gx) - col("bx"))
                            + col("dy_meters_ball") * (lit(gy) - col("by"))
                        )
                        / (col("ball_speed") * col("dist_ball_goal")),
                    ).otherwise(0.0)
                )
            ),
        )
        return df

    def _compute_goal_centroid(
        self, goals_df: DataFrame
    ) -> Optional[Tuple[float, float]]:
        centroids = goals_df.withColumn("cx", (col("x1") + col("x2")) / 2).withColumn(
            "cy", (col("y1") + col("y2")) / 2
        )
        avg_row = centroids.select(
            avg("cx").alias("cx"), avg("cy").alias("cy")
        ).collect()[0]
        if avg_row.cx is not None and avg_row.cy is not None:
            return (float(avg_row.cx), float(avg_row.cy))
        return None

    def _update_players(
        self,
        match_id: int,
        possession_df: DataFrame,
        shot_df: DataFrame,
        goal_assignments: Dict[int, int],
        session: Session,
    ) -> None:
        possession_dict = {
            row.player_id: row.possession_time for row in possession_df.collect()
        }
        shot_dict = {row.player_id: row.shots for row in shot_df.collect()}
        players = PlayerRepository.get_players_by_match_id(match_id, session)
        if not players:
            return
        for player in players:
            if player.id in possession_dict:
                player.ball_possession_time = float(possession_dict[player.id])
            # if player.id in shot_dict:
            #     player.goals += int(shot_dict[player.id])
            if player.id in goal_assignments or player.id in shot_dict:
                if player.goals == 0 or player.goals != int(goal_assignments[player.id])  or player.goals != int(shot_dict[player.id]):
                    player.goals = int(goal_assignments[player.id])
                    player.goals += int(shot_dict[player.id])
                elif int(goal_assignments[player.id]) > 0:
                    player.goals = int(float(int(goal_assignments[player.id]) * 0.2) + player.goals)
                    player.goals = int(float(int(shot_dict[player.id]) * 0.8) + player.goals)

        session.commit()
        logfire.info("Player stats updated.")


ball_possession_analyzer = BallPossessionMLAnalyzer()
