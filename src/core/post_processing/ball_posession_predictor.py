from typing import Optional
import numpy as np
import logfire
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    col,
    lit,
    when,
    row_number,
    sqrt,
    pow,
    broadcast,
    coalesce,
    abs as sql_abs,
    collect_list,
    udf,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    BooleanType,
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline, PipelineModel
from sqlmodel import Session, select, col as modelcol
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.utils import spark
from pyspark.ml.functions import vector_to_array


class BallPossessionPredictor:
    DISTANCE_THRESHOLD_METERS = 2.0
    MIN_CONFIDENCE = 0.5
    METER_PER_PIXEL = 0.01
    ML_MIN_SAMPLES = 30
    TRAIN_SAMPLE_FRACTION = 0.2
    SMOOTHING_WINDOW = 3

    def __init__(self):
        self.spark = spark

    def predict(self, match_id: int, session: Session) -> None:
        self.match_id = match_id
        self.session = session
        logfire.info(f"[BallPossessionPredictor] Starting for match {self.match_id}")

        ball_df = self._fetch_ball_dataframe()
        player_df = self._fetch_player_dataframe()

        if ball_df.count() == 0 or player_df.count() == 0:
            logfire.warning("[BallPossessionPredictor] No data available.")
            return

        combined = self._build_combined_dataframe(ball_df, player_df)
        labeled = self._label_training_data(combined)
        pos_count = labeled.filter(col("label") == 1).count()

        model = None
        if pos_count >= self.ML_MIN_SAMPLES:
            train_sample = labeled.sample(fraction=self.TRAIN_SAMPLE_FRACTION, seed=42)
            model = self._train_model(train_sample)
            logfire.info(
                "[BallPossessionPredictor] ML model trained on 20% of labeled data."
            )
        else:
            logfire.warning(
                "[BallPossessionPredictor] Insufficient positive samples; using heuristic."
            )

        raw_predictions = self._predict_possession(combined, model)
        smoothed = self._smooth_predictions(raw_predictions, combined)
        self._update_database(smoothed)

        logfire.info(f"[BallPossessionPredictor] Finished for match {self.match_id}")

    def _fetch_ball_dataframe(self) -> DataFrame:
        stmt = (
            select(BallState)
            .where(BallState.match_id == self.match_id)
            .order_by(modelcol(BallState.frame_number))
        )
        balls = list(self.session.exec(stmt).all())
        if not balls:
            return self.spark.createDataFrame([], schema=self._ball_schema())

        rows = []
        for b in balls:
            cx, cy = self._compute_centroid(b.x1, b.y1, b.x2, b.y2)
            rows.append(
                {
                    "ball_state_id": b.id,
                    "frame_number": b.frame_number,
                    "ball_timestamp": b.timestamp,
                    "ball_cx": cx,
                    "ball_cy": cy,
                    "ball_mx": cx * self.METER_PER_PIXEL,
                    "ball_my": cy * self.METER_PER_PIXEL,
                    "ball_confidence": b.confidence,
                    "ball_vx": b.vx or 0.0,
                    "ball_vy": b.vy or 0.0,
                    "ball_speed_kmh": b.speed_kmh or 0.0,
                }
            )
        return self.spark.createDataFrame(rows, schema=self._ball_schema())

    @staticmethod
    def _ball_schema() -> StructType:
        return StructType(
            [
                StructField("ball_state_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("ball_timestamp", FloatType(), True),
                StructField("ball_cx", FloatType(), True),
                StructField("ball_cy", FloatType(), True),
                StructField("ball_mx", FloatType(), True),
                StructField("ball_my", FloatType(), True),
                StructField("ball_confidence", FloatType(), True),
                StructField("ball_vx", FloatType(), True),
                StructField("ball_vy", FloatType(), True),
                StructField("ball_speed_kmh", FloatType(), True),
            ]
        )

    def _fetch_player_dataframe(self) -> DataFrame:
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(modelcol(PlayerModel.match_id) == self.match_id)
            .order_by(modelcol(PlayerState.frame_number))
        )
        results = list(self.session.exec(stmt).all())
        if not results:
            return self.spark.createDataFrame([], schema=self._player_schema())

        rows = []
        for state, player in results:
            cx, cy = self._compute_centroid(state.x1, state.y1, state.x2, state.y2)
            area = max(0.0, (state.x2 or 0) - (state.x1 or 0)) * max(
                0.0, (state.y2 or 0) - (state.y1 or 0)
            )
            rows.append(
                {
                    "player_state_id": state.id,
                    "player_id": player.id,
                    "track_id": player.track_id,
                    "team_id": player.team_id,
                    "frame_number": state.frame_number,
                    "player_timestamp": state.timestamp,
                    "player_cx": cx,
                    "player_cy": cy,
                    "player_mx": cx * self.METER_PER_PIXEL,
                    "player_my": cy * self.METER_PER_PIXEL,
                    "player_area": area,
                    "player_confidence": state.confidence,
                    "has_ball": state.has_ball,
                    "player_vx": state.vx or 0.0,
                    "player_vy": state.vy or 0.0,
                    "player_speed_kmh": state.speed_kmh or 0.0,
                }
            )
        return self.spark.createDataFrame(rows, schema=self._player_schema())

    @staticmethod
    def _player_schema() -> StructType:
        return StructType(
            [
                StructField("player_state_id", IntegerType(), True),
                StructField("player_id", IntegerType(), True),
                StructField("track_id", IntegerType(), True),
                StructField("team_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("player_timestamp", FloatType(), True),
                StructField("player_cx", FloatType(), True),
                StructField("player_cy", FloatType(), True),
                StructField("player_mx", FloatType(), True),
                StructField("player_my", FloatType(), True),
                StructField("player_area", FloatType(), True),
                StructField("player_confidence", FloatType(), True),
                StructField("has_ball", BooleanType(), True),
                StructField("player_vx", FloatType(), True),
                StructField("player_vy", FloatType(), True),
                StructField("player_speed_kmh", FloatType(), True),
            ]
        )

    @staticmethod
    def _compute_centroid(x1, y1, x2, y2):
        if None in (x1, y1, x2, y2):
            return (0.0, 0.0)
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _build_combined_dataframe(
        self, ball_df: DataFrame, player_df: DataFrame
    ) -> DataFrame:
        combined = player_df.join(broadcast(ball_df), on="frame_number", how="inner")

        combined = (
            combined.withColumn(
                "dist_px",
                sqrt(
                    pow(col("player_cx") - col("ball_cx"), 2)
                    + pow(col("player_cy") - col("ball_cy"), 2)
                ),
            )
            .withColumn(
                "dist_m",
                sqrt(
                    pow(col("player_mx") - col("ball_mx"), 2)
                    + pow(col("player_my") - col("ball_my"), 2)
                ),
            )
            .withColumn("to_ball_x", col("ball_mx") - col("player_mx"))
            .withColumn("to_ball_y", col("ball_my") - col("player_my"))
            .withColumn(
                "player_speed_norm",
                sqrt(pow(col("player_vx"), 2) + pow(col("player_vy"), 2)),
            )
            .withColumn(
                "ball_speed_norm", sqrt(pow(col("ball_vx"), 2) + pow(col("ball_vy"), 2))
            )
            .withColumn(
                "dot_to_ball",
                col("player_vx") * col("to_ball_x")
                + col("player_vy") * col("to_ball_y"),
            )
            .withColumn(
                "angle_cos",
                when(
                    (col("player_speed_norm") > 0) & (col("dist_m") > 0),
                    col("dot_to_ball") / (col("player_speed_norm") * col("dist_m")),
                ).otherwise(0.0),
            )
            .withColumn("area_norm", col("player_area") / 10000.0)
            .withColumn("player_conf", col("player_confidence"))
            .withColumn("ball_conf", col("ball_confidence"))
        )

        feature_cols = [
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "player_conf",
            "ball_conf",
            "area_norm",
        ]
        for c in feature_cols:
            combined = combined.withColumn(c, coalesce(col(c), lit(0.0)))

        return combined.select(
            "frame_number",
            "player_timestamp",
            "ball_timestamp",
            "player_state_id",
            "player_id",
            "track_id",
            "team_id",
            "ball_state_id",
            "player_conf",
            "ball_conf",
            "player_mx",
            "player_my",
            "ball_mx",
            "ball_my",
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "area_norm",
        )

    def _label_training_data(self, combined: DataFrame) -> DataFrame:
        candidates = combined.filter(
            (col("dist_m") <= self.DISTANCE_THRESHOLD_METERS)
            & (col("ball_conf") >= self.MIN_CONFIDENCE)
        )
        w = Window.partitionBy("frame_number").orderBy(col("dist_m").asc())
        labeled = candidates.withColumn("rn", row_number().over(w))
        labeled = labeled.withColumn("label", when(col("rn") == 1, 1).otherwise(0))
        feature_cols = [
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "player_conf",
            "ball_conf",
            "area_norm",
        ]
        return labeled.select("frame_number", "player_state_id", *feature_cols, "label")

    def _train_model(self, labeled: DataFrame) -> PipelineModel:
        pos = labeled.filter(col("label") == 1)
        neg = labeled.filter(col("label") == 0)
        pos_count = pos.count()
        neg_count = neg.count()
        if neg_count > pos_count:
            neg_sampled = neg.sample(
                withReplacement=False, fraction=pos_count / neg_count, seed=42
            )
            train = pos.union(neg_sampled)
        else:
            train = labeled

        feature_cols = [
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "player_conf",
            "ball_conf",
            "area_norm",
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
            maxDepth=8,
            seed=42,
        )
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        return pipeline.fit(train)

    def _predict_possession(
        self, combined: DataFrame, model: Optional[PipelineModel]
    ) -> DataFrame:
        if model is not None:
            feature_cols = [
                "dist_m",
                "angle_cos",
                "player_speed_norm",
                "ball_speed_norm",
                "player_conf",
                "ball_conf",
                "area_norm",
            ]
            for c in feature_cols:
                if c not in combined.columns:
                    combined = combined.withColumn(c, lit(0.0))
            predictions = model.transform(combined)
            prob_array = vector_to_array(col("probability"))
            predictions = predictions.withColumn(
                "prob_positive", prob_array.getItem(1)
            )
        else:
            max_dist = (
                combined.select(sql_abs("dist_m"))
                .agg({"dist_m": "max"})
                .collect()[0][0]
                or 1.0
            )
            predictions = combined.withColumn(
                "prob_positive",
                when(col("dist_m") > 0, 1.0 - (col("dist_m") / max_dist)).otherwise(
                    1.0
                ),
            )

        return predictions.select("frame_number", "player_state_id", "prob_positive")

    def _smooth_predictions(self, raw: DataFrame, combined: DataFrame) -> DataFrame:
        w_max = Window.partitionBy("frame_number").orderBy(col("prob_positive").desc())
        assigned = (
            raw.withColumn("rn", row_number().over(w_max))
            .filter(col("rn") == 1)
            .select("frame_number", col("player_state_id").alias("pred_player"))
        )

        all_frames = combined.select("frame_number").distinct()
        assigned_full = all_frames.join(assigned, on="frame_number", how="left")

        window_spec = Window.orderBy("frame_number").rowsBetween(
            -self.SMOOTHING_WINDOW // 2, self.SMOOTHING_WINDOW // 2
        )
        smoothed = assigned_full.withColumn(
            "neighbor_players", collect_list("pred_player").over(window_spec)
        )

        def mode(arr):
            if not arr:
                return None
            # Filtrar None
            arr_filtered = [x for x in arr if x is not None]
            if not arr_filtered:
                return None
            values, counts = np.unique(arr_filtered, return_counts=True)
            return int(values[np.argmax(counts)])

        mode_udf = udf(mode, IntegerType())
        smoothed = smoothed.withColumn("smoothed_player", mode_udf("neighbor_players"))

        result = (
            combined.select("frame_number", "player_state_id")
            .join(
                smoothed.select("frame_number", "smoothed_player"),
                on="frame_number",
                how="inner",
            )
            .withColumn(
                "has_ball_pred",
                when(col("player_state_id") == col("smoothed_player"), True).otherwise(
                    False
                ),
            )
            .select("frame_number", "player_state_id", "has_ball_pred")
        )

        return result

    def _update_database(self, predictions: DataFrame) -> None:
        rows = predictions.select("player_state_id", "has_ball_pred").collect()
        update_map = {row.player_state_id: row.has_ball_pred for row in rows}

        stmt = (
            select(PlayerState)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(PlayerModel.match_id == self.match_id)
        )
        all_states = self.session.exec(stmt).all()

        to_update = []
        for state in all_states:
            new_val = update_map.get(state.id)
            if new_val is not None and state.has_ball != new_val:
                state.has_ball = new_val
                to_update.append(state)

        if to_update:
            self.session.bulk_update_mappings(
                PlayerState, [{"id": s.id, "has_ball": s.has_ball} for s in to_update]
            )
            self.session.commit()
            logfire.info(
                f"[BallPossessionPredictor] Updated {len(to_update)} PlayerState records."
            )
        else:
            logfire.info("[BallPossessionPredictor] No changes needed.")


ball_possession_predictor_cls = BallPossessionPredictor()
