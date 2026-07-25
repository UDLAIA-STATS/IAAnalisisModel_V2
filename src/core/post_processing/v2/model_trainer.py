from pyspark.sql import DataFrame
from pyspark.ml import PipelineModel, Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import Window
from pyspark.sql.functions import col, when, lit, row_number

from src.core.post_processing.v2.processing_config import PostProcessingConfig
import logfire
import pathlib
from typing import Optional, Tuple

from src.entities.utils.spark_instance import spark


class ModelTrainer:
    def __init__(self, config: PostProcessingConfig):
        self.config = config
        self.spark = spark

    def _save_model(self, model: PipelineModel, path: str) -> bool:
        try:
            model.write().overwrite().save(path)
            logfire.info(f"[ModelTrainer] Model saved to {path}")
            return True
        except Exception as e:
            logfire.error(f"[ModelTrainer] Failed to save model to {path}: {e}")
            return False

    def _load_model(self, path: str) -> Optional[PipelineModel]:
        try:
            model = PipelineModel.load(path)
            logfire.info(f"[ModelTrainer] Model loaded from {path}")
            return model
        except Exception as e:
            logfire.warning(f"[ModelTrainer] Could not load model from {path}: {e}")
            return None

    def _train_classifier(
        self,
        df: DataFrame,
        label_col: str,
        feature_cols: list,
        model_path: Optional[str] = None,
        force_retrain: bool = False,
        sample_fraction: float = 0.8,
    ) -> Optional[PipelineModel]:
        if df.isEmpty():
            return None

        if sample_fraction < 1.0:
            df = df.sample(fraction=sample_fraction, seed=42)

        if model_path and not force_retrain and pathlib.Path(model_path).exists():
            model = self._load_model(model_path)
            # historical_df = spark.read.parquet(MODELS_TRAINING_DIR.as_posix())
            if model:
                return model
        

        distinc_labels = df.select(label_col).distinct().count()
        if distinc_labels < 2:
            if model_path:
                other_models = list(pathlib.Path(model_path).parent.glob(f"{model_path.split('/')[-1].split("_")[0]}*"))
                if len(other_models) > 1:
                    last_model = other_models[-1]
                    logfire.warning(
                        f"[ModelTrainer] Multiple models found for {model_path}; using {last_model}"
                    )
                    model = self._load_model(last_model.as_posix())
                    if model:
                        return model

            logfire.warning(
                f"[ModelTrainer] Only one class present for {label_col}; skipping ML."
            )
            return None


        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(
            inputCol="features",
            outputCol="scaled_features",
            withStd=True,
            withMean=True,
        )
        rf = RandomForestClassifier(
            featuresCol="scaled_features",
            labelCol=label_col,
            numTrees=50,
            maxDepth=10,
            seed=42,
        )
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(df)

        if model_path:
            self._save_model(model, model_path)

        return model

    def train_possession_model(
        self,
        features_df: DataFrame,
        model_path: str,
        force_retrain: bool = False,
    ) -> Optional[PipelineModel]:
        # Filtrar solo filas etiquetadas (has_ball no nulo)
        train_df = features_df.withColumn("has_ball", col("has_ball").cast("double"))
        if train_df.count() < 10:
            logfire.warning(
                "[ModelTrainer] Insufficient possession labels; skipping ML."
            )
            return None
        feature_cols = [
            "dist_m",
            "angle_cos",
            "player_speed_norm",
            "ball_speed_norm",
            "player_conf",
            "ball_conf",
            "area_norm",
        ]
        return self._train_classifier(
            train_df,
            "has_ball",
            feature_cols,
            model_path,
            force_retrain,
            self.config.train_sample_fraction,
        )

    def train_goal_linker_model(
        self,
        candidates_df: DataFrame,
        model_path: str,
        force_retrain: bool = False,
    ) -> Optional[PipelineModel]:
        # Necesitamos etiquetas: para cada goal_id, el mejor candidato (score más bajo) es positivo (label=1)
        w = Window.partitionBy("goal_id").orderBy(col("score").asc())
        labeled = candidates_df.withColumn("rn", row_number().over(w)).withColumn(
            "label", when(col("rn") == 1, 1).otherwise(0)
        )
        train_df = (
            labeled.filter(col("label").isNotNull())
            .select(
                "label",
                "frame_gap",
                "ball_to_goal_dist_px",
                "dist_to_goal_line_m",
                "player_speed",
                "ball_speed",
                "confidence",
            )
            .na.drop()
        )
        if train_df.count() < 10:
            logfire.warning(
                "[ModelTrainer] Insufficient goal linker labels; skipping ML."
            )
            return None
        feature_cols = [
            "frame_gap",
            "ball_to_goal_dist_px",
            "dist_to_goal_line_m",
            "player_speed",
            "ball_speed",
            "confidence",
        ]
        return self._train_classifier(
            train_df,
            "label",
            feature_cols,
            model_path,
            force_retrain,
            self.config.train_sample_fraction,
        )

    def train_shot_model(
        self,
        features_df: DataFrame,
        goals_df: DataFrame,
        model_path: str,
        force_retrain: bool = False,
    ) -> Optional[PipelineModel]:
        if goals_df.isEmpty():
            return None
        threshold_m = self.config.near_goal_cm_threshold / 100.0
        goal_timestamps = [
            row.timestamp for row in goals_df.select("timestamp").collect()
        ]
        goal_frame_numbers = [
            row.frame_number for row in goals_df.select("frame_number").collect()
        ]
        neg = features_df.filter(
            (col("ball_timestamp").isin(goal_timestamps))
            | (col("frame_number").isin(goal_frame_numbers))
            | (col("dist_to_goal_line") <= threshold_m)
            | (col("inside_shot_area") == True)
        ).withColumn("label", lit(0))
        pos = features_df.filter(
            (col("dist_to_goal_line") > 2 * threshold_m)
            & (col("inside_shot_area") == False)
        ).withColumn("label", lit(1))
        pos_count = pos.count()
        if pos_count < 5:
            logfire.warning(
                "[ModelTrainer] Too few positive shot examples; skipping ML."
            )
            return None
        neg_count = neg.count()
        if neg_count > pos_count:
            neg_sampled = neg.sample(
                withReplacement=False, fraction=pos_count / neg_count, seed=42
            )
            train_df = pos.union(neg_sampled)
        else:
            train_df = pos.union(neg)
        feature_cols = [
            "dist_to_goal_line",
            "ball_speed",
            "player_speed",
            "angle_to_goal",
            "confidence",
        ]
        return self._train_classifier(
            train_df,
            "label",
            feature_cols,
            model_path,
            force_retrain,
            self.config.train_sample_fraction,
        )

    def train_trajectory_model(
        self,
        ball_df: DataFrame,
        model_path_cx: Optional[str] = None,
        model_path_cy: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Tuple[Optional[PipelineModel], Optional[PipelineModel]]:
        if ball_df.count() < 10:
            return None, None
        # Features: frame_number, frame_number^2
        train_df = ball_df.select("frame_number", "cx", "cy").withColumn(
            "frame2", col("frame_number") ** 2
        )
        feature_cols = ["frame_number", "frame2"]

        def _train_regressor(
            label_col: str, model_path: Optional[str]
        ) -> Optional[PipelineModel]:
            if model_path and not force_retrain and pathlib.Path(model_path).exists():
                model = self._load_model(model_path)
                if model:
                    return model
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            rf = RandomForestRegressor(
                featuresCol="features",
                labelCol=label_col,
                numTrees=50,
                maxDepth=6,
                seed=42,
            )
            pipeline = Pipeline(stages=[assembler, rf])
            model = pipeline.fit(train_df)
            if model_path:
                self._save_model(model, model_path)
            return model

        model_cx = _train_regressor("cx", model_path_cx)
        model_cy = _train_regressor("cy", model_path_cy)
        return model_cx, model_cy

    def train_post_clustering_model(
        self,
        posts_df: DataFrame,
        model_path: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Optional[PipelineModel]:
        if posts_df.count() < 10:
            return None
        if model_path and not force_retrain and pathlib.Path(model_path).exists():
            return self._load_model(model_path)
        assembler = VectorAssembler(inputCols=["cx", "cy"], outputCol="features")
        kmeans = KMeans(featuresCol="features", k=2, seed=42)
        pipeline = Pipeline(stages=[assembler, kmeans])
        model = pipeline.fit(posts_df)
        if model_path:
            self._save_model(model, model_path)
        return model
