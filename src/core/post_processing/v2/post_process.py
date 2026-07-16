import traceback
from sqlmodel import Session
import logfire

from src.core.post_processing.v2.processing_config import PostProcessingConfig
from src.core.post_processing.v2.data_cleaner import DataCleaner
from src.core.post_processing.v2.feature_engineer import FeatureEngineer
from src.core.post_processing.v2.model_trainer import ModelTrainer
from src.core.post_processing.v2.event_detector import EventDetector
from src.core.post_processing.v2.database_updater import DatabaseUpdater

from src.entities.utils.spark_instance import spark
from src.core.post_processing.v2.data_loader import DataLoader


class PostProcessor:
    def __init__(self, match_id: int, session: Session, config: PostProcessingConfig):
        self.match_id = match_id
        self.session = session
        self.config = config
        self.spark = spark

        self.loader = DataLoader(session)
        self.cleaner = DataCleaner(config)
        self.engineer = FeatureEngineer(config)
        self.trainer = ModelTrainer(config)
        self.detector = EventDetector(config)
        self.updater = DatabaseUpdater(session, config)

    def run(self):
        logfire.info(f"[Pipeline] Starting for match {self.match_id}")
        try:
            raw_ball_df, player_df, goals_df, players_df = self.loader.load_all(
                self.match_id
            )

            if raw_ball_df is None or player_df is None:
                logfire.warning("[Pipeline] Missing ball or player data; aborting.")
                return

            ball_df = self.cleaner.clean_ball_states(raw_ball_df)
            player_df = self.cleaner.clean_player_states(player_df)
            goals_df = self.cleaner.clean_goal_posts(goals_df)

            if self.config.cache_dataframes:
                ball_df.cache()
                player_df.cache()

            possession_features = self.engineer.build_possession_features(
                ball_df, player_df
            )
            goal_line = (
                self.engineer.compute_goal_line(goals_df, ball_df)
                if goals_df is not None and not goals_df.isEmpty()
                else None
            )
            goal_candidates = None
            shot_features = None
            if goal_line and goals_df is not None and not goals_df.isEmpty():
                goal_candidates = self.engineer.build_goal_linking_features(
                    goals_df, ball_df, player_df, goal_line
                )
                shot_features = self.engineer.build_shot_features(
                    goals_df, ball_df, player_df, goal_line
                )

            # 4. Entrenamiento de modelos (si use_ml)
            possession_model = None
            goal_model = None
            shot_model = None
            traj_model_cx = None
            traj_model_cy = None

            if self.config.use_ml:
                # Modelo de posesión
                if (
                    possession_features is not None
                    and not possession_features.isEmpty()
                ):
                    model_path = (
                        f"{self.config.model_dir.as_posix()}/possession_{self.match_id}"
                    )
                    possession_model = self.trainer.train_possession_model(
                        possession_features, model_path, self.config.force_retrain
                    )

                # Modelo de goleador
                if goal_candidates is not None and not goal_candidates.isEmpty():
                    model_path = f"{self.config.model_dir.as_posix()}/goal_linker_{self.match_id}"
                    goal_model = self.trainer.train_goal_linker_model(
                        goal_candidates, model_path, self.config.force_retrain
                    )

                # Modelo de disparos
                if shot_features is not None and not shot_features.isEmpty():
                    model_path = (
                        f"{self.config.model_dir.as_posix()}/shot_{self.match_id}"
                    )
                    shot_model = self.trainer.train_shot_model(
                        shot_features, goals_df, model_path, self.config.force_retrain
                    )

                # Modelos de trayectoria
                if ball_df is not None and not ball_df.isEmpty():
                    model_cx_path = (
                        f"{self.config.model_dir.as_posix()}/traj_cx_{self.match_id}"
                    )
                    model_cy_path = (
                        f"{self.config.model_dir.as_posix()}/traj_cy_{self.match_id}"
                    )
                    traj_model_cx, traj_model_cy = self.trainer.train_trajectory_model(
                        ball_df, model_cx_path, model_cy_path, self.config.force_retrain
                    )

            # 5. Detección de eventos
            # Posesión
            possession_pred = self.detector.predict_possession(
                possession_features, possession_model
            )
            if self.config.cache_dataframes and possession_pred is not None:
                possession_pred.cache()

            # Asignación de goles
            goal_results = []
            goal_links = {}
            if goal_candidates is not None and not goal_candidates.isEmpty():
                goal_results, goal_links = self.detector.predict_goal_scorer(
                    goal_candidates, goal_model
                )

            # Disparos
            shot_pred = None
            if shot_features is not None and not shot_features.isEmpty():
                shot_pred = self.detector.predict_shots(shot_features, shot_model)
                if self.config.cache_dataframes and shot_pred is not None:
                    shot_pred.cache()

            # Tiempo de posesión y conteo de disparos
            possession_times = self.detector.compute_possession_time(possession_pred)
            shot_counts = (
                self.detector.compute_shot_counts(shot_pred)
                if shot_pred is not None
                else None
            )

            # Trayectoria e interpolación
            interpolated_ball = None
            if (
                traj_model_cx is not None
                and traj_model_cy is not None
                and ball_df is not None
            ):
                total_frames = ball_df.select("frame_number").distinct().count()
                traj_df = self.detector.predict_trajectory(
                    traj_model_cx, traj_model_cy, total_frames, self.match_id
                )
                if traj_df is not None:
                    # winners_df es ball_df después de limpieza (ya tiene uno por frame)
                    interpolated_ball = self.detector.interpolate_ball_states(
                        ball_df, traj_df, total_frames
                    )

            # 6. Actualización de BD
            self.updater.update_player_states(possession_pred)
            if possession_times is not None or shot_counts is not None or goal_results:
                self.updater.update_player_stats(
                    self.match_id, possession_times, shot_counts, goal_links
                )
            if interpolated_ball is not None:
                self.updater.update_ball_states(raw_ball_df, ball_df, interpolated_ball)
            if goals_df is not None:
                valid_ids = set(
                    goals_df.select("id").rdd.flatMap(lambda x: x).collect()
                )
                self.updater.delete_invalid_goal_posts(valid_ids, self.match_id)
            if goal_results:
                self.updater.mark_goal_frames(goal_results)

            logfire.info(f"[Pipeline] Completed for match {self.match_id}")

        except Exception as e:
            logfire.error(f"[Pipeline] Failed: {traceback.format_exc()[:300]}")
            self.session.rollback()
            raise
            # if not self.config.fallback_to_heuristic:
            #     raise
            # else:
            #     logfire.warning("[Pipeline] Falling back to heuristic processing (partial results may be saved).")
            #     # Podríamos ejecutar una versión reducida con solo heurísticas aquí
            #     # (se omite por brevedad)
