from typing import Optional, Tuple
import logfire
from pyspark.sql import DataFrame
from sqlmodel import Session

from src.entities.predictors.base_processor import BaseProcessor
from src.core.post_processing.ball_predictor import BallPredictor
from src.core.post_processing.goal_predictor import GoalPostValidator
from src.core.post_processing.events_predictor import BallPossessionMLAnalyzer
from src.core.post_processing.ball_posession_predictor import BallPossessionPredictor
from src.core.post_processing.goal_scorer_detector import GoalScorerDetector

class MatchPostProcessor(BaseProcessor):
    def predict(self, match_id: int, total_frames: int, fps: float, session: Session) -> None:
        logfire.info(f"[MatchPostProcessor] Starting full post-processing for match {match_id}")

        ball_df = self.fetch_ball_states(match_id, session)
        player_df = self.fetch_player_states(match_id, session)
        goal_df = self.fetch_goals(match_id, session)

        if ball_df.isEmpty() or player_df.isEmpty():
            logfire.warning("No ball or player data; aborting.")
            return


