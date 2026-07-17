from sqlmodel import Session, select, col as modelcol
from pyspark.sql import DataFrame
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.goal_model import GoalModel
from src.core.post_processing.v2.processing_config import PostProcessingConfig
import logfire
from typing import List, Dict, Optional, Set


class DatabaseUpdater:
    def __init__(self, session: Session, config: PostProcessingConfig):
        self.session = session
        self.config = config

    def update_player_states(self, possession_df: DataFrame):
        if possession_df is None or possession_df.isEmpty():
            return
        # Actualizar has_ball en PlayerState para los frames predichos
        rows = possession_df.select("player_state_id", "possession_pred").collect()
        to_update = []
        for row in rows:
            state = self.session.get(PlayerState, row.player_state_id)
            if state and state.has_ball != (row.possession_pred == 1):
                state.has_ball = row.possession_pred == 1
                to_update.append(state)
        if to_update:
            self.session.bulk_update_mappings(
                PlayerState, [{"id": s.id, "has_ball": s.has_ball} for s in to_update]
            )
            self.session.commit()
            logfire.info(
                f"[DatabaseUpdater] Updated {len(to_update)} PlayerState records"
            )

    def update_player_stats(
        self,
        match_id: int,
        event_results: Dict[str, List[Dict]],
        possession_times: DataFrame,
    ) -> None:
        """
        Actualiza estadísticas de jugadores (goles, tiros, tiempo de posesión, goles de equipo)
        a partir de los eventos detectados (goles y tiros).
        """
        players = {
            p.id: p
            for p in self.session.exec(
                select(PlayerModel).where(PlayerModel.match_id == match_id)
            ).all()
        }
        if not players:
            return

        logfire.info(f"[DatabaseUpdater] Updating player stats for match {match_id}, Event results: {event_results}")
        if possession_times is not None and not possession_times.isEmpty():
            for row in possession_times.collect():
                pid = row.player_id
                if pid in players:
                    players[pid].ball_possession_time += int(row.possession_time)

        for goal in event_results.get("goals", []):
            pid = goal["player_id"]
            if pid in players:
                players[pid].goals += 1

        for shot in event_results.get("shots", []):
            pid = shot["player_id"]
            if pid in players:
                players[pid].shots += 1

        team_goals_map = {}
        for goal in event_results.get("goals", []):
            tid = goal["team_id"]
            if tid is not None:
                team_goals_map[tid] = team_goals_map.get(tid, 0) + 1

        for player in players.values():
            if player.team_id in team_goals_map:
                player.team_goals = team_goals_map[player.team_id]
            else:
                player.team_goals = 0

        self.session.commit()
        logfire.info("[DatabaseUpdater] Player stats updated")

    def mark_goal_frames(self, goals: List[Dict]) -> None:
        """Marca los frames de posesión como is_goal=True para los goles."""
        for goal in goals:
            if goal.get("is_goal"):
                stmt = select(PlayerState).where(
                    PlayerState.player_id == goal["player_id"],
                    PlayerState.frame_number == goal["possession_frame"],
                    PlayerState.has_ball == True,
                )
                state = self.session.exec(stmt).first()
                if state:
                    state.is_goal = True
        self.session.commit()
        logfire.info("[DatabaseUpdater] Marked goal frames")

    def update_ball_states(
        self,
        original_ball_df: DataFrame,
        winners_df: DataFrame,
        interpolated_df: DataFrame,
    ):
        winner_ids = {
            row.id for row in winners_df.select("id").collect() if row.id is not None
        }
        original_ids = {
            row.id
            for row in original_ball_df.select("id").collect()
            if row.id is not None
        }
        to_delete = original_ids - winner_ids
        for bid in to_delete:
            state = self.session.get(BallState, bid)
            if state:
                self.session.delete(state)

        if interpolated_df is not None and not interpolated_df.isEmpty():
            for row in interpolated_df.collect():
                new_state = BallState(
                    match_id=row.match_id,
                    frame_number=row.frame_number,
                    timestamp=row.timestamp,
                    confidence=row.confidence,
                    x1=row.x1,
                    y1=row.y1,
                    x2=row.x2,
                    y2=row.y2,
                    dx=row.dx,
                    dy=row.dy,
                    dx_meters=row.dx_meters,
                    dy_meters=row.dy_meters,
                    distance_meters=row.distance_meters,
                    speed_kmh=row.speed_kmh,
                    vx=row.vx,
                    vy=row.vy,
                    ax=row.ax,
                    ay=row.ay,
                    acceleration=row.acceleration,
                )
                self.session.add(new_state)

        self.session.commit()
        logfire.info("[DatabaseUpdater] Ball states updated")

    def delete_invalid_goal_posts(self, valid_post_ids: Set[int], match_id: int):
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        all_posts = self.session.exec(stmt).all()
        deleted = 0
        for post in all_posts:
            if post.id not in valid_post_ids:
                self.session.delete(post)
                deleted += 1
        self.session.commit()
        logfire.info(f"[DatabaseUpdater] Deleted {deleted} invalid goal posts")
