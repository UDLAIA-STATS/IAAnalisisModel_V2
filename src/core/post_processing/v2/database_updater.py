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
                state.has_ball = (row.possession_pred == 1)
                to_update.append(state)
        if to_update:
            self.session.bulk_update_mappings(PlayerState, 
                [{"id": s.id, "has_ball": s.has_ball} for s in to_update])
            self.session.commit()
            logfire.info(f"[DatabaseUpdater] Updated {len(to_update)} PlayerState records")

    def update_player_stats(self, match_id: int, possession_times: DataFrame,
                            shot_counts: Optional[DataFrame], goal_assignments: Dict[int, int]):
        # Obtener jugadores
        players = {p.id: p for p in self.session.exec(select(PlayerModel).where(PlayerModel.match_id == match_id)).all()}
        if not players:
            return

        # Procesar tiempos de posesión
        if possession_times is not None and not possession_times.isEmpty():
            for row in possession_times.collect():
                pid = row.player_id
                if pid in players:
                    players[pid].ball_possession_time += int(row.possession_time)

        # Procesar disparos
        if shot_counts is not None and not shot_counts.isEmpty():
            for row in shot_counts.collect():
                pid = row.player_id
                if pid in players:
                    players[pid].shots += int(row.shots)

        # Procesar goles
        for pid, goals in goal_assignments.items():
            if pid in players:
                players[pid].goals += goals

        # Actualizar team_goals (por equipo)
        team_goals = {}
        for pid, player in players.items():
            if player.team_id:
                team_goals[player.team_id] = team_goals.get(player.team_id, 0) + player.goals
        for player in players.values():
            if player.team_id in team_goals:
                player.team_goals = team_goals[player.team_id]

        # Commit
        self.session.commit()
        logfire.info("[DatabaseUpdater] Player stats updated")

    def update_ball_states(self, original_ball_df: DataFrame, winners_df: DataFrame,
                           interpolated_df: DataFrame):
        # Eliminar detecciones no ganadoras
        winner_ids = {row.id for row in winners_df.select("id").collect() if row.id is not None}
        original_ids = {row.id for row in original_ball_df.select("id").collect() if row.id is not None}
        to_delete = original_ids - winner_ids
        for bid in to_delete:
            state = self.session.get(BallState, bid)
            if state:
                self.session.delete(state)

        # Insertar interpolados
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

        # Actualizar métricas de ganadores (ya están en winners_df, pero podrían actualizarse)
        # (Omitimos por brevedad)
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

    def mark_goal_frames(self, goal_links: List[Dict]):
        for link in goal_links:
            if link.get("is_goal"):
                stmt = select(PlayerState).where(
                    PlayerState.player_id == link["player_id"],
                    PlayerState.frame_number == link["possession_frame"],
                    PlayerState.has_ball == True
                )
                state = self.session.exec(stmt).first()
                if state:
                    state.is_goal = True
        self.session.commit()
        logfire.info("[DatabaseUpdater] Marked goal frames")