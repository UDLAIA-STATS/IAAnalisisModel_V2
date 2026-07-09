from typing import Dict, List

import httpx
import logfire
import numpy as np
import traceback

from sqlmodel import Session, select

from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.player_resume import PlayerResume
from src.config.configuration import settings

class PlayerResumeGenerator:

    def send_resumes(self, match_id: int, color: str, session: Session) -> None:
        resumes = self.generate_resumes(match_id, session)

        try:
            logfire.info(f"[PlayerResumeGenerator] Sending {len(resumes)} resumes")
            resp = httpx.post(
                url=f"{settings.STATS_NOTIFY_URL}/update-stats/",
                json={"match_id": match_id, "stats": resumes, "color": color},
            )
            logfire.info(f"[PlayerResumeGenerator] Resumes sent, status code: {resp.status_code}")
            resp.raise_for_status()
        except httpx.HTTPError as e:
            logfire.error(f"[PlayerResumeGenerator] Error sending resumes: {traceback.format_exc()[:300]}")
            raise
        except Exception as e:
            logfire.error(f"[PlayerResumeGenerator] Error sending resumes: {traceback.format_exc()[:300]}")
            raise


    def generate_resumes(self, match_id: int, session: Session) -> List[Dict]:
        players = session.exec(
            select(PlayerModel).where(PlayerModel.match_id == match_id)
        ).all()

        resumes: List[Dict] = []

        for player in players:
            avg_acceleration = np.mean([p.acceleration for p in player.states])
            avg_speed = np.mean([p.speed_kmh for p in player.states])
            avg_distance = np.mean([p.distance_meters for p in player.states])

            resume = PlayerResume(
                player_id=player.id,
                track_id=player.track_id,
                match_id=player.match_id,
                goals=player.goals,
                team_color=player.team_color,
                avg_acceleration=float(avg_acceleration),
                avg_speed=float(avg_speed),
                avg_distance=float(avg_distance),
                movement_trajectories_path=player.movement_trajectories_path,
                player_crop_path=player.crop_path,
                player_heatmap_path=player.heatmap_path,
                team_heatmap_path=player.team_heatmap_path,
                ball_possession_time=player.ball_possession_time,
                shirt_number=player.shirt_number,
            )

            resumes.append(resume.model_dump())

        return resumes

resume_generator = PlayerResumeGenerator()