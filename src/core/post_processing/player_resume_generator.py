from typing import Dict, List

import httpx
import logfire
import numpy as np
import traceback

from sqlmodel import Session, select
from decimal import Decimal, ROUND_DOWN

from src.entities.models.soccer.player_model import PlayerModel
from src.entities.models.soccer.player_resume import PlayerResume
from src.config.configuration import settings

def truncate(value, decimals: int = 2) -> float:
    quantizer = Decimal("1." + "0" * decimals)
    return float(Decimal(str(value)).quantize(quantizer, rounding=ROUND_DOWN))

class PlayerResumeGenerator:

    def send_resumes(self, match_id: int, color: str, analized: bool, session: Session) -> None:
        resumes = self.generate_resumes(match_id, session)

        try:
            logfire.info(f"[PlayerResumeGenerator] Sending {len(resumes)} resumes")
            resp = httpx.post(
                url=f"{settings.STATS_NOTIFY_URL}/update-stats/",
                json={"match_id": match_id, "stats": resumes, "color": color, "analized": analized},
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
            avg_acceleration = truncate(np.mean([p.acceleration for p in player.states]), 6)
            avg_speed = truncate(np.mean([p.speed_kmh for p in player.states]), 6)
            avg_distance = truncate(np.mean([p.distance_meters for p in player.states]), 6)

            resume = PlayerResume(
                player_id=player.id,
                track_id=player.track_id,
                match_id=player.match_id,
                team_goals=player.team_goals,
                goals=player.goals,
                passes=player.shots,
                team_color=player.team_color or "",
                avg_acceleration=float(avg_acceleration),
                avg_speed_kmh=float(avg_speed),
                distance_km=float(avg_distance),
                movement_trajectories_path=player.movement_trajectories_path or "",
                player_crop_path=player.crop_path or "",
                heatmap_image_path=player.heatmap_path or "",
                team_heatmap_path=player.team_heatmap_path or "",
                avg_possession_time_s=truncate(player.ball_possession_time, 6),
                player_movement_trajectories_path=player.player_movement_trajectories_path or "",
                team_color_time_kde_path=player.team_color_time_kde_path or "",
                shirt_number=player.shirt_number or 1,
                voronoi_territories_path=player.voronoi_territories_path or "",
            )

            resumes.append(resume.model_dump())

        return resumes

resume_generator = PlayerResumeGenerator()