import math
from typing import Dict, List, Tuple

import logfire
import numpy as np
from sqlmodel import Session

from core.repository.player_repository import PlayerRepository
from entities.models.soccer.player_model import PlayerState


class PhysicsCalculatorBase:
    def calculate_distance(
        self, xo: np.ndarray, xf: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        delta_x = np.array([xo[0] - xf[0], xo[1] - xf[1]])
        return math.fabs(np.linalg.norm(delta_x)), delta_x

    def calculate_speed(
        self, delta_x: np.ndarray, delta_t: float
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        vo = delta_x / delta_t
        acceleration = (2 * (delta_x - vo * delta_t)) / delta_t**2
        vf = vo + acceleration * delta_t

        speed_ms = np.linalg.norm(vf)
        speed_kmh = float(speed_ms * 3.6)
        acceleration_ms = float(np.linalg.norm(acceleration))

        return vo, vf, acceleration_ms, speed_kmh

    def get_players(
        self, match_id: int, session: Session
    ) -> Dict[int, List[PlayerState]]:
        players = PlayerRepository.get_players_by_match_id(match_id, session)
        states: Dict[int, List[PlayerState]] = {}

        if len(players) == 0:
            logfire.error(
                f"[PhysicsCalculator] No players found for match {match_id} in database"
            )
            return states

        for player in players:
            if player.track_id not in states:
                states[player.track_id] = []
            states[player.track_id].extend(player.states)

        for value in states.values():
            value.sort(key=lambda state: state.frame_number)

        return states

    def _bbox_to_center(self, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        return np.asarray([cx, cy])
