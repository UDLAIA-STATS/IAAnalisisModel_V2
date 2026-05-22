from collections import defaultdict
from typing import List

import numpy as np
from sqlmodel import Session

from core.repository.player_states_repository import PlayerStatesRepository
from entities.models.soccer.player_model import PlayerState


class PlayerValidator:
    MAX_DELTA: int = 40
    MIN_FRAMES = 10

    def validate(self, match_id: int, total_frames: int, session: Session):
        states = PlayerStatesRepository.get_states_by_frame_range(
            match_id=match_id,
            min_frame=1,
            max_frame=self.MIN_FRAMES,
            session=session
        )        
        players = defaultdict(list)

        for state in states:
            players[state.player_id].append(state)

        player_centroids = {}
        
        for player_id, player_states in players.items():
            centroids = [
                self.calculate_centroid(state)
                for state in player_states
            ]
            avg_x = sum(x for x, _ in centroids) / len(centroids)
            avg_y = sum(y for _, y in centroids) / len(centroids)
            player_centroids[player_id] = (avg_x, avg_y)
        
        merged = set()

        for player_a, centroid_a in player_centroids.items():
            if player_a in merged:
                continue

            for player_b, centroid_b in player_centroids.items():
                if player_a == player_b or player_b in merged:
                    continue

                distance = self.calculate_distance(centroid_a, centroid_b)

                if distance < self.MAX_DELTA:
                    PlayerStatesRepository.merge_states(player_a, player_b, session)
                    merged.add(player_b)
        
        session.commit()

    def calculate_distance(self, player1: PlayerState, player2: PlayerState):
        cx1, cy1 = self.calculate_centroid(player1)
        cx2, cy2 = self.calculate_centroid(player2)

        return np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

    def calculate_centroid(self, player: PlayerState):
        cx = (player.x1 + player.x2) / 2
        cy = (player.y1 + player.y2) / 2
        return cx, cy
