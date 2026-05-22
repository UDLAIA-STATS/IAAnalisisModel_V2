import numpy as np
from sqlmodel import Session

from core.repository.player_repository import PlayerRepository
from entities.models.soccer.player_model import PlayerState


class PlayerValidator:
    MAX_DELTA: int = 40
    MIN_FRAMES = 10

    def validate(self, match_id: int, session: Session):
        players = PlayerRepository.get_players_by_match_id(match_id, session)
    
    def calculate_distance(self, player1: PlayerState, player2: PlayerState):
        cx1, cy1 = self.calculate_centroid(player1)
        cx2, cy2 = self.calculate_centroid(player2)
        
        return np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    
    def calculate_centroid(self, player: PlayerState):
        cx = (player.x1 + player.x2) / 2
        cy = (player.y1 + player.y2) / 2
        return cx, cy