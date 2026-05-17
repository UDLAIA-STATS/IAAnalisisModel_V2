from typing import Sequence

from sqlmodel import Session, select

from entities.models.soccer.player_model import PlayerModel


class PlayerRepository:
    
    def get_player_by_track_id(self, track_id: int, match_id: int, session: Session) -> PlayerModel | None:
        query = select(PlayerModel).where(PlayerModel.track_id == track_id and PlayerModel.match_id == match_id)
        return session.exec(query).first()
    
    def get_players_by_match_id(self, match_id: int, session: Session) -> Sequence[PlayerModel] | None:
        query = select(PlayerModel).where(PlayerModel.match_id == match_id)
        return session.exec(query).all()
    
    def get_player_by_id(self, player_id: int, session: Session) -> PlayerModel | None:
        query = select(PlayerModel).where(PlayerModel.id == player_id)
        return session.exec(query).first()