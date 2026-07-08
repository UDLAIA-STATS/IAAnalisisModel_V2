from typing import Sequence

from sqlmodel import Session, select

from src.entities.models.soccer.player_model import PlayerModel


class PlayerRepository:
    @staticmethod
    def get_player_by_track_id(track_id: int, match_id: int, session: Session) -> PlayerModel | None:
        query = select(PlayerModel).where(PlayerModel.track_id == track_id and PlayerModel.match_id == match_id)
        return session.exec(query).first()

    @staticmethod
    def get_players_by_match_id(match_id: int, session: Session) -> Sequence[PlayerModel]:
        query = select(PlayerModel).where(PlayerModel.match_id == match_id)
        return session.exec(query).all()

    @staticmethod
    def get_player_by_id(player_id: int, session: Session) -> PlayerModel | None:
        return session.get(PlayerModel, player_id)

    @staticmethod
    def upsert_player(player: PlayerModel, session: Session) -> int:
        existing = session.exec(
            select(PlayerModel)
            .where(PlayerModel.match_id == player.match_id, PlayerModel.track_id == player.track_id)
        ).first()

        if existing:
            existing.sqlmodel_update(player.model_dump(exclude_unset=True))
            session.add(existing)
            return existing.id

        session.add(player)
        session.flush()
        player_id = player.id
        return player_id

    
