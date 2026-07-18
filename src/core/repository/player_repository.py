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
    
    @staticmethod
    def delete_player(player_id: int, session: Session) -> None:
        player = session.get(PlayerModel, player_id)

        if player is None:
            return

        # for state in player.states:
        #     session.delete(state)

        session.delete(player)
        session.commit()

    
    @staticmethod
    def upload_heatmap(player_id: int, heatmap_url: str, session: Session) -> None:
        player = session.get(PlayerModel, player_id)
        if not player:
            return

        player.heatmap_path = heatmap_url

        session.add(player)
        session.flush()

    @staticmethod
    def upload_match_files(team_heatmap_path: str, movement_trajectories_path: str, match_id: int, session: Session):
        players = session.exec(
            select(PlayerModel).where(PlayerModel.match_id == match_id)
        ).all()

        for player in players:
            player_db = session.get(PlayerModel, player.id)

            if not player_db:
                continue

            player_db.team_heatmap_path = team_heatmap_path
            player_db.movement_trajectories_path = movement_trajectories_path

            session.add(player)
            session.flush()
