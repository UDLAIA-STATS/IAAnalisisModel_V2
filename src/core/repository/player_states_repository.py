from typing import Tuple

from sqlmodel import Session, select

from entities.models.soccer.player_model import PlayerModel, PlayerState


class PlayerStatesRepository:
    @staticmethod
    def get_player_state_by_track_id(
            frame_number: int,
            match_id: int, track_id: int,
            session: Session) -> Tuple[PlayerModel, PlayerState] | Tuple[None, None] | Tuple[PlayerModel, None]:
        player_query = select(PlayerModel).where(
            PlayerModel.track_id == track_id and
            PlayerModel.match_id == match_id)

        player_result = session.exec(player_query).first()

        if player_result is None:
            return None, None
        
        states_query = select(PlayerState).where(
            PlayerState.player_id == player_result.id and
            PlayerState.frame_number == frame_number
        )

        states_result = session.exec(states_query).first()

        if states_result is None:
            return player_result, None

        return player_result, states_result

    @staticmethod
    def get_player_state_by_id(player_state_id: int, session: Session) -> PlayerState | None:
        query = select(PlayerState).where(PlayerState.id == player_state_id)
        return session.exec(query).first()
    
