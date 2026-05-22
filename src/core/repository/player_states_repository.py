from typing import Sequence, Tuple

import logfire
from sqlmodel import Session, select

from src.entities.models.soccer.player_model import PlayerModel, PlayerState


class PlayerStatesRepository:
    @staticmethod
    def get_state_by_track_id(
        frame_number: int, match_id: int, track_id: int, session: Session
    ) -> Tuple[PlayerModel, PlayerState] | Tuple[None, None] | Tuple[PlayerModel, None]:
        player_query = select(PlayerModel).where(PlayerModel.match_id == match_id).where(PlayerModel.track_id == track_id)
        player = session.exec(player_query).first()

        if player is None:
            return None, None

        state_query = select(PlayerState).where(PlayerState.player_id == player.id, PlayerState.frame_number == frame_number)
        result = session.exec(state_query).first()

        if result is None:
            return None, None

        return player, result

    @staticmethod
    def get_state_by_id(player_state_id: int, session: Session) -> PlayerState | None:
        query = select(PlayerState).where(PlayerState.id == player_state_id)
        return session.exec(query).first()

    @staticmethod
    def get_states_by_frame(match_id: int, frame_number: int, session: Session) -> Sequence[PlayerState]:
        query = (
            select(PlayerState)
            .join(PlayerModel, PlayerState.player_id == PlayerModel.id)
            .where(PlayerState.frame_number == frame_number, PlayerModel.match_id == match_id)
        )
        results = session.exec(query).all()

        if results is None or len(results) == 0:
            logfire.error(
                f"PlayerStatesRepository.get_player_states_by_frame_num: No results found for frame number "
                f"{frame_number} and match id {match_id} in database"
            )
        return results
