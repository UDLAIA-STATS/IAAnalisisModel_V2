from functools import lru_cache
from typing import List, Sequence, Tuple

import logfire
from sqlmodel import Session, col, func, select

from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.depth_history import DepthHistory
from src.core.repository.depth_history_repository import DepthRepository

class PlayerStatesRepository:
    @staticmethod
    def get_state_by_track_id(
        frame_number: int, match_id: int, track_id: int, session: Session
    ) -> Tuple[PlayerModel, PlayerState] | Tuple[None, None] | Tuple[PlayerModel, None]:
        player_query = select(PlayerModel).where(PlayerModel.match_id == match_id, PlayerModel.track_id == track_id)
        player = session.exec(player_query).first()

        if player is None:
            return None, None

        state_query = select(PlayerState).where(PlayerState.player_id == player.id, PlayerState.frame_number == frame_number)
        state = session.exec(state_query).first()

        if state is None:
            return None, None

        return player, state

    @staticmethod
    def get_state_by_id(player_state_id: int, session: Session) -> PlayerState | None:
        query = select(PlayerState).where(PlayerState.id == player_state_id)
        return session.exec(query).first()

    @staticmethod
    def get_states_by_frame(match_id: int, frame_number: int, session: Session) -> Sequence[PlayerState]:
        query = (
            select(PlayerState)
            .join(target=PlayerModel, onclause=col(PlayerState.player_id) == col(PlayerModel.id), full=True)
            .where(PlayerState.frame_number == frame_number, PlayerModel.match_id == match_id)
        )
        results = session.exec(query).all()

        if results is None or len(results) == 0:
            logfire.error(
                f"PlayerStatesRepository.get_player_states_by_frame_num: No results found for frame number "
                f"{frame_number} and match id {match_id} in database"
            )
        return results

    @staticmethod
    def get_states_by_frame_range(match_id: int, min_frame: int, max_frame: int, session: Session) -> Sequence[PlayerState]:
        states = session.exec(
            select(PlayerState)
            .join(target=PlayerModel, onclause=col(PlayerState.player_id) == col(PlayerModel.id), full=True)
            .where(PlayerState.frame_number >= min_frame, PlayerState.frame_number <= max_frame, PlayerModel.match_id == match_id)
            .order_by(col(PlayerState.frame_number))).all()

        return states
    
    @staticmethod
    def merge_states(keep_player_id: int, remove_player_id: int, session: Session):
        states = session.exec(
            select(PlayerState)
            .where(PlayerState.player_id == remove_player_id)
        ).all()

        depths = DepthRepository.get_depths_by_player(match_id=states[0].player.match_id, player_id=remove_player_id, session=session)
        for depth in depths:
            depth = session.get(DepthHistory, depth.id)
            depth.player_id = keep_player_id # type: ignore
            session.add(depth)
            session.flush()

        for state in states:
            state.player_id = keep_player_id
            session.add(state)
            session.flush()

        duplicate_player = session.get(
            PlayerModel, remove_player_id
        )

        if duplicate_player:
            session.delete(duplicate_player)

        session.flush()
    
    @staticmethod
    def update_state(state: PlayerState, session: Session):
        original_state = session.get(PlayerState, state.id)

        if not original_state:
            logfire.error(f"PlayerStatesRepository.update_state: No state found for id {state.id} in database")
            raise ValueError(f"PlayerStatesRepository.update_state: No state found for id {state.id} in database")

        original_state.sqlmodel_update(state.model_dump(exclude_unset=True))
        session.flush()
 
    @staticmethod
    def get_max_aparitions(match_id: int, session: Session):
        aparitions = func.count(col(PlayerState.player_id)).label("aparitions")  # store it

        query = (
            select(aparitions)
            .join(target=PlayerModel, onclause=col(PlayerState.player_id) == col(PlayerModel.id), full=True)
            .where(PlayerModel.match_id == match_id)
            .group_by(col(PlayerState.player_id))
            .order_by(aparitions.desc())
        )
        appearance = session.exec(query).first()

        logfire.info(f"PlayerStatesRepository.get_max_aparitions: aparitions: {appearance}")

        if appearance is None:
            appearance = 180

        return appearance // 4


    @staticmethod
    def get_states_appearances(
        match_id: int,
        limit_appearance: int,
        session: Session
    ) -> Tuple[List[PlayerState], List[PlayerState]]:

        appearances_subquery = (
            select(
                PlayerState.player_id,
                func.count(col(PlayerState.id)).label("appearances")
            )
            .join(
                PlayerModel,
                col(PlayerState.player_id) == col(PlayerModel.id)
            )
            .where(PlayerModel.match_id == match_id)
            .group_by(col(PlayerState.player_id))
            .subquery()
        )

        below_player_ids = (
            select(appearances_subquery.c.player_id)
            .where(appearances_subquery.c.appearances <= limit_appearance)
        )

        correct_player_ids = (
            select(appearances_subquery.c.player_id)
            .where(appearances_subquery.c.appearances > limit_appearance)
        )

        below_states = session.exec(
            select(PlayerState)
            .where(col(PlayerState.player_id).in_(below_player_ids))
        ).all()

        correct_states = session.exec(
            select(PlayerState)
            .where(col(PlayerState.player_id).in_(correct_player_ids))
        ).all()

        logfire.notice(
            f"[PlayerStatesRepository.get_states_appearances] below states: {len(below_states)}"
        )

        logfire.notice(
            f"[PlayerStatesRepository.get_states_appearances] correct states: {len(correct_states)}"
        )

        return list(below_states), list(correct_states)

    @staticmethod
    def recalculate_physics(match_id: int, session: Session):
        states = session.exec(
            select(PlayerState)
            .join(target=PlayerModel, onclause=col(PlayerState.player_id) == col(PlayerModel.id), full=True)
            .where(PlayerModel.match_id == match_id)
            .order_by(col(PlayerState.frame_number))).all()

        for state in states:
            state.recalculate_physics()
            session.add(state)
            session.flush()
