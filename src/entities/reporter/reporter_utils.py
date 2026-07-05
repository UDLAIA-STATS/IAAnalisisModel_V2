from collections import defaultdict
from typing import Dict, List

from sqlmodel import Session, select, col

from src.entities.models.soccer.player_model import PlayerState, PlayerModel


def group_states_by_id(match_id: int, session: Session) -> Dict[int, List[PlayerState]]:
    query = (
        select(PlayerState)
        .join(PlayerModel, col(PlayerState.player_id) == col(PlayerModel.id))
        .where(PlayerModel.match_id == match_id)
        .order_by(col(PlayerState.player_id), col(PlayerState.frame_number))
    )

    results = session.exec(query).all()

    grouped: Dict[int, List[PlayerState]] = defaultdict(list)

    for state in results:
        grouped[state.player_id].append(state)

    return dict(grouped)