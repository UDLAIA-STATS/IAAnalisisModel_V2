from functools import lru_cache
from typing import Sequence

from sqlmodel import Session, select

from src.entities.models.soccer.ball_model import BallState


class BallRepository:
    @staticmethod
    def get_ball_by_frame_num(match_id: int, frame_num: int, session: Session) -> BallState | None:
        query = select(BallState).where(BallState.match_id == match_id and BallState.frame_number == frame_num)
        return session.exec(query).first()

    @staticmethod
    @lru_cache
    def get_balls_by_match_id(match_id: int, session: Session) -> Sequence[BallState]:
        query = select(BallState).where(BallState.match_id == match_id)
        return session.exec(query).all()
