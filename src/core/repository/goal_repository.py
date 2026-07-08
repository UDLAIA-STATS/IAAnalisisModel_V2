from typing import Sequence

from sqlmodel import Session, select

from src.entities.models.soccer.goal_model import GoalModel


class GoalRepository:
    @staticmethod
    def get_goals_by_frame_num(match_id, frame_number: int, session: Session) -> Sequence[GoalModel]:
        query = select(GoalModel).where(GoalModel.match_id == match_id and GoalModel.frame_number == frame_number)
        return session.exec(query).all()

    @staticmethod
    def get_goals_by_match_id(match_id: int, session: Session) -> Sequence[GoalModel]:
        query = select(GoalModel).where(GoalModel.match_id == match_id)
        return session.exec(query).all()
