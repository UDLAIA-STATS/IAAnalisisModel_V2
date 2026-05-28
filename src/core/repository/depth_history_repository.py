from typing import Optional, Sequence

from sqlmodel import Session, select

from src.entities.models.soccer.depth_history import DepthHistory

class DepthRepository:
    @staticmethod
    def get_depth_history(match_id: int, session: Session) -> Sequence[DepthHistory]:
        return session.exec(select(DepthHistory).where(DepthHistory.match_id == match_id)).all()
    
    @staticmethod
    def get_depth_by_player(match_id: int, player_id: int, frame_num: int, session: Session) -> Optional[DepthHistory]:
        query = select(DepthHistory).where(DepthHistory.match_id == match_id, DepthHistory.player_id == player_id, DepthHistory.frame_num == frame_num)
        return session.exec(query).first()
