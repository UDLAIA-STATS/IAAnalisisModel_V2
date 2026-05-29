from functools import lru_cache
from typing import Optional, Sequence

from sqlmodel import Session, col, select

from src.entities.models.soccer.depth_history import DepthHistory

class DepthRepository:
    @staticmethod
    def get_depth_history(match_id: int, session: Session) -> Sequence[DepthHistory]:
        return session.exec(select(DepthHistory).where(DepthHistory.match_id == match_id)).all()
    
    @staticmethod
    def get_depth_by_player(match_id: int, player_id: int, frame_num: int, session: Session) -> Optional[DepthHistory]:
        query = select(DepthHistory).where(DepthHistory.match_id == match_id, DepthHistory.player_id == player_id, DepthHistory.frame_num == frame_num)
        return session.exec(query).first()

    @staticmethod
    def get_depths_by_player(match_id: int, player_id: int, session: Session):
        query = select(DepthHistory).where(DepthHistory.match_id == match_id, DepthHistory.player_id == player_id)
        return session.exec(query).all()
    
    @staticmethod
    @lru_cache(maxsize=648)
    def get_max_values(session: Session):
        max_depth_query = select(DepthHistory.depth).order_by(col(DepthHistory.depth).desc()).limit(1)
        max_pixels_to_meters_query = select(DepthHistory.pixels_to_meters).order_by(col(DepthHistory.pixels_to_meters).desc()).limit(1)

        max_depth, max_pixels_to_meters = session.exec(max_depth_query).first(), session.exec(max_pixels_to_meters_query).first()

        if max_depth is None:
            max_depth = 20
        
        if max_pixels_to_meters is None:
            max_pixels_to_meters = 0.42

        return max_depth, max_pixels_to_meters