from typing import Optional, Sequence

from sqlmodel import Session, select
from src.entities.models.homography import HomographyResult

def _add_homography(homography: HomographyResult, session: Session):
    session.add(homography)
    session.flush()
    return homography

class HomographyRepository:
    @staticmethod
    def get_homographies_by_match_id(match_id: int, session: Session) -> Sequence[HomographyResult]:
        query = select(HomographyResult).where(HomographyResult.match_id == match_id)
        return session.exec(query).all()
    @staticmethod
    def get_homography(match_id: int, frame_number: int, session: Session) -> Optional[HomographyResult]:
        query = (
            select(HomographyResult)
            .where(HomographyResult.match_id == match_id and HomographyResult.frame_num == frame_number))
        return session.exec(query).first()


    @staticmethod
    def upsert_homography(match_id: int, frame_number, homography: HomographyResult, session: Session):
        homography_db = HomographyRepository.get_homography(match_id, frame_number, session)

        if homography_db:
            homography_db = session.get(HomographyResult, homography_db.id)
            if not homography_db:
                homography = _add_homography(homography, session)
                return homography
        else:
            homography = _add_homography(homography, session)
            return homography

        homography_db.sqlmodel_update(homography.model_dump(exclude_unset=True))
        session.add(homography_db)
        session.flush()

        return homography

