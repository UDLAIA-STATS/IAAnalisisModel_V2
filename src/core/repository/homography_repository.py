import json
from typing import Optional, Sequence

import logfire
import numpy as np
from sqlmodel import Float, Session, and_, case, cast, col, func, or_, select
from sqlalchemy.dialects.postgresql import JSONB

from src.entities.models.homography.homography_models import CleanHomographyFrame, DetectedKeypoint
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

    @staticmethod
    def get_clean_homographies(
        session: Session,
        match_id: int,
        max_reprojection_error: float = 2.0,
        min_detected_confidence: float = 0.45,
        min_keypoints: int = 4,
        scale_x_range: tuple[float, float] = (0.11, 0.17),
        max_skew: float = 0.020,
    ) -> list[CleanHomographyFrame]:
        """
        Returns validated homography frames for a match, filtering out:
        - frames with reprojection error above threshold
        - detected keypoints below confidence threshold
                    (predetermined keypoints are always included)
        - frames with too few keypoints to constrain the homography
        - frames with anomalous matrix structure (scale or skew outliers)
        """

        H_jsonb = cast(HomographyResult.H_json, JSONB)
        scale_x = cast(H_jsonb[0][0].astext, Float)
        skew_yx = cast(H_jsonb[1][0].astext, Float)

        kpt_included = or_(
            DetectedKeypoint.source == "predetermined",
            DetectedKeypoint.confidence >= min_detected_confidence,
        )

        total_kpts    = func.count(col(DetectedKeypoint.id)).label("total_kpts")
        detected_kpts = func.count(
            case((DetectedKeypoint.source == "detected", DetectedKeypoint.id))
        ).label("detected_kpts")
        avg_det_conf  = func.avg(
            case((DetectedKeypoint.source == "detected", DetectedKeypoint.confidence))
        ).label("avg_det_conf")

        stmt = (
            select(
                HomographyResult.frame_num,
                HomographyResult.H_json,
                HomographyResult.reprojection_error,
                total_kpts,
                detected_kpts,
                avg_det_conf,
            ) # type: ignore
            .join(
                DetectedKeypoint,
                DetectedKeypoint.homography_result_id == HomographyResult.id,
            )
            .where(
                and_(
                    HomographyResult.match_id == match_id,
                    HomographyResult.is_valid == True,
                    HomographyResult.reprojection_error <= max_reprojection_error,
                    kpt_included,
                    scale_x.between(*scale_x_range),
                    func.abs(skew_yx) < max_skew,
                )
            )
            .group_by(
                HomographyResult.frame_num,
                HomographyResult.H_json,
                HomographyResult.reprojection_error,
            )
            .having(func.count(col(DetectedKeypoint.id)) >= min_keypoints)
            .order_by(HomographyResult.frame_num)
        )

        rows = session.exec(stmt).all()
        logfire.notice("[Homography Repository] got {} of expected 6 homographies for match {}".format(len(rows), match_id))

        return [
            CleanHomographyFrame(
                frame_num=row.frame_num,
                H=np.array(json.loads(row.H_json), dtype=np.float32),
                reprojection_error=row.reprojection_error,
                total_kpts=row.total_kpts,
                detected_kpts=row.detected_kpts,
                avg_det_conf=row.avg_det_conf,
            )
            for row in rows
        ]
