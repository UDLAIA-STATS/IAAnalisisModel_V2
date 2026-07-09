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
        max_reprojection_error: float = 3.0,        # raised from 2.0 — valid frames sit at 2.21px
        min_detected_confidence: float = 0.40,
        min_keypoints: int = 4,
        min_inliers: int = 6,                        # replaces scale_x_range + max_skew
    ) -> list[CleanHomographyFrame]:
        """
        Returns validated homography frames for a match, filtering out:
        - frames with reprojection error above threshold
        - frames with fewer inliers than min_inliers
        - frames with too few keypoints to constrain the homography
        - duplicate result rows per frame (keeps highest inliers, then lowest error)

        Removed vs original:
        - scale_x_range / max_skew: H[0][0] encodes zoom+tilt simultaneously so a
        fixed window silently drops valid wide/tilted frames. Geometry validity is
        already enforced at computation time via validate_homography().
        - kpt_included WHERE filter: filtering join rows before GROUP BY corrupted
        frame inclusion. Confidence is now checked in the list comprehension so
        aggregation counts remain correct.
        """

        total_kpts = func.count(col(DetectedKeypoint.id)).label("total_kpts")

        detected_kpts = func.count(
            case((DetectedKeypoint.source == "detected", DetectedKeypoint.id))
        ).label("detected_kpts")

        predetermined_kpts = func.count(
            case((DetectedKeypoint.source == "predetermined", DetectedKeypoint.id))
        ).label("predetermined_kpts")

        avg_conf = func.avg(
            DetectedKeypoint.confidence
        ).label("avg_conf")

        avg_det_conf = func.avg(
            case((DetectedKeypoint.source == "detected", DetectedKeypoint.confidence))
        ).label("avg_det_conf")

        row_num = (
            func.row_number()
            .over(
                partition_by=col(HomographyResult.frame_num),
                order_by=[
                    col(HomographyResult.inlier_count).desc(),
                    col(HomographyResult.reprojection_error).asc(),
                ],
            )
            .label("rn")
        )

        inner = (
            select(
                HomographyResult.frame_num,
                HomographyResult.H_json,
                HomographyResult.reprojection_error,
                HomographyResult.inlier_count,
                total_kpts,
                detected_kpts,
                predetermined_kpts,
                avg_conf,
                avg_det_conf,
                row_num,
            )
            .join(
                DetectedKeypoint,
                DetectedKeypoint.homography_result_id == HomographyResult.id,
            )
            .where(
                and_(
                    HomographyResult.match_id == match_id,
                    HomographyResult.is_valid == True,
                )
            )
            .group_by(
                HomographyResult.frame_num,
                HomographyResult.id,
                HomographyResult.H_json,
                HomographyResult.reprojection_error,
                HomographyResult.inlier_count,
            )
            .having(func.count(col(DetectedKeypoint.id)) >= min_keypoints)
            .subquery()
        )

        stmt = (
            select(
                inner.c.frame_num,
                inner.c.H_json,
                inner.c.reprojection_error,
                inner.c.inlier_count,
                inner.c.total_kpts,
                inner.c.detected_kpts,
                inner.c.predetermined_kpts,
                inner.c.avg_conf,
                inner.c.avg_det_conf,
            )
            .where(
                and_(
                    inner.c.rn == 1,
                    inner.c.reprojection_error <= max_reprojection_error,
                    inner.c.inlier_count >= min_inliers,
                )
            )
            .order_by(inner.c.frame_num)
        )

        rows = session.exec(stmt).all()  # type: ignore
        logfire.notice(
            "[Homography Repository] got {} clean homographies for match {}".format(
                len(rows), match_id
            )
        )

        return [
            CleanHomographyFrame(
                frame_num=row.frame_num,
                H=np.array(json.loads(row.H_json), dtype=np.float32),
                reprojection_error=row.reprojection_error,
                inlier_count=row.inlier_count,
                total_kpts=row.total_kpts,
                detected_kpts=row.detected_kpts,
                predetermined_kpts=row.predetermined_kpts,
                avg_conf=row.avg_conf,
                avg_det_conf=row.avg_det_conf,
            )
            for row in rows
            if row.avg_det_conf is None or row.avg_det_conf >= min_detected_confidence
        ]

