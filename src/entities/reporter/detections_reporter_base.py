import csv
from pathlib import Path
from typing import List, Tuple

import logfire

from sqlmodel import Session

from src.entities.reporter.diagrams_generator import DiagramsGenerator
from src.core.reporter.detections_reporter import ReportRow
from src.core.repository.homography_repository import HomographyRepository
from src.entities.types.bucket_types import FilePurposeTypes
from src.core.repository.ball_repository import BallRepository
from src.core.repository.goal_repository import GoalRepository
from src.core.repository.player_repository import PlayerRepository
from src.core.repository.r2_repository import files_repository


class DetectionsReporterBase(DiagramsGenerator):

    def generate_report(
        self, match_id: int, session: Session
    ) -> Tuple[str, str, str, str, str, str, str]:
        raise NotImplementedError

    def upload_reports(self, reports: list[tuple[str, str]], match_id: int):
        chart_keys = {}
        for name, path_str in reports:
            if path_str:
                p = Path(path_str)
                key = files_repository.generate_key(
                    match_id, p.stem, FilePurposeTypes.REPORTS, p.suffix[1:]
                )
                files_repository.upload_report(key, p)
                chart_keys[name] = key
            else:
                chart_keys[name] = None

        return chart_keys

    def get_report_rows(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = []

        report_rows.extend(self.get_goals(match_id, session))
        report_rows.extend(self.get_balls(match_id, session))
        report_rows.extend(self.get_players(match_id, session))
        report_rows.extend(self.get_homography_values(match_id, session))

        return report_rows

    def write_detections_report(self, report_path: Path, report_rows: List[ReportRow]):
        with open(report_path, mode="w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)

            csv_writer.writerow(
                [
                    "frame_number",
                    "object_type",
                    "track_id",
                    "dx_meters",
                    "dy_meters",
                    "bbox",
                    "confidence",
                    "shirt_color",
                    "shirt_number",
                    "speed",
                    "distance",
                    "acceleration",
                    "timestamp",
                    "homography_results",
                ]
            )

            for row in report_rows:
                csv_writer.writerow(
                    [
                        row.frame_number,
                        row.object_type,
                        row.track_id,
                        row.dx_meters,
                        row.dy_meters,
                        row.bbox,
                        row.confidence,
                        row.shirt_color,
                        row.shirt_number,
                        row.speed,
                        row.distance,
                        row.acceleration,
                        row.timestamp,
                        row.homography_results,
                    ]
                )

        logfire.info(f"[DetectionsReporter] Report generated: {report_path.as_posix()}")

    def get_homography_values(self, match_id: int, session: Session):
        homographies = HomographyRepository.get_homographies_by_match_id(
            match_id, session
        )
        report_rows: List[ReportRow] = []

        for homography in homographies:
            report_rows.append(
                ReportRow(
                    frame_number=homography.frame_num,
                    object_type="homography",
                    track_id=0,
                    bbox="",
                    confidence=homography.confidence,
                    timestamp=0,
                    homography_results=homography.H_json,
                )
            )

        return report_rows

    def get_balls(self, match_id: int, session: Session) -> list[ReportRow]:
        balls = BallRepository.get_balls_by_match_id(match_id, session)
        report_rows: List[ReportRow] = []

        for ball in balls:
            bbox = f"{ball.x1}, {ball.y1}, {ball.x2}, {ball.y2}"
            report_rows.append(
                ReportRow(
                    frame_number=ball.frame_number,
                    object_type="ball",
                    track_id=0,
                    bbox=bbox,
                    confidence=ball.confidence,
                    timestamp=ball.timestamp,
                )
            )

        return report_rows

    def get_players(self, match_id: int, session: Session) -> list[ReportRow]:
        players = PlayerRepository.get_players_by_match_id(match_id, session)

        report_rows: List[ReportRow] = []

        for player in players:
            states = player.states
            for state in states:
                bbox = f"{state.x1}, {state.y1}, {state.x2}, {state.y2}"
                report_rows.append(
                    ReportRow(
                        frame_number=state.frame_number,
                        object_type="player",
                        track_id=player.track_id,
                        bbox=bbox,
                        confidence=state.confidence,
                        speed=state.speed_kmh,
                        distance=state.distance_meters,
                        acceleration=state.acceleration,
                        shirt_color=player.team_color,
                        shirt_number=str(
                            player.shirt_number
                            if player.shirt_number is not None
                            else ""
                        ),
                        timestamp=state.timestamp,
                        dx_meters=str(state.dx_meters),
                        dy_meters=str(state.dy_meters),
                    )
                )

        return report_rows

    def get_goals(self, match_id: int, session: Session) -> list[ReportRow]:
        goals = GoalRepository.get_goals_by_match_id(match_id, session)

        report_rows: List[ReportRow] = []

        for goal in goals:
            bbox = f"{goal.x1}, {goal.y1}, {goal.x2}, {goal.y2}"
            report_rows.append(
                ReportRow(
                    frame_number=goal.frame_number,
                    object_type="goal",
                    track_id=0,
                    bbox=bbox,
                    confidence=goal.confidence,
                    timestamp=goal.timestamp,
                )
            )

        return report_rows
