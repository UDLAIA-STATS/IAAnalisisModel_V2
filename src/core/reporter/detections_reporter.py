import csv
from typing import List
import uuid

from pydantic import BaseModel
from sqlmodel import Session

from src.config.routes import DETECTED_OBJECTS_METRICS_DIR
from src.core.repository.ball_repository import BallRepository
from src.core.repository.goal_repository import GoalRepository
from src.core.repository.player_repository import PlayerRepository

class ReportRow(BaseModel):
    frame_number: int
    object_type: str
    track_id: int
    bbox: str
    confidence: float
    shirt_color: str = ""
    shirt_number: str = ""
    speed: float = 0.0
    distance: float = 0.0
    timestamp_ms: int

class DetectionsReporter:

    def generate_report(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = []

        report_rows.extend(self.get_goals(match_id, session))
        report_rows.extend(self.get_balls(match_id, session))
        report_rows.extend(self.get_players(match_id, session))

        report_name = DETECTED_OBJECTS_METRICS_DIR / f"report_{match_id}_{uuid.uuid4()}.csv"

        with open(report_name, mode="w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)

            csv_writer.writerow([
                "frame_number", "object_type", "track_id", 
                "bbox", "confidence", "shirt_color", "shirt_number",
                "speed", "distance", "timestamp_ms"])

            for row in report_rows:
                csv_writer.writerow([
                    row.frame_number, row.object_type, row.track_id, 
                    row.bbox, row.confidence, row.shirt_color, row.shirt_number,
                    row.speed, row.distance, row.timestamp_ms
                ])

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
                    timestamp_ms=ball.timestamp_ms,
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
                        shirt_color=player.team_color,
                        shirt_number=str(player.shirt_number),
                        timestamp_ms=state.timestamp_ms,
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
                    timestamp_ms=goal.timestamp_ms,
                )
            )

        return report_rows

reporter = DetectionsReporter()