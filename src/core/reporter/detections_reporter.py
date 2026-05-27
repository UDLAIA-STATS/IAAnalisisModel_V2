import csv
from pathlib import Path
from typing import List
import uuid

import logfire
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
    timestamp: int


class DetectionsReporter:
    def generate_report(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = []

        report_rows.extend(self.get_goals(match_id, session))
        report_rows.extend(self.get_balls(match_id, session))
        report_rows.extend(self.get_players(match_id, session))

        report_name = (
            DETECTED_OBJECTS_METRICS_DIR / f"report_{match_id}_{uuid.uuid4()}.csv"
        )

        with open(report_name, mode="w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)

            csv_writer.writerow(
                [
                    "frame_number",
                    "object_type",
                    "track_id",
                    "bbox",
                    "confidence",
                    "shirt_color",
                    "shirt_number",
                    "speed",
                    "distance",
                    "timestamp",
                ]
            )

            for row in report_rows:
                csv_writer.writerow(
                    [
                        row.frame_number,
                        row.object_type,
                        row.track_id,
                        row.bbox,
                        row.confidence,
                        row.shirt_color,
                        row.shirt_number,
                        row.speed,
                        row.distance,
                        row.timestamp,
                    ]
                )

        logfire.info(f"[DetectionsReporter] Report generated: {report_name.as_posix()}")
        self.generate_diagrams(report_name)

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
                        shirt_color=player.team_color,
                        shirt_number=str(player.shirt_number),
                        timestamp=state.timestamp,
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

    def generate_diagrams(self, report_path: Path):
        """
        Generates:
        1. Recognition frequency per player
        2. Average confidence per player
        3. Tracking persistence
        """

        df = pd.read_csv(report_path)

        # Keep only player detections
        players_df = df[df["object_type"] == "player"]

        if players_df.empty:
            logfire.warning("[DetectionsReporter] No player data found")
            return

        diagrams_dir = report_path.parent / "diagrams"
        diagrams_dir.mkdir(exist_ok=True)

        # =========================================================
        # 1. PLAYER RECOGNITION COUNT
        # =========================================================

        recognition_counts = (
            players_df.groupby("shirt_number").size().sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))

        recognition_counts.plot(kind="bar")

        plt.title("Player Recognition Frequency")
        plt.xlabel("Shirt Number")
        plt.ylabel("Recognition Count")

        plt.tight_layout()

        recognition_chart = (
            diagrams_dir / f"recognition_frequency_{report_path.stem}.png"
        )

        plt.savefig(recognition_chart)
        plt.close()

        # =========================================================
        # 2. AVERAGE CONFIDENCE
        # =========================================================

        avg_confidence = (
            players_df.groupby("shirt_number")["confidence"]
            .mean()
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))

        avg_confidence.plot(kind="bar")

        plt.title("Average Recognition Confidence")
        plt.xlabel("Shirt Number")
        plt.ylabel("Average Confidence")

        plt.tight_layout()

        confidence_chart = diagrams_dir / f"average_confidence_{report_path.stem}.png"

        plt.savefig(confidence_chart)
        plt.close()

        # =========================================================
        # 3. TRACKING PERSISTENCE
        # =========================================================

        persistence = (
            players_df.groupby("shirt_number")["frame_number"]
            .agg(lambda x: x.max() - x.min())
            .sort_values(ascending=False)
        )

        plt.figure(figsize=(10, 6))

        persistence.plot(kind="bar")

        plt.title("Tracking Persistence")
        plt.xlabel("Shirt Number")
        plt.ylabel("Frames Persisted")

        plt.tight_layout()

        persistence_chart = (
            diagrams_dir / f"tracking_persistence_{report_path.stem}.png"
        )


reporter = DetectionsReporter()
