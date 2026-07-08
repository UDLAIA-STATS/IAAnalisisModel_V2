import csv
import uuid

import logfire
from sqlmodel import Session, col, select

from src.config.routes import DETECTED_OBJECTS_METRICS_DIR
from src.entities.models.soccer.player_model import PlayerModel, PlayerNumbers


class NumberReporter:
    def generate_number_report(self, match_id: int, session: Session) -> str:
        numbers = session.exec(
            select(PlayerNumbers)
            .join(PlayerModel, col(PlayerModel.id) == PlayerNumbers.player_id)
            .where(PlayerModel.match_id == match_id)
        ).all()
        logfire.info(f"[NumberReporter] Generated report for {len(numbers)} numbers")
        report_path = (
            DETECTED_OBJECTS_METRICS_DIR
            / f"report_numbers_{match_id}_{uuid.uuid4()}.csv"
        )

        with open(report_path, mode="w", newline="", encoding="utf-8") as file:
            csv_writer = csv.writer(file)

            csv_writer.writerow(
                [
                    "id",
                    "track_id",
                    "number",
                    "confidence",
                    "visible_prob",
                    "tens_pred",
                    "tens_prob",
                    "tens_none_prob",
                    "units_pred",
                    "units_prob",
                ]
            )

            for row in numbers:
                csv_writer.writerow(
                    [
                        row.id,
                        row.player.track_id,
                        row.number,
                        row.confidence,
                        row.visible_prob,
                        row.tens_pred,
                        row.tens_prob,
                        row.tens_none_prob,
                        row.units_pred,
                        row.units_prob,
                    ]
                )

        return report_path.as_posix()
