from typing import List
import uuid


from sqlmodel import Session

from src.entities.models.app.report_models import ReportRow
from src.entities.reporter.detections_reporter_base import DetectionsReporterBase
from src.config.routes import DETECTED_OBJECTS_METRICS_DIR


class DetectionsReporter(DetectionsReporterBase):

    def generate_report(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = self.get_report_rows(match_id, session)

        report_path = (
            DETECTED_OBJECTS_METRICS_DIR
            / f"report_{match_id}_{uuid.uuid4()}.csv"
        )

        self.write_detections_report(report_path, report_rows)

        class_chart_str, time_chart_str, dynamics_chart_str, heatmap_chart_str = (
            self.generate_diagrams(report_path, match_id)
        )


        speed_chart, distance_chart = self.generate_per_player_timeseries(
            report_path, match_id
        )

        reports = [
            ("report_detections", report_path.as_posix()),
            ("class_chart", class_chart_str),
            ("time_chart", time_chart_str),
            ("dynamics_chart", dynamics_chart_str),
            ("heatmap_chart", heatmap_chart_str),
            ("speed_chart", speed_chart),
            ("distance_chart", distance_chart),
        ]
        chart_keys = self.upload_reports(reports, match_id)

        return (
            chart_keys["report_detections"],
            chart_keys["class_chart"],
            chart_keys["time_chart"],
            chart_keys["dynamics_chart"],
            chart_keys["heatmap_chart"],
            chart_keys["speed_chart"],
            chart_keys["distance_chart"],
        )


reporter = DetectionsReporter()
