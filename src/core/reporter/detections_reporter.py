from typing import List
import uuid

from sqlmodel import Session

from src.core.repository.player_repository import PlayerRepository
from src.entities.types.bucket_types import FilePurposeTypes
from src.entities.models.app.report_models import ReportRow
from src.entities.reporter.detections_reporter_base import DetectionsReporterBase
from src.entities.reporter.match_spatial_analyzer import MatchSpatialAnalyzer
from src.config.routes import DETECTED_OBJECTS_METRICS_DIR
from src.config.configuration import settings

class DetectionsReporter(DetectionsReporterBase):
    def __init__(self):
        
        super().__init__()
        self.spatial_analyzer = MatchSpatialAnalyzer()

    def generate_report(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = self.get_report_rows(match_id, session)

        report_path = (
            DETECTED_OBJECTS_METRICS_DIR / f"report_{match_id}_{uuid.uuid4()}.csv"
        )

        self.write_detections_report(report_path, report_rows)
        number_report = self.generate_number_report(match_id, session)

        class_chart_str, time_chart_str, dynamics_chart_str, heatmap_chart_str = (
            self.generate_diagrams(report_path, match_id)
        )


        speed_chart, distance_chart = self.generate_per_player_timeseries(
            report_path, match_id
        )

        (
            scatter_chart,
            voronoi_chart,
            velocity_kde_chart,
            dist_matrix_chart,
            traj_chart,
        ) = self.spatial_analyzer.generate_spatial_diagrams(report_path, match_id)
        player_heatmaps = self.spatial_analyzer.generate_per_player_heatmaps(report_path, match_id, session) 
        


        reports = [
            ("report_detections", report_path.as_posix()),
            ("class_chart", class_chart_str),
            ("time_chart", time_chart_str),
            ("dynamics_chart", dynamics_chart_str),
            ("heatmap_chart", heatmap_chart_str),
            ("speed_chart", speed_chart),
            ("distance_chart", distance_chart),
            ("position_velocity_scatter", scatter_chart),
            ("voronoi_territories", voronoi_chart),
            ("velocity_kde_by_team", velocity_kde_chart),
            ("team_distance_matrix", dist_matrix_chart),
            ("movement_trajectories", traj_chart),
            ("number_report", number_report),
        ]



        chart_keys = self.upload_reports(reports, match_id)

        for player_id, heatmap_path in player_heatmaps:
            key = self.upload_report(heatmap_path, match_id, FilePurposeTypes.HEATMAP)
            PlayerRepository.upload_heatmap(player_id, settings.PLAYER_DATA_PUBLIC_URL + "/" + key, session)


        velocity_kde_by_team_path = settings.PLAYER_DATA_PUBLIC_URL + "/" + chart_keys["velocity_kde_by_team"]
        voronoi_territories_path = settings.PLAYER_DATA_PUBLIC_URL + "/" + chart_keys["voronoi_territories"]

        PlayerRepository.upload_match_files(
            settings.PLAYER_DATA_PUBLIC_URL + "/" + chart_keys["heatmap_chart"],
            settings.PLAYER_DATA_PUBLIC_URL + "/" + chart_keys["movement_trajectories"],
            match_id,
            session,
        )

        return (
            chart_keys["report_detections"],
            chart_keys["class_chart"],
            chart_keys["time_chart"],
            chart_keys["dynamics_chart"],
            chart_keys["heatmap_chart"],
            chart_keys["speed_chart"],
            chart_keys["distance_chart"],
            chart_keys["position_velocity_scatter"],
            chart_keys["voronoi_territories"],
            chart_keys["velocity_kde_by_team"],
            chart_keys["team_distance_matrix"],
            chart_keys["movement_trajectories"],
            chart_keys["number_report"],
        )


reporter = DetectionsReporter()
