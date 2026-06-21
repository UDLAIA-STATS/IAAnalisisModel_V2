from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

from src.entities.reporter.diagrams_generator import DiagramsGenerator


class MatchSpatialAnalyzerBase(DiagramsGenerator, ABC):
    """
    Abstract base for spatial and tactical analysis diagrams.
    Inherits from DiagramsGenerator for shared plotting utilities.
    """

    _FIELD_COLOR = "#4a7c2f"
    _VORONOI_LINE_COLOR = "orange"
    _VORONOI_LINE_ALPHA = 0.6
    _VORONOI_LINE_WIDTH = 2

    @abstractmethod
    def generate_spatial_diagrams(
        self, report_path: Path, match_id: int
    ) -> Tuple[str, str, str, str, str]:
        """
        Returns paths for:
        (position_velocity_scatter, voronoi_territories, velocity_kde_by_team,
         team_distance_matrix, movement_trajectories)
        """
        raise NotImplementedError