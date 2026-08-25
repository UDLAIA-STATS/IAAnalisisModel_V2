from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import logfire
import numpy as np
import pandas as pd

from src.entities.reporter.diagrams_generator_base import DiagramsGeneratorBase


class MatchSpatialAnalyzerBase(DiagramsGeneratorBase, ABC):
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
    

    def _clip_polygon_to_bounds(self, polygon: list, bounds: dict) -> list:
        """
        Clip a polygon to a bounding box using Sutherland-Hodgman algorithm.

        Args:
            polygon: List of [x, y] vertices
            bounds: Dict with x_min, x_max, y_min, y_max

        Returns:
            Clipped polygon vertices, or empty list if fully outside
        """

        def inside(p, edge):
            if edge == "left":
                return p[0] >= bounds["x_min"]
            if edge == "right":
                return p[0] <= bounds["x_max"]
            if edge == "bottom":
                return p[1] >= bounds["y_min"]
            if edge == "top":
                return p[1] <= bounds["y_max"]
            return False

        def intersection(s, e, edge):
            x1, y1 = s
            x2, y2 = e

            if edge == "left":
                x = bounds["x_min"]
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 != x1 else y1
            elif edge == "right":
                x = bounds["x_max"]
                y = y1 + (y2 - y1) * (x - x1) / (x2 - x1) if x2 != x1 else y1
            elif edge == "bottom":
                y = bounds["y_min"]
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1) if y2 != y1 else x1
            elif edge == "top":
                y = bounds["y_max"]
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1) if y2 != y1 else x1
            else:
                return [x1, y1]

            return [x, y]

        output = polygon
        for edge in ["left", "right", "bottom", "top"]:
            input_list = output
            output = []
            if not input_list:
                break

            s = input_list[-1]
            for e in input_list:
                if inside(e, edge):
                    if not inside(s, edge):
                        output.append(intersection(s, e, edge))
                    output.append(e)
                elif inside(s, edge):
                    output.append(intersection(s, e, edge))
                s = e

        return output

    def _clip_line_to_bounds(
        self, v0: np.ndarray, v1: np.ndarray, bounds: dict
    ) -> list | None:
        """
        Clip a line segment to a bounding box using Liang-Barsky algorithm.

        Returns:
            [[x0, y0], [x1, y1]] if segment intersects bounds, None otherwise
        """
        x0, y0 = v0[0], v0[1]
        x1, y1 = v1[0], v1[1]

        dx = x1 - x0
        dy = y1 - y0

        p = [-dx, dx, -dy, dy]
        q = [
            x0 - bounds["x_min"],
            bounds["x_max"] - x0,
            y0 - bounds["y_min"],
            bounds["y_max"] - y0,
        ]

        u1, u2 = 0.0, 1.0

        for i in range(4):
            if p[i] == 0:
                if q[i] < 0:
                    return None
            else:
                t = q[i] / p[i]
                if p[i] < 0:
                    u1 = max(u1, t)
                else:
                    u2 = min(u2, t)

        if u1 > u2:
            return None

        return [
            [x0 + u1 * dx, y0 + u1 * dy],
            [x0 + u2 * dx, y0 + u2 * dy],
        ]

    def _parse_rgb_color(self, color_str: str | None) -> str:
        """
        Parse RGB color string "R, G, B" to matplotlib-compatible color.

        Supports:
        - "R, G, B" (comma-separated integers 0-255)
        - "R, G, B, A" (with alpha, alpha is ignored)
        - Existing hex colors (passthrough)
        - Named colors (passthrough)
        - None/NaN -> returns "#95A5A6" (grey)
        """
        if pd.isna(color_str) or color_str is None:
            return "#95A5A6"  # grey for unknown

        color_str = str(color_str).strip()

        if color_str.startswith("#"):
            return color_str

        if "," not in color_str and color_str.isalpha():
            return color_str

        try:
            parts = [int(x.strip()) for x in color_str.split(",")]
            if len(parts) >= 3:
                r, g, b = parts[0], parts[1], parts[2]

                r = max(0, min(255, r))
                g = max(0, min(255, g))
                b = max(0, min(255, b))
                return f"#{r:02x}{g:02x}{b:02x}"
        except (ValueError, IndexError):
            pass

        logfire.warning(
            f"[MatchSpatialAnalyzer] Could not parse color: {color_str}, using grey"
        )
        return "#95A5A6"
