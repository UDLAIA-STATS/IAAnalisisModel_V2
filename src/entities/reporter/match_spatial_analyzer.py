from pathlib import Path
from typing import List, Tuple

import logfire
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi
from scipy.spatial.distance import pdist
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sqlmodel import Session, col, select

from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.reporter.match_spatial_analyzer_base import MatchSpatialAnalyzerBase
from src.config.routes import DIAGRAMS_DIR, OUTPUT_DIAGRAMS

from .reporter_utils import group_states_by_id

class MatchSpatialAnalyzer(MatchSpatialAnalyzerBase):
    """
    Generates spatial and tactical analysis diagrams for soccer match tracking.
    Complements DiagramsGenerator with team-level and spatial analysis.

    team_color must be in CSV as "R, G, B" (comma-separated RGB values).
        This class parses RGB strings to matplotlib-compatible colors.
    """

    def generate_spatial_diagrams(
        self, report_path: Path, match_id: int
    ) -> Tuple[str, str, str, str, str]:
        """
        Generates 5 spatial analysis diagrams from the detections report CSV:
        1. Position scatter with velocity encoding (icefire palette)
        2. Voronoi territorial tessellation (orange lines)
        3. Velocity KDE fill plot by team color
        4. Team distance matrix heatmap
        5. Movement trajectories per track_id

        NOTE: Uses dx_meters and dy_meters for spatial diagrams (converted to meters),
              with bbox center_x/center_y as fallback for velocity scatter hue/size.
        """
        df = pd.read_csv(report_path)
        parent_dir = DIAGRAMS_DIR / str(match_id)
        parent_dir.mkdir(exist_ok=True, parents=True)
        stem = report_path.stem
        self.player_heatmaps_dir = OUTPUT_DIAGRAMS / str(match_id)
        self.player_heatmaps_dir.mkdir(exist_ok=True, parents=True)

        if df.empty:
            logfire.warning(
                "[MatchSpatialAnalyzer] Report is empty, skipping spatial diagrams"
            )
            return "", "", "", "", ""

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning(
                "[MatchSpatialAnalyzer] No player data for spatial analysis"
            )
            return "", "", "", "", ""

        players_df["parsed_team_color"] = players_df["shirt_color"].apply(
            self._parse_rgb_color
        )

        bbox_values = players_df["bbox"].str.split(", ", expand=True).astype(float)
        players_df["center_x"] = (bbox_values[0] + bbox_values[2]) / 2
        players_df["center_y"] = (bbox_values[1] + bbox_values[3]) / 2

        players_df["dx_meters"] = pd.to_numeric(
            players_df["dx_meters"], errors="coerce"
        )
        players_df["dy_meters"] = pd.to_numeric(
            players_df["dy_meters"], errors="coerce"
        )

        players_df["speed"] = pd.to_numeric(players_df["speed"], errors="coerce")
        players_df["frame_number"] = pd.to_numeric(
            players_df["frame_number"], errors="coerce"
        )

        scatter_path = self._generate_position_velocity_scatter(
            players_df, parent_dir, stem
        )
        voronoi_path = self._generate_voronoi_territories(players_df, parent_dir, stem)
        kde_path = self._generate_velocity_kde_by_team(players_df, parent_dir, stem)
        dist_matrix_path = self._generate_team_distance_matrix(
            players_df, parent_dir, stem
        )
        traj_path = self._generate_movement_trajectories(players_df, parent_dir, stem)

        return (
            scatter_path.as_posix() if scatter_path else "",
            voronoi_path.as_posix() if voronoi_path else "",
            kde_path.as_posix() if kde_path else "",
            dist_matrix_path.as_posix() if dist_matrix_path else "",
            traj_path.as_posix() if traj_path else "",
        )

    def _generate_position_velocity_scatter(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """Position scatter with velocity as hue and size (icefire palette).

        Uses bbox center_x/center_y (pixel coordinates) for position,
        speed for hue and size encoding.
        """
        try:
            sns.set_context("poster")
            fig, ax = plt.subplots(figsize=(16, 12))

            sns.scatterplot(
                data=players_df,
                x="center_x",
                y="center_y",
                hue="speed",
                size="speed",
                alpha=0.7,
                legend=False,
                sizes=(1, 100),
                palette=sns.color_palette("icefire", as_cmap=True),
                ax=ax,
            )

            ax.set_facecolor(self._FIELD_COLOR)
            ax.set_xlim(
                players_df["center_x"].min() - 10,
                players_df["center_x"].max() + 10,
            )
            ax.set_ylim(
                players_df["center_y"].max() + 10,
                players_df["center_y"].min() - 10,
            )
            ax.axis("off")

            plt.tight_layout()

            out_path = parent_dir / f"position_velocity_scatter_{stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logfire.info(f"[MatchSpatialAnalyzer] Saved {out_path.as_posix()}")
            return out_path

        except Exception as e:
            logfire.error(f"[MatchSpatialAnalyzer] Error in position scatter: {e}")
            return None

    def _generate_voronoi_territories(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """Voronoi territorial diagram with team-colored points.

        Uses dx_meters and dy_meters for meter-scale spatial analysis.
        For display, negative coordinates are shifted into a non-negative frame
        so the exported visualization does not expose negative distances.
        Voronoi regions are clipped to the field bounding box — no infinite
        extensions beyond the field limits.
        """
        try:
            valid_players = players_df.dropna(subset=["dx_meters", "dy_meters"]).copy()
            points = valid_players[["dx_meters", "dy_meters"]].values

            if len(points) < 3:
                logfire.warning("[MatchSpatialAnalyzer] Not enough points for Voronoi")
                return None

            x_shift = max(0.0, -float(valid_players["dx_meters"].min()))
            y_shift = max(0.0, -float(valid_players["dy_meters"].min()))
            valid_players["display_x"] = valid_players["dx_meters"] + x_shift
            valid_players["display_y"] = valid_players["dy_meters"] + y_shift
            display_points = valid_players[["display_x", "display_y"]].values

            x_min, x_max = (
                valid_players["display_x"].min(),
                valid_players["display_x"].max(),
            )
            y_min, y_max = (
                valid_players["display_y"].min(),
                valid_players["display_y"].max(),
            )

            x_pad = max((x_max - x_min) * 0.05, 1)
            y_pad = max((y_max - y_min) * 0.05, 1)

            bounds = {
                "x_min": x_min - x_pad,
                "x_max": x_max + x_pad,
                "y_min": y_min - y_pad,
                "y_max": y_max + y_pad,
            }

            margin = max(x_max - x_min, y_max - y_min) * 2
            boundary_points = np.array(
                [
                    [x_min - margin, y_min - margin],
                    [x_max + margin, y_min - margin],
                    [x_max + margin, y_max + margin],
                    [x_min - margin, y_max + margin],
                ]
            )
            points_with_boundary = np.vstack([display_points, boundary_points])

            vor = Voronoi(points_with_boundary)

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_facecolor(self._FIELD_COLOR)

            regions = []
            region_colors = []

            for point_idx, region_idx in enumerate(
                vor.point_region[: len(display_points)]
            ):
                region = vor.regions[region_idx]
                if not region or -1 in region:
                    continue

                polygon = [vor.vertices[i] for i in region]

                clipped = self._clip_polygon_to_bounds(polygon, bounds)
                if not clipped or len(clipped) < 3:
                    continue

                team_color = (
                    valid_players.iloc[point_idx]["parsed_team_color"]
                    if point_idx < len(valid_players)
                    else "#95A5A6"
                )

                regions.append(Polygon(clipped, closed=True))
                region_colors.append(team_color)

            if regions:
                patch_collection = PatchCollection(
                    regions,
                    facecolors=region_colors,
                    edgecolors="white",
                    linewidths=0.3,
                    alpha=0.3,
                )
                ax.add_collection(patch_collection)

            for ridge in vor.ridge_vertices:
                if -1 in ridge:
                    continue
                v0, v1 = vor.vertices[ridge[0]], vor.vertices[ridge[1]]

                clipped_line = self._clip_line_to_bounds(v0, v1, bounds)
                if clipped_line:
                    ax.plot(
                        [clipped_line[0][0], clipped_line[1][0]],
                        [clipped_line[0][1], clipped_line[1][1]],
                        color=self._VORONOI_LINE_COLOR,
                        linewidth=self._VORONOI_LINE_WIDTH,
                        alpha=self._VORONOI_LINE_ALPHA,
                    )

            for color in valid_players["parsed_team_color"].dropna().unique():
                team_data = valid_players[valid_players["parsed_team_color"] == color]
                ax.scatter(
                    team_data["display_x"],
                    team_data["display_y"],
                    c=color,
                    s=15,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=5,
                )

            ax.set_xlim(bounds["x_min"], bounds["x_max"])
            ax.set_ylim(bounds["y_min"], bounds["y_max"])
            ax.set_xlabel("X Position", fontsize=12)
            ax.set_ylabel("Y Position", fontsize=12)
            ax.set_title("Voronoi Territories", fontsize=14)

            plt.tight_layout()

            out_path = parent_dir / f"voronoi_territories_{stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logfire.info(f"[MatchSpatialAnalyzer] Saved {out_path.as_posix()}")
            return out_path

        except Exception as e:
            logfire.error(f"[MatchSpatialAnalyzer] Error in Voronoi: {e}")
            return None

    def _generate_velocity_kde_by_team(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """Filled KDE plot of velocity distribution by team color."""
        try:
            unique_colors = players_df["parsed_team_color"].dropna().unique()
            if len(unique_colors) == 0:
                logfire.warning(
                    "[MatchSpatialAnalyzer] No team colors for velocity KDE"
                )
                return None

            color_palette = {c: c for c in unique_colors}

            fig, ax = plt.subplots(figsize=(8, 6))

            sns.kdeplot(
                data=players_df,
                x="speed",
                hue="parsed_team_color",
                multiple="fill",
                common_norm=False,
                ax=ax,
                palette=color_palette,
                alpha=0.7,
                cut=0,
            )

            ax.set_xlabel("Velocity (km/h)", fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_title("Velocity Distribution by Team", fontsize=14)

            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

            plt.tight_layout()

            out_path = parent_dir / f"velocity_kde_by_team_{stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logfire.info(f"[MatchSpatialAnalyzer] Saved {out_path.as_posix()}")
            return out_path

        except Exception as e:
            logfire.error(f"[MatchSpatialAnalyzer] Error in velocity KDE: {e}")
            return None

    def _generate_team_distance_matrix(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """
        Heatmap showing:
        - Diagonal: intra-team average pairwise distance (compactness)
        - Off-diagonal: inter-team centroid distance (separation)

        Uses dx_meters and dy_meters for meter-scale distances.
        """
        try:
            teams = players_df["parsed_team_color"].dropna().unique()
            if len(teams) < 1:
                return None

            team_stats = {}
            for team in teams:
                team_data = players_df[players_df["parsed_team_color"] == team]
                points = team_data[["dx_meters", "dy_meters"]].dropna().values

                centroid = points.mean(axis=0) if len(points) > 0 else np.array([0, 0])
                intra_dist = 0.0
                if len(points) > 1:
                    intra_dist = pdist(points).mean()

                team_stats[team] = {
                    "centroid": centroid,
                    "intra_dist": intra_dist,
                    "count": len(points),
                }

            n_teams = len(teams)
            dist_matrix = np.zeros((n_teams, n_teams))
            team_names = list(team_stats.keys())

            for i, t1 in enumerate(team_names):
                for j, t2 in enumerate(team_names):
                    if i == j:
                        dist_matrix[i, j] = team_stats[t1]["intra_dist"]
                    else:
                        p1 = team_stats[t1]["centroid"]
                        p2 = team_stats[t2]["centroid"]
                        dist_matrix[i, j] = np.linalg.norm(p1 - p2)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(dist_matrix, cmap="YlOrRd", aspect="auto")

            ax.set_xticks(range(n_teams))
            ax.set_yticks(range(n_teams))
            ax.set_xticklabels(
                [f"Team {t}" for t in team_names], rotation=45, ha="right"
            )
            ax.set_yticklabels([f"Team {t}" for t in team_names])

            for i in range(n_teams):
                for j in range(n_teams):
                    val = dist_matrix[i, j]
                    color = "white" if val > dist_matrix.max() * 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.1f}m",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=10,
                    )

            plt.colorbar(im, ax=ax, label="Distance (meters)")
            ax.set_title(
                "Team Distance Matrix (Meters)\n"
                "(Diagonal = Intra-team spread, Off-diagonal = Centroid separation)",
                fontsize=12,
            )

            plt.tight_layout()

            out_path = parent_dir / f"team_distance_matrix_{stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logfire.info(f"[MatchSpatialAnalyzer] Saved {out_path.as_posix()}")
            return out_path

        except Exception as e:
            logfire.error(f"[MatchSpatialAnalyzer] Error in distance matrix: {e}")
            return None

    def _generate_movement_trajectories(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """Connected movement paths per track_id with start/end markers.

        Uses dx_meters and dy_meters for meter-scale trajectories.
        For display, negative coordinates are shifted into a non-negative frame
        so the exported visualization does not expose negative distances.
        """
        try:
            valid_players = players_df.dropna(subset=["dx_meters", "dy_meters"]).copy()
            if valid_players.empty:
                return None

            x_shift = max(0.0, -float(valid_players["dx_meters"].min()))
            y_shift = max(0.0, -float(valid_players["dy_meters"].min()))
            valid_players["display_x"] = valid_players["dx_meters"] + x_shift
            valid_players["display_y"] = valid_players["dy_meters"] + y_shift

            valid_players = valid_players.sort_values(["track_id", "frame_number"])

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_facecolor(self._FIELD_COLOR)

            track_ids = sorted(valid_players["track_id"].unique())
            cmap = plt.get_cmap("tab20")
            fallback_colors = [cmap(i % 20) for i in range(len(track_ids))]

            for idx, track_id in enumerate(track_ids):
                track_data = valid_players[valid_players["track_id"] == track_id]
                if len(track_data) < 2:
                    continue

                team_color = None
                valid_colors = track_data["parsed_team_color"].dropna()
                if len(valid_colors) > 0:
                    team_color = valid_colors.iloc[0]

                color = team_color if team_color else fallback_colors[idx]

                ax.plot(
                    track_data["display_x"],
                    track_data["display_y"],
                    color=color,
                    alpha=0.5,
                    linewidth=1.2,
                )

                ax.scatter(
                    track_data["display_x"].iloc[0],
                    track_data["display_y"].iloc[0],
                    c="lime",
                    s=40,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )

                ax.scatter(
                    track_data["display_x"].iloc[-1],
                    track_data["display_y"].iloc[-1],
                    c="red",
                    s=60,
                    marker="X",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )

            ax.set_xlim(
                valid_players["display_x"].min() - 5,
                valid_players["display_x"].max() + 5,
            )
            ax.set_ylim(
                valid_players["display_y"].max() + 5,
                valid_players["display_y"].min() - 5,
            )
            ax.set_title("Player Movement Trajectories", fontsize=14)
            ax.set_xlabel("X Position", fontsize=11)
            ax.set_ylabel("Y Position", fontsize=11)

            plt.tight_layout()

            out_path = parent_dir / f"movement_trajectories_{stem}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            logfire.info(f"[MatchSpatialAnalyzer] Saved {out_path.as_posix()}")
            return out_path

        except Exception as e:
            logfire.error(f"[MatchSpatialAnalyzer] Error in trajectories: {e}")
            return None

    def generate_per_player_heatmaps(
        self, report_path: Path, match_id: int, session: Session
    ) -> List[Tuple[int, Path]]:
        """
        Generates one heatmap per player using DB states (primary source)
        and CSV as fallback enrichment.
        """

        df = pd.read_csv(report_path)
        stem = report_path.stem
        output_dir = self.player_heatmaps_dir

        if df.empty:
            logfire.warning("[MatchSpatialAnalyzer] Empty report CSV")
            return []

        state_groups = group_states_by_id(match_id, session)

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty and not state_groups:
            logfire.warning("[MatchSpatialAnalyzer] No player data at all")
            return []

        players_df["parsed_team_color"] = players_df["shirt_color"].apply(
            self._parse_rgb_color
        )

        players_df["id"] = pd.to_numeric(players_df["id"], errors="coerce")

        saved_paths: List[Tuple[int, Path]] = []

        for player_id, states in state_groups.items():

            if not states or len(states) < 3:
                logfire.warning(
                    f"[MatchSpatialAnalyzer] Not enough DB states for player_id={player_id}"
                )
                continue

            dx = []
            dy = []

            for s in states:
                if s.dx_meters is None or s.dy_meters is None:
                    dx.append(s.dx)
                    dy.append(s.dy)
                else:
                    dx.append(s.dx_meters)
                    dy.append(s.dy_meters)


            if len(dx) < 3:
                logfire.warning(
                    f"[MatchSpatialAnalyzer] Not enough valid coordinates for player_id={player_id}"
                )
                continue

            csv_row = players_df[players_df["id"] == player_id]

            if not csv_row.empty:
                team_color = csv_row["parsed_team_color"].dropna().iloc[0] \
                    if not csv_row["parsed_team_color"].dropna().empty else "#4a7c2f"

                track_id = csv_row["track_id"].iloc[0] if "track_id" in csv_row.columns else "N/A"
            else:
                team_color = "#4a7c2f"
                track_id = "N/A"


            try:
                fig, ax = plt.subplots(figsize=(10, 10))

                fig.patch.set_facecolor("#4a7c2f")
                ax.set_facecolor(team_color)
                ax.patch.set_alpha(0.15)

                dx_arr = pd.Series(dx)
                dy_arr = pd.Series(dy)

                x_min, x_max = dx_arr.min(), dx_arr.max()
                y_min, y_max = dy_arr.min(), dy_arr.max()

                x_pad = max((x_max - x_min) * 0.1, 2)
                y_pad = max((y_max - y_min) * 0.1, 2)

                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

                hb = ax.hexbin(
                    dx_arr,
                    dy_arr,
                    gridsize=20,
                    cmap="YlOrRd",
                    alpha=0.85,
                    mincnt=1,
                )

                cbar = plt.colorbar(hb, ax=ax)
                cbar.set_label("Density")

                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.invert_yaxis()

                ax.set_title(
                    f"Player Heatmap — ID: {player_id} | Track: {track_id}\n"
                    f"Frames: {len(dx)}",
                    fontsize=12,
                )

                plt.tight_layout()

                out_path = output_dir / f"{player_id}_heatmap_{stem}.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append((player_id, out_path))

                logfire.info(
                    f"[MatchSpatialAnalyzer] Saved heatmap: {out_path.as_posix()}"
                )

            except Exception as e:
                logfire.error(
                    f"[MatchSpatialAnalyzer] Heatmap error player_id={player_id}: {e}"
                )
                continue

        logfire.info(
            f"[MatchSpatialAnalyzer] Generated {len(saved_paths)} heatmaps"
        )

        return saved_paths
