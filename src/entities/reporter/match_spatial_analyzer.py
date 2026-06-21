from pathlib import Path
from typing import List, Tuple

import logfire
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import pdist

from src.entities.reporter.match_spatial_analyzer_base import MatchSpatialAnalyzerBase
from src.config.routes import DIAGRAMS_DIR, OUTPUT_DIAGRAMS

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
            logfire.warning("[MatchSpatialAnalyzer] Report is empty, skipping spatial diagrams")
            return "", "", "", "", ""

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning("[MatchSpatialAnalyzer] No player data for spatial analysis")
            return "", "", "", "", ""

        players_df["parsed_team_color"] = players_df["shirt_color"].apply(
            self._parse_rgb_color
        )

        bbox_values = players_df["bbox"].str.split(", ", expand=True).astype(float)
        players_df["center_x"] = (bbox_values[0] + bbox_values[2]) / 2
        players_df["center_y"] = (bbox_values[1] + bbox_values[3]) / 2

        players_df["dx_meters"] = pd.to_numeric(players_df["dx_meters"], errors="coerce")
        players_df["dy_meters"] = pd.to_numeric(players_df["dy_meters"], errors="coerce")

        players_df["speed"] = pd.to_numeric(players_df["speed"], errors="coerce")
        players_df["frame_number"] = pd.to_numeric(players_df["frame_number"], errors="coerce")

        scatter_path = self._generate_position_velocity_scatter(players_df, parent_dir, stem)

        voronoi_path = self._generate_voronoi_territories(players_df, parent_dir, stem)

        kde_path = self._generate_velocity_kde_by_team(players_df, parent_dir, stem)

        dist_matrix_path = self._generate_team_distance_matrix(players_df, parent_dir, stem)

        traj_path = self._generate_movement_trajectories(players_df, parent_dir, stem)

        return (
            scatter_path.as_posix() if scatter_path else "",
            voronoi_path.as_posix() if voronoi_path else "",
            kde_path.as_posix() if kde_path else "",
            dist_matrix_path.as_posix() if dist_matrix_path else "",
            traj_path.as_posix() if traj_path else "",
        )

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

        logfire.warning(f"[MatchSpatialAnalyzer] Could not parse color: {color_str}, using grey")
        return "#95A5A6"

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
        """
        try:
            points = players_df[["dx_meters", "dy_meters"]].dropna().values

            if len(points) < 3:
                logfire.warning("[MatchSpatialAnalyzer] Not enough points for Voronoi")
                return None

            x_min, x_max = players_df["dx_meters"].min(), players_df["dx_meters"].max()
            y_min, y_max = players_df["dy_meters"].min(), players_df["dy_meters"].max()
            margin = max(x_max - x_min, y_max - y_min) * 2

            boundary_points = np.array([
                [x_min - margin, y_min - margin],
                [x_max + margin, y_min - margin],
                [x_max + margin, y_max + margin],
                [x_min - margin, y_max + margin],
            ])
            points_with_boundary = np.vstack([points, boundary_points])

            vor = Voronoi(points_with_boundary)

            fig, ax = plt.subplots(figsize=(16, 8))
            ax.set_facecolor(self._FIELD_COLOR)

            voronoi_plot_2d(
                vor,
                show_vertices=False,
                line_colors=self._VORONOI_LINE_COLOR,
                line_width=self._VORONOI_LINE_WIDTH,
                line_alpha=self._VORONOI_LINE_ALPHA,
                point_size=2,
                ax=ax,
            )

            for color in players_df["parsed_team_color"].dropna().unique():
                team_data = players_df[players_df["parsed_team_color"] == color]
                ax.scatter(
                    team_data["dx_meters"],
                    team_data["dy_meters"],
                    c=color,
                    s=15,
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.3,
                    zorder=5,
                )

            ax.set_xlim(x_min - 5, x_max + 5)
            ax.set_ylim(y_max + 5, y_min - 5)
            ax.set_xlabel("X Position (meters)", fontsize=12)
            ax.set_ylabel("Y Position (meters)", fontsize=12)
            ax.set_title("Voronoi Territories (Meters)", fontsize=14)

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
                logfire.warning("[MatchSpatialAnalyzer] No team colors for velocity KDE")
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
            ax.set_xticklabels([f"Team {t}" for t in team_names], rotation=45, ha="right")
            ax.set_yticklabels([f"Team {t}" for t in team_names])

            for i in range(n_teams):
                for j in range(n_teams):
                    val = dist_matrix[i, j]
                    color = "white" if val > dist_matrix.max() * 0.6 else "black"
                    ax.text(j, i, f"{val:.1f}m", ha="center", va="center", color=color, fontsize=10)

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
        """
        try:
            players_df = players_df.sort_values(["track_id", "frame_number"])

            fig, ax = plt.subplots(figsize=(14, 10))
            ax.set_facecolor(self._FIELD_COLOR)

            track_ids = sorted(players_df["track_id"].unique())
            cmap = plt.cm.get_cmap("tab20")
            fallback_colors = [cmap(i % 20) for i in range(len(track_ids))]

            for idx, track_id in enumerate(track_ids):
                track_data = players_df[players_df["track_id"] == track_id]
                if len(track_data) < 2:
                    continue

                team_color = None
                valid_colors = track_data["parsed_team_color"].dropna()
                if len(valid_colors) > 0:
                    team_color = valid_colors.iloc[0]

                color = team_color if team_color else fallback_colors[idx]

                ax.plot(
                    track_data["dx_meters"],
                    track_data["dy_meters"],
                    color=color,
                    alpha=0.5,
                    linewidth=1.2,
                )

                ax.scatter(
                    track_data["dx_meters"].iloc[0],
                    track_data["dy_meters"].iloc[0],
                    c="lime",
                    s=40,
                    marker="o",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )

                ax.scatter(
                    track_data["dx_meters"].iloc[-1],
                    track_data["dy_meters"].iloc[-1],
                    c="red",
                    s=60,
                    marker="X",
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )

            ax.set_xlim(
                players_df["dx_meters"].min() - 5,
                players_df["dx_meters"].max() + 5,
            )
            ax.set_ylim(
                players_df["dy_meters"].max() + 5,
                players_df["dy_meters"].min() - 5,
            )
            ax.set_title("Player Movement Trajectories (Meters)", fontsize=14)
            ax.set_xlabel("X Position (meters)", fontsize=11)
            ax.set_ylabel("Y Position (meters)", fontsize=11)

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
        self, report_path: Path, match_id: int
    ) -> List[Tuple[int, Path]]:
        """
        Generates one heatmap per player (by database `id`) using dx_meters/dy_meters.

        Heatmaps are saved in the custom player_heatmaps_dir folder:
            <player_heatmaps_dir>/<match_id>/<id>_heatmap_<stem>.png

        Returns:
            List of (player_id, path) tuples.
        """
        df = pd.read_csv(report_path)
        stem = report_path.stem

        output_dir = self.player_heatmaps_dir

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning("[MatchSpatialAnalyzer] No player data for per-player heatmaps")
            return []

        players_df["parsed_team_color"] = players_df["shirt_color"].apply(
            self._parse_rgb_color
        )

        players_df["dx_meters"] = pd.to_numeric(players_df["dx_meters"], errors="coerce")
        players_df["dy_meters"] = pd.to_numeric(players_df["dy_meters"], errors="coerce")
        players_df["id"] = pd.to_numeric(players_df["id"], errors="coerce")

        saved_paths: List[Tuple[int, Path]] = []
        player_ids = sorted(players_df["id"].dropna().unique().astype(int))

        for player_id in player_ids:
            player_data = players_df[players_df["id"] == player_id].copy()
            player_data = player_data.dropna(subset=["dx_meters", "dy_meters"])

            if len(player_data) < 3:
                logfire.warning(
                    f"[MatchSpatialAnalyzer] Not enough data for player_id={player_id}, skipping heatmap"
                )
                continue

            try:
                fig, ax = plt.subplots(figsize=(10, 10))

                team_color = player_data["parsed_team_color"].dropna().iloc[0] if not player_data["parsed_team_color"].dropna().empty else "#4a7c2f"

                fig.patch.set_facecolor("white")
                ax.set_facecolor(team_color)
                ax.patch.set_alpha(0.15)

                x_min, x_max = player_data["dx_meters"].min(), player_data["dx_meters"].max()
                y_min, y_max = player_data["dy_meters"].min(), player_data["dy_meters"].max()
                x_pad = max((x_max - x_min) * 0.1, 2)
                y_pad = max((y_max - y_min) * 0.1, 2)

                ax.set_xlim(x_min - x_pad, x_max + x_pad)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)

                hb = ax.hexbin(
                    player_data["dx_meters"],
                    player_data["dy_meters"],
                    gridsize=20,
                    cmap="YlOrRd",
                    alpha=0.85,
                    mincnt=1,
                )

                plt.colorbar(hb, ax=ax, label="Detection count")
                ax.invert_yaxis()

                team_name = player_data["parsed_team_color"].dropna().iloc[0] if not player_data["parsed_team_color"].dropna().empty else "Unknown"
                track_id = player_data["track_id"].iloc[0] if "track_id" in player_data.columns else "N/A"
                ax.set_title(
                    f"Player Heatmap — ID: {player_id} | Track: {track_id}\n"
                    f"Team: {team_name} | Frames: {len(player_data)}",
                    fontsize=12,
                )
                ax.set_xlabel("X Position (meters)", fontsize=10)
                ax.set_ylabel("Y Position (meters)", fontsize=10)

                plt.tight_layout()

                out_path = output_dir / f"{player_id}_heatmap_{stem}.png"
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append((player_id, out_path))
                logfire.info(
                    f"[MatchSpatialAnalyzer] Saved player heatmap: {out_path.as_posix()}"
                )

            except Exception as e:
                logfire.error(
                    f"[MatchSpatialAnalyzer] Error generating heatmap for player_id={player_id}: {e}"
                )
                continue

        logfire.info(
            f"[MatchSpatialAnalyzer] Generated {len(saved_paths)} per-player heatmaps in {output_dir.as_posix()}"
        )
        return saved_paths
