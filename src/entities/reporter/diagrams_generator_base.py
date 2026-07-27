from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt, ticker
from matplotlib.patches import Arc, Circle, Rectangle

import logfire
import pandas as pd

from src.config.constants import PITCH_WIDTH, PITCH_LENGTH
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.config.routes import DIAGRAMS_DIR


class DiagramsGeneratorBase:
    _TRACK_COLORS = [
        "#4C72B0",  # blue
        "#DD8452",  # orange
        "#55A868",  # green
        "#C44E52",  # red
        "#8172B3",  # purple
        "#937860",  # brown
        "#DA8BC3",  # pink
        "#8C8C8C",  # grey
    ]

    def generate_diagrams(
        self, report_path: Path, match_id: int
    ) -> Tuple[str, str, str, str]:
        """
        Generates:
        1. Detection count by object class (ball, player, goal)
        2. Detections over time (by timestamp) — EXCLUDES homography
        3. Player dynamics: speed, distance, acceleration per track_id
        4. Player position heatmap based on dx_meters/dy_meters (meter-scale)
        """

        df = pd.read_csv(report_path)
        parent_dir = DIAGRAMS_DIR / str(match_id)
        parent_dir.mkdir(exist_ok=True, parents=True)
        stem = report_path.stem

        if df.empty:
            logfire.warning("[DetectionsReporter] Report is empty, skipping diagrams")
            return "", "", "", ""

        class_chart = self._generate_class_counts_chart(df, parent_dir, stem)

        time_chart = self._generate_detections_over_time_chart(df, parent_dir, stem)

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning("[DetectionsReporter] No player data for dynamics chart")
            dynamics_chart = ""
            heatmap_chart = ""
        else:
            dynamics_chart = self._generate_player_dynamics_chart(
                players_df, parent_dir, stem
            )

            heatmap_chart = self._generate_heatmap_meters(players_df, parent_dir, stem)

        return (
            class_chart.as_posix() if class_chart else "",
            time_chart.as_posix() if time_chart else "",
            dynamics_chart.as_posix() if dynamics_chart else "",
            heatmap_chart.as_posix() if heatmap_chart else "",
        )

    def _generate_class_counts_chart(
        self, df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """
        Bar chart: detection count grouped by object_type.
        Includes all object types present in the report.
        """
        try:
            class_counts = df.groupby("object_type").size().sort_values(ascending=False)

            fig, ax = self._generate_plot(
                (8, 5),
                "Detection count by object class",
                "Object type",
                "Detection count",
            )
            class_counts.plot(
                kind="bar",
                ax=ax,
                color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
            )
            ax.tick_params(axis="x", rotation=0)
            plt.tight_layout()

            out_path = parent_dir / f"detections_by_class_{stem}.png"
            plt.savefig(out_path)
            plt.close(fig)

            logfire.info(
                f"[DiagramsGenerator] Saved class counts: {out_path.as_posix()}"
            )
            return out_path

        except Exception as e:
            logfire.error(f"[DiagramsGenerator] Error in class counts chart: {e}")
            return None

    def _generate_detections_over_time_chart(
        self, df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """
        Stacked area chart: detections over time by object_type.
        EXCLUDES "homography" rows — only ball, player, goal, etc.
        """
        try:
            df_no_homography = df[df["object_type"] != "homography"].copy()

            if df_no_homography.empty:
                logfire.warning(
                    "[DiagramsGenerator] No non-homography data for time chart"
                )
                return None

            df_sorted = df_no_homography.sort_values("timestamp")
            df_sorted["timestamp_bin"] = pd.cut(df_sorted["timestamp"], bins=30)
            time_counts = (
                df_sorted.groupby(["timestamp_bin", "object_type"], observed=True)
                .size()
                .unstack(fill_value=0)
            )

            fig, ax = self._generate_plot(
                (12, 5),
                "Detections over time",
                "Timestamp",
                "Detection count",
            )
            time_counts.plot(kind="area", ax=ax, alpha=0.6, stacked=True)
            ax.set_xticks([])
            ax.legend(title="Object type")
            plt.tight_layout()

            out_path = parent_dir / f"detections_over_time_{stem}.png"
            plt.savefig(out_path)
            plt.close(fig)

            logfire.info(
                f"[DiagramsGenerator] Saved detections over time: {out_path.as_posix()}"
            )
            return out_path

        except Exception as e:
            logfire.error(
                f"[DiagramsGenerator] Error in detections over time chart: {e}"
            )
            return None

    def _generate_player_dynamics_chart(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """
        3-panel bar chart: avg speed, distance, acceleration per track_id.
        """
        try:
            player_dynamics = (
                players_df.groupby("track_id")[["speed", "distance", "acceleration"]]
                .mean()
                .sort_values("track_id", ascending=True)
            )

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            player_dynamics["speed"].plot(kind="bar", ax=axes[0], color="#4C72B0")
            axes[0].set_title("Avg speed (km/h)")
            axes[0].set_xlabel("Track ID")
            axes[0].set_ylabel("km/h")

            player_dynamics["distance"].plot(kind="bar", ax=axes[1], color="#55A868")
            axes[1].set_title("Avg distance (m)")
            axes[1].set_xlabel("Track ID")
            axes[1].set_ylabel("Meters")

            player_dynamics["acceleration"].plot(
                kind="bar", ax=axes[2], color="#DD8452"
            )
            axes[2].set_title("Avg acceleration (m/s²)")
            axes[2].set_xlabel("Track ID")
            axes[2].set_ylabel("m/s²")

            plt.suptitle("Player dynamics by track ID", fontsize=13, y=1.02)
            plt.tight_layout()

            out_path = parent_dir / f"player_dynamics_{stem}.png"
            plt.savefig(out_path)
            plt.close(fig)

            logfire.info(
                f"[DiagramsGenerator] Saved player dynamics: {out_path.as_posix()}"
            )
            return out_path

        except Exception as e:
            logfire.error(f"[DiagramsGenerator] Error in player dynamics chart: {e}")
            return None

    def _generate_heatmap_meters(
        self, players_df: pd.DataFrame, parent_dir: Path, stem: str
    ) -> Path | None:
        """
        Hexbin heatmap of player positions using dx_meters and dy_meters
        for real-world meter-scale spatial analysis.
        """
        try:
            players_df["dx_meters"] = pd.to_numeric(
                players_df["dx_meters"], errors="coerce"
            )
            players_df["dy_meters"] = pd.to_numeric(
                players_df["dy_meters"], errors="coerce"
            )

            valid_data = players_df.dropna(subset=["dx_meters", "dy_meters"])

            if valid_data.empty:
                logfire.warning(
                    "[DiagramsGenerator] No valid dx_meters/dy_meters for heatmap"
                )
                return None

            fig, ax = self._generate_plot(
                (10, 10),
                "Player Position Heatmap (Metros)",
                "Posición X (metros)",
                "Posición Y (metros)",
            )

            ax.set_facecolor("#4a7c2f")

            x_min = valid_data["dx_meters"].min() - 2
            x_max = valid_data["dx_meters"].max() + 2
            y_min = valid_data["dy_meters"].min() - 2
            y_max = valid_data["dy_meters"].max() + 2

            self._draw_pitch(ax, x_min, x_max, y_min, y_max)

            # ax.set_xlim(x_min,x_max)
            # ax.set_ylim(y_min, y_max)
            # ax.invert_yaxis()

            hb = ax.hexbin(
                valid_data["dx_meters"],
                valid_data["dy_meters"],
                gridsize=30,
                cmap="YlOrRd",
                alpha=0.75,
                mincnt=1,
            )

            plt.colorbar(hb, ax=ax, label="Contador de detecciones")
            plt.tight_layout()

            out_path = parent_dir / f"player_heatmap_meters_{stem}.png"
            # ax.plot(
            #     [0, 120],
            #     [0, 0],
            #     color="cyan",
            #     linewidth=10,
            #     zorder=100
            # )
            plt.savefig(out_path)
            plt.close(fig)

            logfire.info(
                f"[DiagramsGenerator] Saved heatmap (meters): {out_path.as_posix()}"
            )
            return out_path

        except Exception as e:
            logfire.error(f"[DiagramsGenerator] Error in heatmap (meters): {e}")
            return None

    def _generate_plot(
        self,
        figsize: Tuple[int, int],
        title: str,
        x_label: str,
        y_label: str,
    ):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        return fig, ax

    def _build_chart(
        self,
        metric: str,
        ylabel: str,
        filename_prefix: str,
        n_players: int,
        report_path: Path,
        players_df: pd.DataFrame,
        match_id: int,
        track_ids,
    ) -> str:
        """Draw one stacked-panel chart for *metric* and return the saved path."""
        fig_height = max(3.0, n_players * 1.8)
        fig, axes = plt.subplots(
            n_players,
            1,
            figsize=(12, fig_height),
            sharex=True,
        )

        if n_players == 1:
            axes = [axes]

        for ax, track_id, color in zip(
            axes,
            track_ids,
            self._TRACK_COLORS * (n_players // len(self._TRACK_COLORS) + 1),
        ):
            player_data = players_df[players_df["track_id"] == track_id]

            ax.plot(
                player_data["frame_number"],
                player_data[metric],
                color=color,
                linewidth=0.9,
                alpha=0.85,
            )

            ax.fill_between(
                player_data["frame_number"],
                player_data[metric],
                alpha=0.08,
                color=color,
            )

            ax.set_title(
                f"Track ID = {track_id}",
                loc="left",
                fontsize=9,
                pad=3,
            )
            ax.set_ylabel(ylabel, fontsize=8)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, integer=False))
            ax.tick_params(axis="both", labelsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", linewidth=0.4, alpha=0.4, linestyle="--")

        axes[-1].set_xlabel("Frame number", fontsize=9)

        fig.suptitle(
            f"{ylabel} by player throughout the match",
            fontsize=11,
            y=1.01,
        )
        plt.tight_layout()

        out_path = (
            DIAGRAMS_DIR / str(match_id) / f"{filename_prefix}_{report_path.stem}.png"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logfire.info(f"[DetectionsReporter] Saved {out_path.as_posix()}")
        return out_path.as_posix()

    def generate_per_player_timeseries(
        self,
        report_path: Path,
        match_id: int,
    ) -> Tuple[str, str]:
        """
        Generates two vertically-stacked multi-panel time-series charts:

        1. Speed (km/h) by player over frame_number
            → saved as  speed_timeseries_<report_stem>.png

        2. Distance (m) by player over frame_number
            → saved as  distance_timeseries_<report_stem>.png

        Each panel in the figure corresponds to one track_id, exactly like
        the per-street panels in Figures 5 & 6 of the reference paper.

        Returns
        -------
        (speed_chart_path_str, distance_chart_path_str)
        Both are empty strings when there is no player data.
        """

        df = pd.read_csv(report_path)
        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning(
                "[DetectionsReporter] No player data — skipping per-player timeseries charts"
            )
            return "", ""

        players_df = players_df.sort_values("frame_number")

        track_ids = sorted(players_df["track_id"].unique())
        n_players = len(track_ids)

        speed_chart = self._build_chart(
            metric="speed",
            ylabel="Speed (km/h)",
            filename_prefix="speed_timeseries",
            n_players=n_players,
            report_path=report_path,
            players_df=players_df,
            track_ids=track_ids,
            match_id=match_id,
        )
        distance_chart = self._build_chart(
            metric="distance",
            ylabel="Distance (m)",
            filename_prefix="distance_timeseries",
            n_players=n_players,
            report_path=report_path,
            players_df=players_df,
            track_ids=track_ids,
            match_id=match_id,
        )

        return speed_chart, distance_chart

    def _draw_pitch(self, ax, x_min=None, x_max=None, y_min=None, y_max=None):
        """
        Dibuja SIEMPRE la cancha completa de futbol (105x68 tipico), con las 4
        porterias laterales (A, B, C, D) como referencia visual fija.

        Los parametros x_min/x_max/y_min/y_max se mantienen por compatibilidad
        con las llamadas existentes, pero YA NO se usan para recortar/zoomear
        la vista: el campo se muestra siempre completo para que ningun
        heatmap/trayectoria salga cortado.
        """
        PENALTY_AREA_LENGTH = 16.5
        PENALTY_AREA_WIDTH = 40.32
        GOAL_AREA_LENGTH = 5.5
        GOAL_AREA_WIDTH = 18.32
        CENTER_CIRCLE_RADIUS = 9.15
        line_color = "#A9B5A6"
        pitch_color = "#1D3A2F"
        lw = 1.4

        ax.set_facecolor(pitch_color)

        ax.add_patch(Rectangle((0, 0), PITCH_LENGTH, PITCH_WIDTH, fill=False,
                                edgecolor=line_color, linewidth=lw, zorder=1))
        ax.plot([PITCH_LENGTH / 2, PITCH_LENGTH / 2], [0, PITCH_WIDTH],
                color=line_color, linewidth=lw, zorder=1)
        ax.add_patch(Circle((PITCH_LENGTH / 2, PITCH_WIDTH / 2), CENTER_CIRCLE_RADIUS,
                            fill=False, edgecolor=line_color, linewidth=lw, zorder=1))
        ax.scatter(PITCH_LENGTH / 2, PITCH_WIDTH / 2, s=20, color=line_color, zorder=2)

        for side in ("left", "right", "top", "bottom"):
            self._draw_goal_box(
                ax, side=side,
                big_depth=PENALTY_AREA_LENGTH, big_width=PENALTY_AREA_WIDTH,
                small_depth=GOAL_AREA_LENGTH, small_width=GOAL_AREA_WIDTH,
                color=line_color, linewidth=lw,
            )

        self._draw_goal_lines(
            ax, x_min=0, x_max=PITCH_LENGTH, y_min=0, y_max=PITCH_WIDTH,
            color=line_color, linewidth=lw,
        )

        padding = 3.0
        GOAL_DEPTH = 2.0
        full_x_min, full_x_max = -GOAL_DEPTH - padding, PITCH_LENGTH + GOAL_DEPTH + padding
        full_y_min, full_y_max = -padding, PITCH_WIDTH + padding

        ax.set_xlim(full_x_min, full_x_max)
        ax.set_ylim(full_y_min, full_y_max)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.invert_yaxis()

        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_goal_box(self, ax, side, big_depth, big_width, small_depth, small_width, color, linewidth):
        """Dibuja área grande + área chica en el lateral indicado: 'left', 'right', 'top', 'bottom'."""
        if side == "left":
            big_xy = (0, (PITCH_WIDTH - big_width) / 2)
            big_size = (big_depth, big_width)
            small_xy = (0, (PITCH_WIDTH - small_width) / 2)
            small_size = (small_depth, small_width)
        elif side == "right":
            big_xy = (PITCH_LENGTH - big_depth, (PITCH_WIDTH - big_width) / 2)
            big_size = (big_depth, big_width)
            small_xy = (PITCH_LENGTH - small_depth, (PITCH_WIDTH - small_width) / 2)
            small_size = (small_depth, small_width)
        elif side == "top":
            big_xy = ((PITCH_LENGTH - big_width) / 2, 0)
            big_size = (big_width, big_depth)
            small_xy = ((PITCH_LENGTH - small_width) / 2, 0)
            small_size = (small_width, small_depth)
        elif side == "bottom":
            big_xy = ((PITCH_LENGTH - big_width) / 2, PITCH_WIDTH - big_depth)
            big_size = (big_width, big_depth)
            small_xy = ((PITCH_LENGTH - small_width) / 2, PITCH_WIDTH - small_depth)
            small_size = (small_width, small_depth)
        else:
            raise ValueError(f"side inválido: {side}")

        ax.add_patch(Rectangle(big_xy, *big_size, fill=False, edgecolor=color, linewidth=linewidth, zorder=1))
        ax.add_patch(Rectangle(small_xy, *small_size, fill=False, edgecolor=color, linewidth=linewidth, zorder=1))

    def _draw_goal_lines(
        self,
        ax,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        goal_width: float = 7.32,
        color: str = "white",
        linewidth: float = 2.5,
    ) -> None:
        """
        Dibuja una linea de porteria centrada en cada uno de los 4 laterales
        del grafico (arriba, abajo, izquierda, derecha).
        Es tolerante a que los limites lleguen invertidos (p.ej. tras
        ax.invert_yaxis()).
        """
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        # Usar abs() para que no importe el orden de los limites
        x_span = abs(x_max - x_min)
        y_span = abs(y_max - y_min)

        half_goal_x = min(goal_width, x_span) / 2
        half_goal_y = min(goal_width, y_span) / 2

        # arriba / abajo (usan la coordenada Y fija tal cual llega)
        ax.plot(
            [x_center - half_goal_x, x_center + half_goal_x],
            [y_min, y_min],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            [x_center - half_goal_x, x_center + half_goal_x],
            [y_max, y_max],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=5,
        )

        # izquierda / derecha
        ax.plot(
            [x_min, x_min],
            [y_center - half_goal_y, y_center + half_goal_y],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=5,
        )
        ax.plot(
            [x_max, x_max],
            [y_center - half_goal_y, y_center + half_goal_y],
            color=color,
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=5,
        )

        for spine in ax.spines.values():
            spine.set_visible(False)
