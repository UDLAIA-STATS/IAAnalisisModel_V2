from pathlib import Path
from typing import Tuple
from matplotlib import pyplot as plt, ticker

import logfire
import pandas as pd

from src.config.routes import DIAGRAMS_DIR


class DiagramsGenerator:
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
        2. Detections over time (by timestamp)
        3. Player dynamics: speed, distance, acceleration per track_id
        4. Player position heatmap based on detected bbox centers
        """

        df = pd.read_csv(report_path)
        parent_dir = DIAGRAMS_DIR / str(match_id)
        parent_dir.mkdir(exist_ok=True, parents=True)

        if df.empty:
            logfire.warning("[DetectionsReporter] Report is empty, skipping diagrams")
            return "", "", "", ""

        class_counts = df.groupby("object_type").size().sort_values(ascending=False)

        fig, ax = self._generate_plot(
            (8, 5),
            "Detection count by object class",
            "Object type",
            "Detection count",
        )
        class_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()

        class_chart = parent_dir / f"detections_by_class_{report_path.stem}.png"
        plt.savefig(class_chart)
        plt.close(fig)

        df_sorted = df.sort_values("timestamp")
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

        time_chart = parent_dir / f"detections_over_time_{report_path.stem}.png"
        plt.savefig(time_chart)
        plt.close(fig)

        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning("[DetectionsReporter] No player data for dynamics chart")
            dynamics_chart = ""
            heatmap_chart = ""
        else:
            player_dynamics = (
                players_df.groupby("track_id")[
                    ["speed", "distance", "acceleration"]
                ]  # NOTE: "acceleration" must be added to ReportRow and CSV if not already present
                .mean()
                .sort_values("speed", ascending=False)
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

            dynamics_chart = parent_dir / f"player_dynamics_{report_path.stem}.png"
            plt.savefig(dynamics_chart)
            plt.close(fig)

            bbox_values = players_df["bbox"].str.split(", ", expand=True).astype(float)
            players_df = players_df.copy()
            players_df["center_x"] = (bbox_values[0] + bbox_values[2]) / 2
            players_df["center_y"] = (bbox_values[1] + bbox_values[3]) / 2

            fig, ax = self._generate_plot((10, 10), "Player position heatmap", "X position (px)", "Y position (px)")

            ax.set_facecolor("#4a7c2f")
            ax.set_xlim(
                players_df["center_x"].min() - 10, players_df["center_x"].max() + 10
            )
            ax.set_ylim(
                players_df["center_y"].min() - 10, players_df["center_y"].max() + 10
            )

            hb = ax.hexbin(
                players_df["center_x"],
                players_df["center_y"],
                gridsize=30,
                cmap="YlOrRd",
                alpha=0.75,
                mincnt=1,
            )

            plt.colorbar(hb, ax=ax, label="Detection count")
            ax.invert_yaxis()
            plt.tight_layout()

            heatmap_chart = parent_dir / f"player_heatmap_{report_path.stem}.png"
            plt.savefig(heatmap_chart)
            plt.close(fig)

            dynamics_chart = dynamics_chart.as_posix()
            heatmap_chart = heatmap_chart.as_posix()

        return (
            class_chart.as_posix(),
            time_chart.as_posix(),
            dynamics_chart,
            heatmap_chart,
        )

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
