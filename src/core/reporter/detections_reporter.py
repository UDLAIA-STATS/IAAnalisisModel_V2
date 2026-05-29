import csv
from pathlib import Path
from typing import List, Tuple
import uuid

import logfire
from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt

from pydantic import BaseModel
from sqlmodel import Session

from src.entities.types.bucket_types import FilePurposeTypes
from src.config.routes import DETECTED_OBJECTS_METRICS_DIR, DIAGRAMS_DIR
from src.core.repository.ball_repository import BallRepository
from src.core.repository.goal_repository import GoalRepository
from src.core.repository.player_repository import PlayerRepository
from src.core.repository.r2_repository import files_repository

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
    acceleration: float = 0.0
    timestamp: float


class DetectionsReporter:
    _TRACK_COLORS = [
        "#4C72B0",   # blue
        "#DD8452",   # orange
        "#55A868",   # green
        "#C44E52",   # red
        "#8172B3",   # purple
        "#937860",   # brown
        "#DA8BC3",   # pink
        "#8C8C8C",   # grey
    ]

    def generate_report(self, match_id: int, session: Session):
        report_rows: List[ReportRow] = []

        report_rows.extend(self.get_goals(match_id, session))
        report_rows.extend(self.get_balls(match_id, session))
        report_rows.extend(self.get_players(match_id, session))

        report_path = (
            DETECTED_OBJECTS_METRICS_DIR / f"report_{match_id}_{uuid.uuid4()}.csv"
        )

        with open(report_path, mode="w", newline="", encoding="utf-8") as file:
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
                    "acceleration",
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
                        row.acceleration,
                        row.timestamp,
                    ]
                )

        logfire.info(f"[DetectionsReporter] Report generated: {report_path.as_posix()}")

        class_chart_str, time_chart_str, dynamics_chart_str, heatmap_chart_str = self.generate_diagrams(report_path)

        report_key = files_repository.generate_key(match_id, report_path.stem, FilePurposeTypes.REPORTS, report_path.suffix[1:])
        files_repository.upload_report(report_key, report_path)
        speed_chart, distance_chart = self.generate_per_player_timeseries(report_path)

        chart_keys = {}
        for name, path_str in [
            ("class_chart", class_chart_str),
            ("time_chart", time_chart_str),
            ("dynamics_chart", dynamics_chart_str),
            ("heatmap_chart", heatmap_chart_str),
            ("speed_chart", speed_chart),
            ("distance_chart", distance_chart),
        ]:
            if path_str:
                p = Path(path_str)
                key = files_repository.generate_key(match_id, p.stem, FilePurposeTypes.REPORTS, p.suffix[1:])
                files_repository.upload_report(key, p)
                chart_keys[name] = key
            else:
                chart_keys[name] = None

        return (
            report_key,
            chart_keys["class_chart"],
            chart_keys["time_chart"],
            chart_keys["dynamics_chart"],
            chart_keys["heatmap_chart"],
            chart_keys["speed_chart"],
            chart_keys["distance_chart"],
        )

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
                        acceleration=state.acceleration,
                        shirt_color=player.team_color,
                        shirt_number=str(player.shirt_number if player.shirt_number is not None else ""),
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

    def generate_diagrams(self, report_path: Path) -> Tuple[str, str, str, str]:
        """
        Generates:
        1. Detection count by object class (ball, player, goal)
        2. Detections over time (by timestamp)
        3. Player dynamics: speed, distance, acceleration per track_id
        4. Player position heatmap based on detected bbox centers
        """

        df = pd.read_csv(report_path)

        if df.empty:
            logfire.warning("[DetectionsReporter] Report is empty, skipping diagrams")
            return "", "", "", ""

        # --- Chart 1: Detections by class ---
        class_counts = df.groupby("object_type").size().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        class_counts.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title("Detection count by object class")
        ax.set_xlabel("Object type")
        ax.set_ylabel("Detection count")
        ax.tick_params(axis="x", rotation=0)
        plt.tight_layout()

        class_chart = DIAGRAMS_DIR / f"detections_by_class_{report_path.stem}.png"
        plt.savefig(class_chart)
        plt.close(fig)

        # --- Chart 2: Detections over time ---
        df_sorted = df.sort_values("timestamp")
        df_sorted["timestamp_bin"] = pd.cut(df_sorted["timestamp"], bins=30)
        time_counts = df_sorted.groupby(["timestamp_bin", "object_type"], observed=True).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(12, 5))
        time_counts.plot(kind="area", ax=ax, alpha=0.6, stacked=True)
        ax.set_title("Detections over time")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Detection count")
        ax.set_xticks([])
        ax.legend(title="Object type")
        plt.tight_layout()

        time_chart = DIAGRAMS_DIR / f"detections_over_time_{report_path.stem}.png"
        plt.savefig(time_chart)
        plt.close(fig)

        # --- Chart 3: Player dynamics (speed, distance, acceleration) ---
        players_df = df[df["object_type"] == "player"].copy()

        if players_df.empty:
            logfire.warning("[DetectionsReporter] No player data for dynamics chart")
            dynamics_chart = ""
            heatmap_chart = ""
        else:
            player_dynamics = (
                players_df.groupby("track_id")[["speed", "distance", "acceleration"]]  # NOTE: "acceleration" must be added to ReportRow and CSV if not already present
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

            player_dynamics["acceleration"].plot(kind="bar", ax=axes[2], color="#DD8452")
            axes[2].set_title("Avg acceleration (m/s²)")
            axes[2].set_xlabel("Track ID")
            axes[2].set_ylabel("m/s²")

            plt.suptitle("Player dynamics by track ID", fontsize=13, y=1.02)
            plt.tight_layout()

            dynamics_chart = DIAGRAMS_DIR / f"player_dynamics_{report_path.stem}.png"
            plt.savefig(dynamics_chart)
            plt.close(fig)

            # --- Chart 4: Player position heatmap ---
            bbox_values = players_df["bbox"].str.split(", ", expand=True).astype(float)
            players_df = players_df.copy()
            players_df["center_x"] = (bbox_values[0] + bbox_values[2]) / 2
            players_df["center_y"] = (bbox_values[1] + bbox_values[3]) / 2

            fig, ax = plt.subplots(figsize=(10, 7))

            # Pitch background
            ax.set_facecolor("#4a7c2f")
            ax.set_xlim(players_df["center_x"].min() - 10, players_df["center_x"].max() + 10)
            ax.set_ylim(players_df["center_y"].min() - 10, players_df["center_y"].max() + 10)

            hb = ax.hexbin(
                players_df["center_x"],
                players_df["center_y"],
                gridsize=30,
                cmap="YlOrRd",
                alpha=0.75,
                mincnt=1,
            )

            plt.colorbar(hb, ax=ax, label="Detection count")
            ax.set_title("Player position heatmap")
            ax.set_xlabel("X position (px)")
            ax.set_ylabel("Y position (px)")
            ax.invert_yaxis()
            plt.tight_layout()

            heatmap_chart = DIAGRAMS_DIR / f"player_heatmap_{report_path.stem}.png"
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

    def _build_chart(
            self,
            metric: str,
            ylabel: str,
            filename_prefix: str,
            n_players: int,
            report_path: Path,
            players_df: pd.DataFrame,
            track_ids) -> str:
        """Draw one stacked-panel chart for *metric* and return the saved path."""
        # Panel height scales with player count; 1.8 in per player looks clean
        fig_height = max(3.0, n_players * 1.8)
        fig, axes = plt.subplots(
            n_players,
            1,
            figsize=(12, fig_height),
            sharex=True,
        )
 
        # Ensure axes is always iterable even for a single player
        if n_players == 1:
            axes = [axes]
 
        for ax, track_id, color in zip(
            axes, track_ids, self._TRACK_COLORS * (n_players // len(self._TRACK_COLORS) + 1)
        ):
            player_data = players_df[players_df["track_id"] == track_id]
 
            ax.plot(
                player_data["frame_number"],
                player_data[metric],
                color=color,
                linewidth=0.9,
                alpha=0.85,
            )
 
            # Light fill under the line for readability
            ax.fill_between(
                player_data["frame_number"],
                player_data[metric],
                alpha=0.08,
                color=color,
            )
 
            # Panel title in the top-left corner (matches paper style)
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
 
        # Shared x-axis label on the bottom-most panel only
        axes[-1].set_xlabel("Frame number", fontsize=9)
 
        fig.suptitle(
            f"{ylabel} by player throughout the match",
            fontsize=11,
            y=1.01,
        )
        plt.tight_layout()
 
        out_path = DIAGRAMS_DIR / f"{filename_prefix}_{report_path.stem}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logfire.info(f"[DetectionsReporter] Saved {out_path.as_posix()}")
        return out_path.as_posix()
 
        
    def generate_per_player_timeseries(
        self,
        report_path: Path,
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
    
        # Sort by frame so lines are drawn in chronological order
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
            track_ids=track_ids
        )
        distance_chart = self._build_chart(
            metric="distance",
            ylabel="Distance (m)",
            filename_prefix="distance_timeseries",
            n_players=n_players,
            report_path=report_path,
            players_df=players_df,
            track_ids=track_ids
        )
    
        return speed_chart, distance_chart


reporter = DetectionsReporter()