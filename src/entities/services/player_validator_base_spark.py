from typing import Dict, List, Tuple
import numpy as np
from skimage import color

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    expr,
    avg,
    first,
    count,
    floor,
    when,
    max as sql_max,
    min as sql_min,
    struct,
    row_number,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    StringType,
)
from sqlmodel import Session


from src.entities.utils.spark_instance import spark
from src.entities.models.soccer.player_model import PlayerState
from src.entities.models.app.player_validator_models import PlayerCentroid, TrackSummary


class PlayerValidatorBase:
    MAX_DISTANCE = 130
    BASE_DISTANCE = 50
    PIXELS_PER_FRAME = 4
    MAX_COLOR_DISTANCE = 80
    FRAME_STEP = 20
    MAX_FRAME_GAP = 80
    MAX_WIDTH_DIFF = 25
    MAX_HEIGHT_DIFF = 25
    IOU_THRESHOLD = 0.15

    W_COLOR: float = 0.35
    W_POSITION: float = 0.50
    W_GAP: float = 0.1
    MERGE_SCORE_THRESHOLD: float = 20.0

    GHOST_TRACK_THRESHOLD: int = 15

    def __init__(self):
        """Initialize Spark Session for distributed processing."""
        self.spark = spark

    def _has_frame_overlap(
        self, summary_a: TrackSummary, summary_b: TrackSummary
    ) -> bool:
        """Return True if the two tracks are active at the same frame."""
        return max(summary_a.frame_start, summary_b.frame_start) <= min(
            summary_a.frame_end, summary_b.frame_end
        )

    def _composite_score(self, color_dist: float, pos_dist: float, gap: int) -> float:
        """Lower score = better match."""
        color_score = color_dist / self.MAX_COLOR_DISTANCE
        position_score = pos_dist / self.MAX_DISTANCE
        gap_score = gap / self.MAX_FRAME_GAP

        return (
            self.W_COLOR * color_score
            + self.W_POSITION * position_score
            + self.W_GAP * gap_score
        )

    def _build_track_summaries(
        self, centroids_df: DataFrame
    ) -> DataFrame:
        window_asc = Window.partitionBy("player_id").orderBy(col("frame_number"))
        window_desc = Window.partitionBy("player_id").orderBy(
            col("frame_number").desc()
        )

        candidate_struct = struct(
            col("cx"),
            col("cy"),
            col("x1"),
            col("y1"),
            col("x2"),
            col("y2"),
            col("width"),
            col("height"),
            col("confidence"),
            col("timestamp"),
            col("frame_number")
        )

        track_stats = centroids_df.groupBy("player_id").agg(
            sql_min("frame_number").alias("frame_start"),
            sql_max("frame_number").alias("frame_end"),
            avg("width").alias("avg_width"),
            avg("height").alias("avg_height"),
            count("*").alias("detection_count"),
            avg("confidence").alias("avg_confidence"),
            sql_min("timestamp").alias("timestamp_start"),
            sql_max("timestamp").alias("timestamp_end"),
        )

        df = (
            centroids_df.withColumn("row_asc", row_number().over(window_asc))
            .join(track_stats.select("player_id", "detection_count"), "player_id")
            .withColumn(
                "row_desc",
                row_number().over(window_desc),
            )
            .withColumn("first_candidate", when(col("row_asc") == 1, candidate_struct))
            .withColumn("mid_row", floor((col("detection_count") + 1) / 2))
            .withColumn(
                "mid_candidate",
                when(col("row_asc") == col("mid_row"), candidate_struct),
            )
            .withColumn("last_candidate", when(col("row_desc") == 1, candidate_struct))
        )

        avg_colors = (
            centroids_df.selectExpr(
                "player_id",
                "CAST(split(color, ',')[0] AS DOUBLE) AS r",
                "CAST(split(color, ',')[1] AS DOUBLE) AS g",
                "CAST(split(color, ',')[2] AS DOUBLE) AS b",
            )
            .groupBy("player_id")
            .agg(
                avg("r").cast(IntegerType()).alias("avg_r"),
                avg("g").cast(IntegerType()).alias("avg_g"),
                avg("b").cast(IntegerType()).alias("avg_b"),
            )
            .withColumn("avg_color", expr("concat_ws(',', avg_r, avg_g, avg_b)"))
            .select("player_id", "avg_color")
        )

        summaries_df = (
            df.groupBy("player_id")
            .agg(
                first("tracker_id").alias("tracker_id"),
                # sql_min("frame_number").alias("frame_start"),
                # sql_max("frame_number").alias("frame_end"),
                # count("*").alias("detection_count"),
                first("first_candidate", ignorenulls=True).alias("first_pos"),
                first("mid_candidate", ignorenulls=True).alias("mid_pos"),
                first("last_candidate", ignorenulls=True).alias("last_pos"),
            )
            .join(track_stats, "player_id")
            .join(avg_colors, "player_id", "left")
        )

        return summaries_df.select(
            col("player_id"),
            col("tracker_id"),
            col("frame_start"),
            col("frame_end"),
            col("detection_count"),
            col("avg_color"),
            col("avg_width"),
            col("avg_height"),
            col("timestamp_start"),
            col("timestamp_end"),
            col("first_pos.cx").alias("first_cx"),
            col("first_pos.cy").alias("first_cy"),
            col("first_pos.x1").alias("first_x1"),
            col("first_pos.y1").alias("first_y1"),
            col("first_pos.x2").alias("first_x2"),
            col("first_pos.y2").alias("first_y2"),
            col("mid_pos.cx").alias("mid_cx"),
            col("mid_pos.cy").alias("mid_cy"),
            col("mid_pos.x1").alias("mid_x1"),
            col("mid_pos.y1").alias("mid_y1"),
            col("mid_pos.x2").alias("mid_x2"),
            col("mid_pos.y2").alias("mid_y2"),
            col("last_pos.cx").alias("last_cx"),
            col("last_pos.cy").alias("last_cy"),
            col("last_pos.x1").alias("last_x1"),
            col("last_pos.y1").alias("last_y1"),
            col("last_pos.x2").alias("last_x2"),
            col("last_pos.y2").alias("last_y2"),
        )

    def _build_centroids(
        self, states: List[PlayerState], session: Session
    ) -> DataFrame:
        """
        Convert player states to Spark DataFrame of centroids.
        Returns a DataFrame instead of dict for distributed processing.
        """
        centroid_list = []

        for state in states:
            cx, cy = self.calculate_centroid(state, session)
            width = state.x2 - state.x1
            height = state.y2 - state.y1
            centroid_list.append(
                {
                    "player_id": state.player_id,
                    "tracker_id": state.player.track_id,
                    "frame_number": state.frame_number,
                    "x1": state.x1,
                    "y1": state.y1,
                    "x2": state.x2,
                    "y2": state.y2,
                    "cx": cx,
                    "cy": cy,
                    "width": width,
                    "height": height,
                    "color": state.player.team_color,
                    "confidence": state.confidence,
                    "timestamp": state.timestamp,
                }
            )

        schema = StructType(
            [
                StructField("player_id", IntegerType(), True),
                StructField("tracker_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("cx", FloatType(), True),
                StructField("cy", FloatType(), True),
                StructField("color", StringType(), True),
                StructField("width", FloatType(), True),
                StructField("height", FloatType(), True),
                StructField("confidence", FloatType(), True),
                StructField("timestamp", FloatType(), True),
            ]
        )

        return self.spark.createDataFrame(centroid_list, schema=schema)

    def _get_incorrect_near_frame(
        self,
        incorrect_centroids_df: DataFrame,
        actual_frame: int,
        total_frames: int,
        search_range: int = 20,
    ) -> List[PlayerCentroid]:
        """
        Return incorrect-track centroids from frames within ±search_range.
        Uses Spark filtering for efficiency.
        """
        start = max(1, actual_frame - search_range)
        stop = min(total_frames, actual_frame + search_range)

        result_df = incorrect_centroids_df.filter(
            (col("frame_number") >= start) & (col("frame_number") <= stop)
        )

        result = []
        for row in result_df.collect():
            result.append(
                PlayerCentroid(
                    player_id=row.player_id,
                    tracker_id=row.tracker_id,
                    cx=row.cx,
                    cy=row.cy,
                    color=row.color,
                    conf=row.conf,
                    frame_number=row.frame_number,
                )
            )

        return result

    def calculate_distance(self, centroid_a: tuple, centroid_b: tuple) -> float:
        cx1, cy1 = centroid_a
        cx2, cy2 = centroid_b
        return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))

    def calculate_centroid(
        self, player: PlayerState, session: Session
    ) -> Tuple[float, float]:
        cx = (player.x1 + player.x2) / 2
        cy = (player.y1 + player.y2) / 2

        return cx, cy

    def calculate_color_distance(self, color_a: str, color_b: str) -> float:
        rgb_a = (
            np.array([int(x.strip()) for x in color_a.split(",")], dtype=np.float32)
            / 255.0
        )
        rgb_b = (
            np.array([int(x.strip()) for x in color_b.split(",")], dtype=np.float32)
            / 255.0
        )

        lab_a = color.rgb2lab(rgb_a.reshape(1, 1, 3))
        lab_b = color.rgb2lab(rgb_b.reshape(1, 1, 3))

        l1, a1, b1 = lab_a[0][0]
        l2, a2, b2 = lab_b[0][0]

        return float(np.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2))
