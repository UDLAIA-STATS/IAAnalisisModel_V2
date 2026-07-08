from typing import Dict

from pyspark.sql import DataFrame
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StringType,
    StructType,
    StructField,
)

from src.entities.models.app.player_validator_models import TrackSummary
from src.entities.models.app.union_model import MotionSnapshot


def physics_summaries_to_dataframe(
    summaries: Dict[int, MotionSnapshot], prefix: str, spark
) -> DataFrame:
    """Convert MotionSnapshot dict to Spark DataFrame."""
    rows = []
    for player_id, snapshot in summaries.items():
        rows.append(
            {
                f"{prefix}_player_id": player_id,
                f"{prefix}_frame_start": snapshot.frame_start,
                f"{prefix}_frame_end": snapshot.frame_end,
                f"{prefix}_timestamp_start": snapshot.timestamp_start,
                f"{prefix}_timestamp_end": snapshot.timestamp_end,
                f"{prefix}_detection_count": snapshot.detection_count,
                f"{prefix}_mean_dx": snapshot.mean_dx,
                f"{prefix}_mean_dy": snapshot.mean_dy,
                f"{prefix}_mean_speed_kmh": snapshot.mean_speed_kmh,
                f"{prefix}_mean_acceleration": snapshot.mean_acceleration,
                f"{prefix}_first_x1": snapshot.first_x1,
                f"{prefix}_first_y1": snapshot.first_y1,
                f"{prefix}_first_x2": snapshot.first_x2,
                f"{prefix}_first_y2": snapshot.first_y2,
                f"{prefix}_last_x1": snapshot.last_x1,
                f"{prefix}_last_y1": snapshot.last_y1,
                f"{prefix}_last_x2": snapshot.last_x2,
                f"{prefix}_last_y2": snapshot.last_y2,
            }
        )

    schema = StructType(
        [
            StructField(f"{prefix}_player_id", IntegerType(), True),
            StructField(f"{prefix}_frame_start", IntegerType(), True),
            StructField(f"{prefix}_frame_end", IntegerType(), True),
            StructField(f"{prefix}_timestamp_start", FloatType(), True),
            StructField(f"{prefix}_timestamp_end", FloatType(), True),
            StructField(f"{prefix}_detection_count", IntegerType(), True),
            StructField(f"{prefix}_mean_dx", FloatType(), True),
            StructField(f"{prefix}_mean_dy", FloatType(), True),
            StructField(f"{prefix}_mean_speed_kmh", FloatType(), True),
            StructField(f"{prefix}_mean_acceleration", FloatType(), True),
            StructField(f"{prefix}_first_x1", FloatType(), True),
            StructField(f"{prefix}_first_y1", FloatType(), True),
            StructField(f"{prefix}_first_x2", FloatType(), True),
            StructField(f"{prefix}_first_y2", FloatType(), True),
            StructField(f"{prefix}_last_x1", FloatType(), True),
            StructField(f"{prefix}_last_y1", FloatType(), True),
            StructField(f"{prefix}_last_x2", FloatType(), True),
            StructField(f"{prefix}_last_y2", FloatType(), True),
        ]
    )

    return spark.createDataFrame(rows, schema=schema)


def _summaries_to_dataframe(
    summaries: Dict[int, TrackSummary], prefix: str, spark
) -> DataFrame:
    """Convert summary dict to Spark DataFrame with early/late positions."""
    rows = []
    for player_id, summary in summaries.items():
        frame_gap = summary.frame_end - summary.frame_start

        rows.append(
            {
                f"{prefix}_player_id": player_id,
                f"{prefix}_frame_start": summary.frame_start,
                f"{prefix}_frame_end": summary.frame_end,
                f"{prefix}_detection_count": summary.detection_count,
                f"{prefix}_avg_color": summary.avg_color,
                "early_cx": summary.first_cx,
                "early_cy": summary.first_cy,
                "late_cx": summary.last_cx,
                "late_cy": summary.last_cy,
                "frame_gap": frame_gap,
            }
        )

    schema = StructType(
        [
            StructField(f"{prefix}_player_id", IntegerType(), True),
            StructField(f"{prefix}_frame_start", IntegerType(), True),
            StructField(f"{prefix}_frame_end", IntegerType(), True),
            StructField(f"{prefix}_detection_count", IntegerType(), True),
            StructField(f"{prefix}_avg_color", StringType(), True),
            StructField("early_cx", FloatType(), True),
            StructField("early_cy", FloatType(), True),
            StructField("late_cx", FloatType(), True),
            StructField("late_cy", FloatType(), True),
            StructField("frame_gap", IntegerType(), True),
        ]
    )

    return spark.createDataFrame(rows, schema=schema)
