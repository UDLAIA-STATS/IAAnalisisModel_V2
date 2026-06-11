import numpy as np
from pyspark.sql.functions import udf
import math
from pyspark.sql.types import (
    FloatType,
)
from skimage import color

@udf(returnType=FloatType())
def color_distance_udf(color_a: str, color_b: str) -> float:
    try:
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
    except:
        return 999.0


@udf(returnType=FloatType())
def euclidean_distance_udf(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    if any(v is None for v in (cx1, cy1, cx2, cy2)):
        return float("inf")
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


@udf(returnType=FloatType())
def hypot_udf(dx: float, dy: float) -> float:
    return math.hypot(dx, dy) if dx is not None and dy is not None else 0.0


@udf(returnType=FloatType())
def direction_delta_udf(dx_a: float, dy_a: float, dx_b: float, dy_b: float) -> float:
    """Calculate angle between two vectors."""
    mag_a = math.hypot(dx_a, dy_a) if dx_a is not None and dy_a is not None else 0.0
    mag_b = math.hypot(dx_b, dy_b) if dx_b is not None and dy_b is not None else 0.0

    if mag_a < 1e-6 or mag_b < 1e-6:
        return 0.0

    dot = dx_a * dx_b + dy_a * dy_b
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_theta))


@udf(returnType=FloatType())
def calculate_distance_udf(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """UDF to calculate Euclidean distance."""
    return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))


@udf(returnType=FloatType())
def bbox_iou_udf(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    """Calculate IoU between two bounding boxes."""
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    if inter == 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0.0 else 0.0
