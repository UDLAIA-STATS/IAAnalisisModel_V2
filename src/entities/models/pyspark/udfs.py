import numpy as np
from pyspark.sql.functions import udf
import math
from pyspark.sql.types import (
    FloatType,
)
from skimage import color
from src.entities.models.pyspark.operations import (
    bbox_iou,
    calculate_distance,
    color_distance,
    direction_delta,
    euclidean_distance,
    hypot,
)


@udf(returnType=FloatType())
def color_distance_udf(color_a, color_b):
    return color_distance(color_a, color_b)


@udf(returnType=FloatType())
def euclidean_distance_udf(cx1, cy1 , cx2, cy2):
    return euclidean_distance(cx1, cy1, cx2, cy2)


@udf(returnType=FloatType())
def hypot_udf(dx, dy):
    return hypot(dx, dy)


@udf(returnType=FloatType())
def direction_delta_udf(dx_a, dy_a, dx_b, dy_b):
    return direction_delta(dx_a, dy_a, dx_b, dy_b)


@udf(returnType=FloatType())
def calculate_distance_udf(cx1, cy1, cx2, cy2):
    return calculate_distance(cx1, cy1, cx2, cy2)


@udf(returnType=FloatType())
def bbox_iou_udf(
    ax1,
    ay1,
    ax2,
    ay2,
    bx1,
    by1,
    bx2,
    by2,
):
    return bbox_iou(ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)
