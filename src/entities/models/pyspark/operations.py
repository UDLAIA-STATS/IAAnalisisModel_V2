import numpy as np
import math
from skimage import color
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logfire
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, lit, when, sqrt, pow, abs as sql_abs, avg, count, row_number,
    lag, lead, coalesce
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.functions import vector_to_array


def color_distance(color_a: str, color_b: str) -> float:
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


def euclidean_distance(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    if any(v is None for v in (cx1, cy1, cx2, cy2)):
        return float("inf")
    return math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)


def hypot(dx: float, dy: float) -> float:
    return math.hypot(dx, dy) if dx is not None and dy is not None else 0.0


def direction_delta(dx_a: float, dy_a: float, dx_b: float, dy_b: float) -> float:
    """Calculate angle between two vectors."""
    mag_a = math.hypot(dx_a, dy_a) if dx_a is not None and dy_a is not None else 0.0
    mag_b = math.hypot(dx_b, dy_b) if dx_b is not None and dy_b is not None else 0.0

    if mag_a < 1e-6 or mag_b < 1e-6:
        return 0.0

    dot = dx_a * dx_b + dy_a * dy_b
    cos_theta = max(-1.0, min(1.0, dot / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_theta))


def calculate_distance(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
    """UDF to calculate Euclidean distance."""
    return float(np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2))


def bbox_iou(
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


def compute_centroid(
    x1: Optional[float], y1: Optional[float], x2: Optional[float], y2: Optional[float]
) -> Tuple[float, float]:
    """Calcula el centroide de un bounding box."""
    if None in (x1, y1, x2, y2):
        return (0.0, 0.0)

    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def compute_area(
    x1: Optional[float], y1: Optional[float], x2: Optional[float], y2: Optional[float]
) -> float:
    """Calcula el área de un bounding box."""
    if None in (x1, y1, x2, y2):
        return 0.0
    return max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))


def add_distance_to_goal_line(
    df: DataFrame,
    goal_line: Dict[str, Any],
    x_col: str,
    y_col: str,
    output_col: str = "dist_to_goal_line"
) -> DataFrame:
    """
    Añade una columna con la distancia perpendicular a la línea de gol.
    goal_line: dict con 'slope', 'intercept', 'is_vertical'.
    x_col, y_col: nombres de columnas con coordenadas (en metros o píxeles).
    """
    if goal_line.get('is_vertical', False):
        return df.withColumn(
            output_col,
            sql_abs(col(x_col) - lit(goal_line['intercept']))
        )
    else:
        m = goal_line['slope']
        b = goal_line['intercept']
        return df.withColumn(
            output_col,
            sql_abs(m * col(x_col) - col(y_col) + b) / sqrt(lit(m * m + 1))
        )


def estimate_goal_line(
    goals_df: DataFrame,
    ball_df: Optional[DataFrame] = None,
    meter_per_pixel: float = 0.01,
    use_ball_positions: bool = True
) -> Dict[str, Any]:
    """
    Estima la línea de gol a partir de los centroides de los goles.
    Si hay suficientes puntos, ajusta una recta; si no, devuelve horizontal por defecto.
    """
    if goals_df.count() < 2:
        return {'slope': 0.0, 'intercept': 68.0, 'is_vertical': False}

    if use_ball_positions and ball_df is not None:
        goals_with_ball = goals_df.join(
            ball_df.select(
                col("frame_number").alias("bframe"),
                col("ball_mx").alias("gx_m"),
                col("ball_my").alias("gy_m")
            ),
            goals_df.frame_number == col("bframe"),
            "left"
        )
        goals_with_ball = goals_with_ball.withColumn(
            "gx_m", coalesce(col("gx_m"), col("goal_cx") * meter_per_pixel)
        ).withColumn(
            "gy_m", coalesce(col("gy_m"), col("goal_cy") * meter_per_pixel)
        )
        points_df = goals_with_ball.select("gx_m", "gy_m").dropna()
    else:
        points_df = goals_df.select(
            (col("goal_cx") * meter_per_pixel).alias("gx_m"),
            (col("goal_cy") * meter_per_pixel).alias("gy_m")
        ).dropna()

    points = points_df.collect()
    if len(points) < 2:
        return {'slope': 0.0, 'intercept': 68.0, 'is_vertical': False}

    xs = np.array([p.gx_m for p in points])
    ys = np.array([p.gy_m for p in points])
    if np.var(xs) < 0.1:
        return {'slope': float('inf'), 'intercept': float(np.mean(xs)), 'is_vertical': True}
    A = np.vstack([xs, np.ones(len(xs))]).T
    m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
    logfire.info(f"[estimate_goal_line] Goal line: y = {m:.3f}x + {b:.3f}")
    return {'slope': float(m), 'intercept': float(b), 'is_vertical': False}


def compute_goal_centroid(goals_df: DataFrame) -> Optional[Tuple[float, float]]:
    """Calcula el centro promedio de los goles en píxeles."""
    if goals_df.count() == 0:
        return None
    avg_row = goals_df.select(avg("goal_cx").alias("cx"), avg("goal_cy").alias("cy")).collect()[0]
    if avg_row.cx is None or avg_row.cy is None:
        return None
    return (float(avg_row.cx), float(avg_row.cy))


def train_random_forest_classifier(
    df: DataFrame,
    feature_cols: List[str],
    label_col: str,
    num_trees: int = 50,
    max_depth: int = 8,
    seed: int = 42,
    balance: bool = True,
    probability_threshold: Optional[float] = None
) -> Optional[PipelineModel]:
    """
    Entrena un clasificador RandomForest con escalado y balanceo opcional.
    Retorna el PipelineModel entrenado o None si no hay datos suficientes.
    """
    distinct_labels = df.select(label_col).distinct().count()
    if distinct_labels < 2:
        logfire.warning("train_random_forest: menos de 2 clases en la variable objetivo.")
        return None

    train_df = df.select([label_col] + feature_cols).na.drop()
    if train_df.count() < 10:
        logfire.warning("train_random_forest: datos insuficientes (<10 filas).")
        return None

    if balance:
        pos = train_df.filter(col(label_col) == 1)
        neg = train_df.filter(col(label_col) == 0)
        pos_count = pos.count()
        neg_count = neg.count()
        if pos_count > 0 and neg_count > 0 and pos_count < neg_count:
            frac = pos_count / neg_count
            neg_sampled = neg.sample(withReplacement=False, fraction=frac, seed=seed)
            train_df = pos.union(neg_sampled)
        elif neg_count < pos_count:
            frac = neg_count / pos_count
            pos_sampled = pos.sample(withReplacement=False, fraction=frac, seed=seed)
            train_df = neg.union(pos_sampled)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                            withStd=True, withMean=True)
    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol=label_col,
                                numTrees=num_trees, maxDepth=max_depth, seed=seed)
    pipeline = Pipeline(stages=[assembler, scaler, rf])
    model = pipeline.fit(train_df)
    logfire.info("train_random_forest: modelo entrenado correctamente.")
    return model


def predict_with_model(
    df: DataFrame,
    model: PipelineModel,
    probability_col: str = "probability",
    threshold: Optional[float] = None
) -> DataFrame:
    """
    Aplica el modelo a un DataFrame y añade la probabilidad de la clase positiva.
    Opcionalmente añade una columna de predicción binaria usando un umbral.
    """
    result = model.transform(df)
    prob_array = vector_to_array(col(probability_col))
    result = result.withColumn("prob_positive", prob_array.getItem(1))
    if threshold is not None:
        result = result.withColumn(
            "prediction_binary",
            when(col("prob_positive") >= threshold, 1).otherwise(0)
        )
    return result


def smooth_predictions(
    predictions_df: DataFrame,
    frame_col: str = "frame_number",
    player_id_col: str = "player_state_id",
    probability_col: str = "prob_positive",
    window_size: int = 3
) -> DataFrame:
    """
    Suaviza las predicciones usando una ventana deslizante: asigna al jugador con mayor
    probabilidad promedio en la ventana.
    """
    w = Window.orderBy(frame_col).rowsBetween(-window_size//2, window_size//2)
    smoothed = predictions_df.withColumn(
        "avg_prob",
        avg(probability_col).over(w)
    )
    w_rank = Window.partitionBy(frame_col).orderBy(col("avg_prob").desc())
    best = smoothed.withColumn("rn", row_number().over(w_rank)).filter(col("rn") == 1)
    return best.select(frame_col, col(player_id_col).alias("smoothed_player"))


def cluster_goal_posts(
    df: DataFrame,
    k: int = 2,
    feature_cols: List[str] = ["cx", "cy"],
    min_consecutive_frames: int = 3,
    distance_threshold: float = 150.0
) -> DataFrame:
    """
    Aplica KMeans para agrupar detecciones de postes y filtra clusters con pocas detecciones.
    Retorna el DataFrame con una columna 'cluster' y filtra los clusters válidos.
    """
    if df.count() < 10:
        return df.withColumn("cluster", lit(-1))

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    kmeans = KMeans(featuresCol="features", k=k, seed=42)
    pipeline = Pipeline(stages=[assembler, kmeans])
    model = pipeline.fit(df)
    clustered = model.transform(df)
    clustered = clustered.withColumnRenamed("prediction", "cluster")

    # Verificar si los centros están muy juntos
    centers = model.stages[-1].clusterCenters()
    if len(centers) == 2:
        dist = np.linalg.norm(centers[0] - centers[1])
        if dist < distance_threshold:
            logfire.info("cluster_goal_posts: clusters muy cercanos, usando k=1.")
            kmeans = KMeans(featuresCol="features", k=1, seed=42)
            pipeline = Pipeline(stages=[assembler, kmeans])
            model = pipeline.fit(df)
            clustered = model.transform(df)
            clustered = clustered.withColumnRenamed("prediction", "cluster")

    # Filtrar clusters con pocas detecciones
    cluster_counts = clustered.groupBy("cluster").agg(count("*").alias("cnt"))
    valid_clusters = cluster_counts.filter(col("cnt") >= min_consecutive_frames) \
                                   .select("cluster").collect()
    valid_ids = [row.cluster for row in valid_clusters]
    if valid_ids:
        clustered = clustered.filter(col("cluster").isin(valid_ids))
    else:
        clustered = clustered.withColumn("cluster", lit(-1))
    return clustered


def compute_movement_metrics(
    df: DataFrame,
    x_col: str,
    y_col: str,
    frame_rate: float = 30.0,
    meter_per_pixel: float = 0.01
) -> DataFrame:
    """
    Añade columnas de velocidad y aceleración a partir de posiciones (x, y).
    Se asume que las columnas están en píxeles y se convierten a metros para velocidad.
    """
    w = Window.orderBy("frame_number")
    df = df.withColumn("prev_x", lag(x_col, 1).over(w)) \
           .withColumn("prev_y", lag(y_col, 1).over(w)) \
           .withColumn("next_x", lead(x_col, 1).over(w)) \
           .withColumn("next_y", lead(y_col, 1).over(w)) \
           .withColumn("prev_frame", lag("frame_number", 1).over(w)) \
           .withColumn("next_frame", lead("frame_number", 1).over(w))

    df = df.withColumn(
        "vx",
        when(
            (col("next_frame").isNotNull()) & (col("prev_frame").isNotNull()),
            (col("next_x") - col("prev_x")) / (col("next_frame") - col("prev_frame")) * frame_rate * meter_per_pixel
        ).otherwise(
            when(
                col("next_frame").isNotNull(),
                (col("next_x") - col(x_col)) / (col("next_frame") - col("frame_number")) * frame_rate * meter_per_pixel
            ).otherwise(
                (col(x_col) - col("prev_x")) / (col("frame_number") - col("prev_frame")) * frame_rate * meter_per_pixel
            )
        )
    ).withColumn(
        "vy",
        when(
            (col("next_frame").isNotNull()) & (col("prev_frame").isNotNull()),
            (col("next_y") - col("prev_y")) / (col("next_frame") - col("prev_frame")) * frame_rate * meter_per_pixel
        ).otherwise(
            when(
                col("next_frame").isNotNull(),
                (col("next_y") - col(y_col)) / (col("next_frame") - col("frame_number")) * frame_rate * meter_per_pixel
            ).otherwise(
                (col(y_col) - col("prev_y")) / (col("frame_number") - col("prev_frame")) * frame_rate * meter_per_pixel
            )
        )
    )

    df = df.withColumn(
        "speed_kmh",
        sqrt(pow(col("vx"), 2) + pow(col("vy"), 2)) * 3.6  # m/s -> km/h
    )

    df = df.withColumn("prev_vx", lag("vx", 1).over(w)) \
           .withColumn("prev_vy", lag("vy", 1).over(w)) \
           .withColumn("next_vx", lead("vx", 1).over(w)) \
           .withColumn("next_vy", lead("vy", 1).over(w))

    df = df.withColumn(
        "ax",
        when(
            (col("next_frame").isNotNull()) & (col("prev_frame").isNotNull()),
            (col("next_vx") - col("prev_vx")) / (col("next_frame") - col("prev_frame")) * frame_rate
        ).otherwise(0.0)
    ).withColumn(
        "ay",
        when(
            (col("next_frame").isNotNull()) & (col("prev_frame").isNotNull()),
            (col("next_vy") - col("prev_vy")) / (col("next_frame") - col("prev_frame")) * frame_rate
        ).otherwise(0.0)
    )
    df = df.withColumn(
        "acceleration",
        sqrt(pow(col("ax"), 2) + pow(col("ay"), 2))
    )

    drop_cols = ["prev_x", "prev_y", "next_x", "next_y", "prev_frame", "next_frame",
                 "prev_vx", "prev_vy", "next_vx", "next_vy"]
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(c)
    return df
