import math
from typing import Dict, List, Optional, Tuple
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, lit, udf, when, row_number, avg, abs as sql_abs,
    sqrt, pow, broadcast, coalesce
)
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import (
    FloatType, IntegerType, BooleanType, StructType, StructField, StringType
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

from sqlmodel import Session, select, col as modelcol

from src.entities.services.goal_scorer_detector_base import GoalScorerDetectorBase
from src.entities.models.soccer.goal_model import GoalModel
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.player_model import PlayerModel, PlayerState


class GoalScorerDetector(GoalScorerDetectorBase):
    """
    Improved goal scorer detector using spatial reasoning and per‑match ML.
    """

    USE_ML_FOR_LINKING: bool = True
    USE_ML_FOR_SHOTS: bool = True
    ML_POSSESSION_MIN_SAMPLES: int = 30
    ML_SHOT_MIN_SAMPLES: int = 20
    GOAL_AREA_REDUCTION_FACTOR: float = 0.75

    def detect(self, match_id: int, session: Session) -> None:
        logfire.info(f"[GoalScorerDetector] Starting detection for match {match_id}")

        goals = self._fetch_goals(match_id, session)
        ball_states = self._fetch_ball_states(match_id, session)
        player_possessions = self._fetch_player_possessions(match_id, session)

        logfire.info(
            f"[GoalScorerDetector] Loaded: {len(goals)} goals, "
            f"{len(ball_states)} ball states, "
            f"{len(player_possessions)} possession events"
        )

        if not ball_states or not player_possessions:
            logfire.info("[GoalScorerDetector] Insufficient data (no ball or possessions)")
            return

        goals_df = self._build_goals_dataframe(goals)
        ball_df = self._build_ball_dataframe(ball_states)
        possession_df = self._build_possession_dataframe(player_possessions)

        goal_line = self._estimate_goal_line(goals_df, ball_df)

        if self.USE_ML_FOR_LINKING and goals_df.count() >= self.ML_POSSESSION_MIN_SAMPLES:
            goal_links = self._link_goals_with_ml(goals_df, ball_df, possession_df, goal_line)
        else:
            goal_links = self._link_goals_heuristic(goals_df, ball_df, possession_df, goal_line)

        if self.USE_ML_FOR_SHOTS and ball_df.count() >= self.ML_SHOT_MIN_SAMPLES:
            near_goal_shots = self._detect_shots_with_ml(goals_df, ball_df, possession_df, goal_line)
        else:
            near_goal_shots = self._detect_shots_heuristic(goals_df, ball_df, possession_df, goal_line)

        self._update_player_goals_shots(match_id, goal_links, near_goal_shots, session)

        logfire.info(f"[GoalScorerDetector] Completed for match {match_id}")

    def _fetch_goals(self, match_id: int, session: Session) -> List[GoalModel]:
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        return list(session.exec(stmt).all())

    def _fetch_ball_states(self, match_id: int, session: Session) -> List[BallState]:
        stmt = (
            select(BallState)
            .where(BallState.match_id == match_id)
            .order_by(modelcol(BallState.frame_number), modelcol(BallState.timestamp))
        )
        return list(session.exec(stmt).all())

    def _fetch_player_possessions(self, match_id: int, session: Session) -> List[Dict]:
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(modelcol(PlayerModel.match_id) == match_id)
            .where(PlayerState.has_ball == True)
            .order_by(modelcol(PlayerState.frame_number), modelcol(PlayerState.timestamp))
        )

        results = []
        for state, player in session.exec(stmt).all():
            results.append({
                "player_state_id": state.id,
                "player_id": player.id,
                "track_id": player.track_id,
                "team_id": player.team_id,
                "team_color": player.team_color,
                "frame_number": state.frame_number,
                "timestamp": state.timestamp,
                "player_x1": state.x1,
                "player_y1": state.y1,
                "player_x2": state.x2,
                "player_y2": state.y2,
                "ball_x": state.ball_x,
                "ball_y": state.ball_y,
                "confidence": state.confidence,
                "shirt_number": player.shirt_number,
                "player_speed": state.speed_kmh or 0.0,
                "vx": state.vx or 0.0,
                "vy": state.vy or 0.0,
            })
        return results


    def _build_goals_dataframe(self, goals: List[GoalModel]) -> DataFrame:
        rows = []
        for g in goals:
            cx, cy = self._compute_centroid(g.x1, g.y1, g.x2, g.y2)
            area = max(0.0, (g.x2 or 0) - (g.x1 or 0)) * max(0.0, (g.y2 or 0) - (g.y1 or 0))
            rows.append({
                "goal_id": g.id,
                "match_id": g.match_id,
                "frame_number": g.frame_number,
                "timestamp": g.timestamp,
                "confidence": g.confidence,
                "x1": g.x1,
                "y1": g.y1,
                "x2": g.x2,
                "y2": g.y2,
                "goal_cx": cx,
                "goal_cy": cy,
                "goal_area": area,
            })

        schema = StructType([
            StructField("goal_id", IntegerType(), True),
            StructField("match_id", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("x1", FloatType(), True),
            StructField("y1", FloatType(), True),
            StructField("x2", FloatType(), True),
            StructField("y2", FloatType(), True),
            StructField("goal_cx", FloatType(), True),
            StructField("goal_cy", FloatType(), True),
            StructField("goal_area", FloatType(), True),
        ])
        return self.spark.createDataFrame(rows, schema=schema)

    def _build_ball_dataframe(self, ball_states: List[BallState]) -> DataFrame:
        rows = []
        for b in ball_states:
            cx, cy = self._compute_centroid(b.x1, b.y1, b.x2, b.y2)
            area = max(0.0, (b.x2 or 0) - (b.x1 or 0)) * max(0.0, (b.y2 or 0) - (b.y1 or 0))

            ball_mx = cx * self.DEFAULT_METER_PER_PIXEL
            ball_my = cy * self.DEFAULT_METER_PER_PIXEL

            rows.append({
                "ball_state_id": b.id,
                "match_id": b.match_id,
                "frame_number": b.frame_number,
                "timestamp": b.timestamp,
                "confidence": b.confidence,
                "x1": b.x1,
                "y1": b.y1,
                "x2": b.x2,
                "y2": b.y2,
                "ball_cx": cx,
                "ball_cy": cy,
                "ball_area": area,
                "ball_mx": ball_mx,
                "ball_my": ball_my,
                "has_meters": False,
                "speed_kmh": b.speed_kmh or 0.0,
                "vx": b.vx or 0.0,
                "vy": b.vy or 0.0,
                "dx": b.dx or 0.0,
                "dy": b.dy or 0.0,
                "dx_meters": b.dx_meters or 0.0,
                "dy_meters": b.dy_meters or 0.0,
            })

        schema = StructType([
            StructField("ball_state_id", IntegerType(), True),
            StructField("match_id", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("x1", FloatType(), True),
            StructField("y1", FloatType(), True),
            StructField("x2", FloatType(), True),
            StructField("y2", FloatType(), True),
            StructField("ball_cx", FloatType(), True),
            StructField("ball_cy", FloatType(), True),
            StructField("ball_area", FloatType(), True),
            StructField("ball_mx", FloatType(), True),
            StructField("ball_my", FloatType(), True),
            StructField("has_meters", BooleanType(), True),
            StructField("speed_kmh", FloatType(), True),
            StructField("vx", FloatType(), True),
            StructField("vy", FloatType(), True),
            StructField("dx", FloatType(), True),
            StructField("dy", FloatType(), True),
            StructField("dx_meters", FloatType(), True),
            StructField("dy_meters", FloatType(), True),
        ])
        return self.spark.createDataFrame(rows, schema=schema)

    def _build_possession_dataframe(self, possessions: List[Dict]) -> DataFrame:
        rows = []
        for p in possessions:
            pcx, pcy = self._compute_centroid(
                p["player_x1"], p["player_y1"], p["player_x2"], p["player_y2"]
            )
            rows.append({
                "possession_id": p["player_state_id"],
                "player_id": p["player_id"],
                "track_id": p["track_id"],
                "team_id": p["team_id"],
                "team_color": p["team_color"],
                "shirt_number": p["shirt_number"],
                "frame_number": p["frame_number"],
                "timestamp": p["timestamp"],
                "player_x1": p["player_x1"],
                "player_y1": p["player_y1"],
                "player_x2": p["player_x2"],
                "player_y2": p["player_y2"],
                "player_cx": pcx,
                "player_cy": pcy,
                "ball_x": p["ball_x"],
                "ball_y": p["ball_y"],
                "confidence": p["confidence"],
                "player_speed": p.get("player_speed", 0.0),
                "vx": p.get("vx", 0.0),
                "vy": p.get("vy", 0.0),
            })

        schema = StructType([
            StructField("possession_id", IntegerType(), True),
            StructField("player_id", IntegerType(), True),
            StructField("track_id", IntegerType(), True),
            StructField("team_id", IntegerType(), True),
            StructField("team_color", StringType(), True),
            StructField("shirt_number", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("timestamp", FloatType(), True),
            StructField("player_x1", FloatType(), True),
            StructField("player_y1", FloatType(), True),
            StructField("player_x2", FloatType(), True),
            StructField("player_y2", FloatType(), True),
            StructField("player_cx", FloatType(), True),
            StructField("player_cy", FloatType(), True),
            StructField("ball_x", FloatType(), True),
            StructField("ball_y", FloatType(), True),
            StructField("confidence", FloatType(), True),
            StructField("player_speed", FloatType(), True),
            StructField("vx", FloatType(), True),
            StructField("vy", FloatType(), True),
        ])
        return self.spark.createDataFrame(rows, schema=schema)

    def _estimate_goal_line(self, goals_df: DataFrame, ball_df: DataFrame) -> Dict:
        """
        Estima la línea de gol a partir de los centroides de los goles.
        Devuelve {'slope': m, 'intercept': b, 'is_vertical': bool}.
        Si no hay suficientes goles, devuelve horizontal estándar.
        """
        if goals_df.count() < 2:
            return {'slope': 0.0, 'intercept': 68.0, 'is_vertical': False}

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
            "gx_m", coalesce(col("gx_m"), col("goal_cx") * self.DEFAULT_METER_PER_PIXEL)
        ).withColumn(
            "gy_m", coalesce(col("gy_m"), col("goal_cy") * self.DEFAULT_METER_PER_PIXEL)
        )

        points = goals_with_ball.select("gx_m", "gy_m").dropna().collect()
        if len(points) < 2:
            return {'slope': 0.0, 'intercept': 68.0, 'is_vertical': False}

        import numpy as np
        xs = np.array([p.gx_m for p in points])
        ys = np.array([p.gy_m for p in points])
        if np.var(xs) < 0.1:
            return {'slope': float('inf'), 'intercept': np.mean(xs), 'is_vertical': True}
        A = np.vstack([xs, np.ones(len(xs))]).T
        m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        logfire.info(f"[GoalScorerDetector] Goal line: y = {m:.3f}x + {b:.3f}")
        return {'slope': float(m), 'intercept': float(b), 'is_vertical': False}


    def _add_distance_to_goal_line(self, df: DataFrame, goal_line: Dict, x_col: str, y_col: str) -> DataFrame:
        """
        Añade columna 'dist_to_goal_line' con distancia perpendicular a la línea.
        x_col, y_col deben ser nombres de columnas con coordenadas en metros.
        """
        if goal_line['is_vertical']:
            return df.withColumn(
                "dist_to_goal_line",
                sql_abs(col(x_col) - lit(goal_line['intercept']))
            )
        else:
            m = goal_line['slope']
            b = goal_line['intercept']
            return df.withColumn(
                "dist_to_goal_line",
                sql_abs(m * col(x_col) - col(y_col) + b) / sqrt(lit(m*m + 1))
            )

    def _link_goals_with_ml(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> List[Dict]:
        """
        Usa un clasificador RandomForest para predecir el jugador que anotó.
        """
        candidates = self._generate_goal_candidates(goals_df, ball_df, possession_df, goal_line)
        if candidates.count() == 0:
            return []

        w = Window.partitionBy("goal_id").orderBy(col("score").asc())
        labeled = candidates.withColumn("rn", row_number().over(w)) \
                            .withColumn("label", when(col("rn") == 1, 1).otherwise(0))

        feature_cols = [
            "frame_gap", "ball_to_goal_dist_px", "dist_to_goal_line",
            "player_speed", "ball_speed", "confidence"
        ]
        for c in feature_cols:
            if c not in labeled.columns:
                labeled = labeled.withColumn(c, lit(0.0))

        train_df = labeled.filter(col("label").isNotNull()).na.drop(subset=feature_cols + ["label"])
        if train_df.count() < 10:
            logfire.warning("Not enough labeled samples for ML linking, falling back to heuristic")
            return self._link_goals_heuristic(goals_df, ball_df, possession_df, goal_line)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                                withStd=True, withMean=True)
        rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label",
                                    numTrees=50, maxDepth=8, seed=42)
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(train_df)

        scored = model.transform(candidates)
        prob = vector_to_array(col("probability")).getItem(1)
        scored = scored.withColumn("ml_score", prob)

        w_goal = Window.partitionBy("goal_id").orderBy(col("ml_score").desc())
        best = scored.withColumn("rn", row_number().over(w_goal)).filter(col("rn") == 1)

        results = []
        for row in best.collect():
            results.append({
                "goal_id": row.goal_id,
                "player_id": row.player_id,
                "track_id": row.track_id,
                "team_id": row.team_id,
                "shirt_number": row.shirt_number,
                "possession_frame": row.frame_number,
                "goal_frame": row.goal_frame,
                "frame_gap": row.frame_gap,
                "dist_to_goal_line": row.dist_to_goal_line,
                "ml_score": row.ml_score,
                "is_goal": True,
                "event_type": "goal"
            })
        logfire.info(f"[GoalScorerDetector] ML linked {len(results)} goals")
        return results

    def _generate_goal_candidates(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> DataFrame:
        """
        Genera candidatos (posesión, gol) y añade información de colisión
        con el área de gol reducida (40% menos).
        """
        g = goals_df.alias("g")
        p = possession_df.alias("p")

        candidates = p.join(broadcast(g), how="inner") \
            .where(col("p.frame_number") < col("g.frame_number")) \
            .where((col("g.frame_number") - col("p.frame_number")) <= self.POSSESSION_LOOKBACK_FRAMES) \
            .where((col("g.timestamp") - col("p.timestamp")) <= self.POSSESSION_LOOKBACK_SECONDS)

        candidates = candidates.withColumn(
            "frame_gap", col("g.frame_number") - col("p.frame_number")
        )
        candidates = candidates.withColumn(
            "ball_to_goal_dist_px",
            sqrt(pow(col("p.ball_x") - col("g.goal_cx"), 2) +
                 pow(col("p.ball_y") - col("g.goal_cy"), 2))
        )

        ball_at_goal = ball_df.select(
            col("frame_number").alias("g_ball_frame"),
            col("ball_cx").alias("ball_at_goal_cx"),
            col("ball_cy").alias("ball_at_goal_cy"),
            col("confidence").alias("ball_conf_at_goal")
        )
        candidates = candidates.join(
            broadcast(ball_at_goal),
            col("g.frame_number") == col("g_ball_frame"),
            "left"
        )

        def is_inside_reduced_goal(x1, y1, x2, y2, bx, by):
            if None in (x1, y1, x2, y2, bx, by):
                return False
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            width = x2 - x1
            height = y2 - y1
            scale = math.sqrt(0.75)
            nw = width * scale
            nh = height * scale
            return (cx - nw/2 <= bx <= cx + nw/2) and (cy - nh/2 <= by <= cy + nh/2)

        is_inside_udf = udf(is_inside_reduced_goal, BooleanType())

        candidates = candidates.withColumn(
            "inside_goal_area",
            is_inside_udf(
                col("g.x1"), col("g.y1"), col("g.x2"), col("g.y2"),
                col("ball_at_goal_cx"), col("ball_at_goal_cy")
            )
        )
        candidates = candidates.fillna({"inside_goal_area": False})

        candidates = candidates.withColumn(
            "ball_at_goal_mx",
            coalesce(col("ball_at_goal_cx"), lit(0.0)) * self.DEFAULT_METER_PER_PIXEL
        )
        candidates = candidates.withColumn(
            "ball_at_goal_my",
            coalesce(col("ball_at_goal_cy"), lit(0.0)) * self.DEFAULT_METER_PER_PIXEL
        )
        candidates = self._add_distance_to_goal_line(
            candidates, goal_line, "ball_at_goal_mx", "ball_at_goal_my"
        )
        candidates = candidates.withColumnRenamed("dist_to_goal_line", "dist_to_goal_line_m")

        ball_speed_at_goal = ball_df.select(
            col("frame_number").alias("g_ball_speed_frame"),
            col("speed_kmh").alias("ball_speed")
        )
        candidates = candidates.join(
            broadcast(ball_speed_at_goal),
            col("g.frame_number") == col("g_ball_speed_frame"),
            "left"
        )
        candidates = candidates.fillna({"ball_speed": 0.0})

        candidates = candidates.withColumn(
            "score",
            when(
                col("inside_goal_area"),
                (col("frame_gap") / self.POSSESSION_LOOKBACK_FRAMES) * 0.10 +
                (col("ball_to_goal_dist_px") / 1000.0) * 0.10 +
                (col("dist_to_goal_line_m") / 10.0) * 0.10 +
                (1.0 - col("p.confidence")) * 0.20 +
                (1.0 - col("ball_conf_at_goal")) * 0.20
            ).otherwise(
                (col("frame_gap") / self.POSSESSION_LOOKBACK_FRAMES) * 0.30 +
                (col("ball_to_goal_dist_px") / 1000.0) * 0.25 +
                (col("dist_to_goal_line_m") / 10.0) * 0.25 +
                (1.0 - col("p.confidence")) * 0.20 +
                10.0
            )
        )

        result = candidates.select(
            col("g.goal_id").alias("goal_id"),
            col("p.frame_number").alias("frame_number"),
            col("p.player_id").alias("player_id"),
            col("p.track_id").alias("track_id"),
            col("p.team_id").alias("team_id"),
            col("p.shirt_number").alias("shirt_number"),
            col("p.confidence").alias("confidence"),
            col("p.player_speed").alias("player_speed"),
            col("g.frame_number").alias("goal_frame"),
            col("g.timestamp").alias("goal_timestamp"),
            col("frame_gap"),
            col("ball_to_goal_dist_px"),
            col("dist_to_goal_line_m"),
            col("ball_speed"),
            col("inside_goal_area"),
            col("score")
        )
        return result
    
    def _compute_reduced_goal_rect(self, row: Dict) -> Dict:
        """
        Dado un registro de gol (con x1,y1,x2,y2), devuelve el rectángulo
        reducido al 60% del área (40% menos) manteniendo el centro.
        """
        x1 = row["x1"]
        y1 = row["y1"]
        x2 = row["x2"]
        y2 = row["y2"]
        if None in (x1, y1, x2, y2):
            return {}

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        scale = math.sqrt(self.GOAL_AREA_REDUCTION_FACTOR)
        new_width = width * scale
        new_height = height * scale
        return {
            "cx": cx,
            "cy": cy,
            "x1": cx - new_width / 2,
            "x2": cx + new_width / 2,
            "y1": cy - new_height / 2,
            "y2": cy + new_height / 2,
        }

    def _is_inside_rect(self, px: float, py: float, rect: Dict) -> bool:
        """Comprueba si el punto (px, py) está dentro del rectángulo."""
        if rect is None:
            return False
        return (rect["x1"] <= px <= rect["x2"]) and (rect["y1"] <= py <= rect["y2"])

    def _link_goals_heuristic(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> List[Dict]:
        candidates = self._generate_goal_candidates(goals_df, ball_df, possession_df, goal_line)
        if candidates.count() == 0:
            return []

        w = Window.partitionBy("goal_id").orderBy(col("score").asc())
        best = candidates.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)

        results = []
        for row in best.collect():
            if row.inside_goal_area:
                results.append({
                    "goal_id": row.goal_id,
                    "player_id": row.player_id,
                    "track_id": row.track_id,
                    "team_id": row.team_id,
                    "shirt_number": row.shirt_number,
                    "possession_frame": row.frame_number,
                    "goal_frame": row.goal_frame,
                    "frame_gap": row.frame_gap,
                    "dist_to_goal_line": row.dist_to_goal_line_m,
                    "score": row.score,
                    "is_goal": True,
                    "event_type": "goal"
                })
        logfire.info(f"[GoalScorerDetector] Heuristic linked {len(results)} goals (inside reduced area)")
        return results

    def _detect_shots_with_ml(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> List[Dict]:
        """
        Entrena un clasificador para detectar disparos que pasan cerca de la portería.
        """
        # Generar candidatos de disparos (posesiones y trayectorias del balón)
        candidates = self._generate_shot_candidates(goals_df, ball_df, possession_df, goal_line)
        if candidates.count() < 10:
            return self._detect_shots_heuristic(goals_df, ball_df, possession_df, goal_line)

        # Etiquetar: usar goles conocidos como positivos (disparos que terminaron en gol)
        # y frames aleatorios lejos de la portería como negativos.
        # Para simplificar, usamos la heurística para generar pseudo-etiquetas.
        # En lugar de eso, podemos usar directamente los goles como positivos.
        # Asumimos que los goles están en goals_df y sus frames.
        goal_frames = set()
        if goals_df.count() > 0:
            goal_frames = {row.frame_number for row in goals_df.select("frame_number").collect()}

        # Añadir columna label: 1 si el frame del balón está en goal_frames, 0 en caso contrario.
        candidates = candidates.withColumn(
            "label",
            when(col("ball_frame").isin(list(goal_frames)), 1).otherwise(0)
        )

        # Filtrar para tener ejemplos balanceados (si hay muy pocos positivos, muestreamos negativos)
        pos = candidates.filter(col("label") == 1)
        neg = candidates.filter(col("label") == 0)
        pos_count = pos.count()
        if pos_count < 5:
            return self._detect_shots_heuristic(goals_df, ball_df, possession_df, goal_line)

        # Muestrear negativos para balancear
        neg_sampled = neg.sample(withReplacement=False, fraction=min(1.0, pos_count*2 / neg.count()), seed=42)
        train_df = pos.union(neg_sampled)

        feature_cols = [
            "dist_to_goal_line", "ball_speed", "player_speed",
            "angle_to_goal", "confidence"
        ]
        for c in feature_cols:
            if c not in train_df.columns:
                train_df = train_df.withColumn(c, lit(0.0))

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        scaler = StandardScaler(inputCol="features", outputCol="scaled_features",
                                withStd=True, withMean=True)
        rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="label",
                                    numTrees=50, maxDepth=8, seed=42)
        pipeline = Pipeline(stages=[assembler, scaler, rf])
        model = pipeline.fit(train_df)

        scored = model.transform(candidates)
        prob = vector_to_array(scored["probability"])
        scored = scored.withColumn("ml_score", prob.getItem(1))

        threshold = 0.6
        final = scored.filter(col("ml_score") >= threshold)

        w = Window.partitionBy("player_id").orderBy(col("ml_score").desc())
        best = final.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)

        results = []
        for row in best.collect():
            results.append({
                "player_id": row.player_id,
                "track_id": row.track_id,
                "team_id": row.team_id,
                "shirt_number": row.shirt_number,
                "possession_frame": row.frame_number,
                "shot_frame": row.ball_frame,
                "dist_to_goal_line": row.dist_to_goal_line,
                "speed_kmh": row.ball_speed,
                "ml_score": row.ml_score,
                "is_goal": False,
                "event_type": "near_goal_shot"
            })
        logfire.info(f"[GoalScorerDetector] ML detected {len(results)} near-goal shots")
        return results

    def _generate_shot_candidates(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> DataFrame:
        """
        Genera candidatos para disparos usando el área original de la portería.
        """
        p = possession_df.alias("p")
        b = ball_df.alias("b")

        candidates = p.join(b, how="inner") \
            .where(col("p.frame_number") < col("b.frame_number")) \
            .where((col("b.frame_number") - col("p.frame_number")) <= self.SHOT_LOOKAHEAD_FRAMES) \
            .where((col("b.timestamp") - col("p.timestamp")) <= self.SHOT_LOOKAHEAD_SECONDS) \
            .where(col("b.speed_kmh") >= self.MIN_SHOT_SPEED_KMH) \
            .where(col("b.speed_kmh") <= self.MAX_SHOT_SPEED_KMH)

        if goals_df.count() > 0:
            avg_rect = goals_df.select(
                avg("x1").alias("avg_x1"),
                avg("y1").alias("avg_y1"),
                avg("x2").alias("avg_x2"),
                avg("y2").alias("avg_y2")
            ).collect()[0]
            if avg_rect.avg_x1 is not None:
                rect_orig = {
                    "x1": avg_rect.avg_x1,
                    "y1": avg_rect.avg_y1,
                    "x2": avg_rect.avg_x2,
                    "y2": avg_rect.avg_y2,
                }
                def is_inside_orig(bx, by):
                    return (rect_orig["x1"] <= bx <= rect_orig["x2"]) and \
                           (rect_orig["y1"] <= by <= rect_orig["y2"])
                is_inside_udf = udf(is_inside_orig, BooleanType())
                candidates = candidates.withColumn(
                    "inside_shot_area",
                    is_inside_udf(col("b.ball_cx"), col("b.ball_cy"))
                )
            else:
                candidates = candidates.withColumn("inside_shot_area", lit(False))
        else:
            candidates = candidates.withColumn("inside_shot_area", lit(False))

        candidates = self._add_distance_to_goal_line(
            candidates, goal_line, "b.ball_mx", "b.ball_my"
        )

        goal_center = self._compute_goal_center(goals_df)
        if goal_center is None:
            goal_center = (0.0, 0.0)

        candidates = candidates.withColumn("goal_center_x", lit(goal_center[0]))
        candidates = candidates.withColumn("goal_center_y", lit(goal_center[1]))
        candidates = candidates.withColumn(
            "to_goal_x", col("goal_center_x") - col("b.ball_mx")
        ).withColumn(
            "to_goal_y", col("goal_center_y") - col("b.ball_my")
        )
        candidates = candidates.withColumn(
            "dot_product",
            col("b.vx") * col("to_goal_x") + col("b.vy") * col("to_goal_y")
        ).withColumn(
            "norm_v", sqrt(col("b.vx")**2 + col("b.vy")**2)
        ).withColumn(
            "norm_to_goal", sqrt(col("to_goal_x")**2 + col("to_goal_y")**2)
        )
        candidates = candidates.withColumn(
            "angle_to_goal",
            when(
                (col("norm_v") > 0) & (col("norm_to_goal") > 0),
                sql_abs(col("dot_product") / (col("norm_v") * col("norm_to_goal")))
            ).otherwise(0.0)
        )

        result = candidates.select(
            col("p.player_id"),
            col("p.track_id"),
            col("p.team_id"),
            col("p.shirt_number"),
            col("p.frame_number"),
            col("p.timestamp"),
            col("b.frame_number").alias("ball_frame"),
            col("b.timestamp").alias("ball_timestamp"),
            col("b.speed_kmh").alias("ball_speed"),
            col("p.player_speed"),
            col("p.confidence"),
            col("dist_to_goal_line"),
            col("angle_to_goal"),
            col("inside_shot_area")
        )
        return result
    
    def _compute_goal_center(self, goals_df: DataFrame) -> Optional[Tuple[float, float]]:
        """Calcula el centro promedio de los goles en metros."""
        if goals_df.count() == 0:
            return None
        avg_row = goals_df.select(
            avg("goal_cx").alias("cx"),
            avg("goal_cy").alias("cy")
        ).collect()[0]
        if avg_row.cx is None or avg_row.cy is None:
            return None
        return (avg_row.cx * self.DEFAULT_METER_PER_PIXEL,
                avg_row.cy * self.DEFAULT_METER_PER_PIXEL)


    def _detect_shots_heuristic(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        goal_line: Dict
    ) -> List[Dict]:
        candidates = self._generate_shot_candidates(goals_df, ball_df, possession_df, goal_line)
        if candidates.count() == 0:
            return []

        threshold_m = self.NEAR_GOAL_CM_THRESHOLD / 100.0
        filtered = candidates.filter(
            (col("inside_shot_area") == True) | (col("dist_to_goal_line") <= threshold_m)
        )

        if goals_df.count() > 0:
            goal_frames = {row.frame_number for row in goals_df.select("frame_number").collect()}
            if goal_frames:
                filtered = filtered.filter(~col("ball_frame").isin(list(goal_frames)))

        w = Window.partitionBy("player_id").orderBy(
            col("inside_shot_area").desc(),
            col("dist_to_goal_line").asc()
        )
        best = filtered.withColumn("rn", row_number().over(w)).filter(col("rn") == 1)

        results = []
        for row in best.collect():
            results.append({
                "player_id": row.player_id,
                "track_id": row.track_id,
                "team_id": row.team_id,
                "shirt_number": row.shirt_number,
                "possession_frame": row.frame_number,
                "shot_frame": row.ball_frame,
                "dist_to_goal_line": row.dist_to_goal_line,
                "speed_kmh": row.ball_speed,
                "is_goal": False,
                "event_type": "near_goal_shot"
            })
        logfire.info(f"[GoalScorerDetector] Heuristic detected {len(results)} near-goal shots")
        return results


    def _update_player_goals_shots(
        self,
        match_id: int,
        goal_links: List[Dict],
        near_goal_shots: List[Dict],
        session: Session,
    ) -> None:

        player_event_counts: Dict[int, Dict] = {}

        for link in goal_links:
            pid = link["player_id"]
            if pid not in player_event_counts:
                player_event_counts[pid] = {"goals": 0, "shots": 0, "events": []}
            player_event_counts[pid]["goals"] += 1
            player_event_counts[pid]["events"].append(link)

        for shot in near_goal_shots:
            pid = shot["player_id"]
            if pid not in player_event_counts:
                player_event_counts[pid] = {"goals": 0, "shots": 0, "events": []}
            player_event_counts[pid]["shots"] += 1
            player_event_counts[pid]["events"].append(shot)

        for player_id, counts in player_event_counts.items():
            player = session.get(PlayerModel, player_id)
            if player:
                player.goals = int(counts["goals"])
                player.shots = int(counts["shots"])
                logfire.info(
                    f"[GoalScorerDetector] Player {player_id} (track {player.track_id}) "
                    f"+{counts['goals']} goals, +{counts['shots']} shots"
                )
                session.add(player)

        for link in goal_links:
            if link.get("is_goal"):
                stmt = (
                    select(PlayerState)
                    .where(PlayerState.player_id == link["player_id"])
                    .where(PlayerState.frame_number == link["possession_frame"])
                    .where(PlayerState.has_ball == True)
                )
                state = session.exec(stmt).first()
                if state:
                    state.is_goal = True
                    logfire.info(
                        f"[GoalScorerDetector] Marked PlayerState {state.id} as is_goal=True"
                    )

        team_goals_map = {}
        for link in goal_links:
            if link.get("is_goal"):
                tid = link["team_id"]
                team_goals_map[tid] = team_goals_map.get(tid, 0) + 1

        if team_goals_map:
            players = session.exec(
                select(PlayerModel).where(PlayerModel.match_id == match_id)
            ).all()
            for player in players:
                if player.team_id in team_goals_map:
                    player.team_goals = team_goals_map[player.team_id]
                    session.add(player)
            logfire.info(
                f"[GoalScorerDetector] Updated team_goals: {team_goals_map}"
            )

        session.commit()
        logfire.info(
            f"[GoalScorerDetector] DB updated: goals and shots assigned, team_goals set."
        )


goal_scorer_detector_cls = GoalScorerDetector()