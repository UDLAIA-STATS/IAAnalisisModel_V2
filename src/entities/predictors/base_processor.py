from typing import List, Dict, Optional, Tuple, Any
import logfire
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, BooleanType, StringType
from pyspark.sql.functions import col, lit, when, sqrt, pow

from sqlmodel import Session, select, col as modelcol

from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.goal_model import GoalModel

from src.entities.models.pyspark.operations import (
    compute_centroid, compute_area, add_distance_to_goal_line,
)

from src.entities.utils import spark as spark_session


class BaseProcessor(ABC):
    """
    Clase base abstracta que proporciona métodos comunes para carga de datos,
    construcción de DataFrames, y sincronización con la base de datos.
    Las subclases deben implementar el método `process()`.
    """

    DEFAULT_METER_PER_PIXEL: float = 0.01
    FRAME_RATE: float = 30.0
    POSSESSION_LOOKBACK_FRAMES: int = 30
    POSSESSION_LOOKBACK_SECONDS: float = 1.0
    SHOT_LOOKAHEAD_FRAMES: int = 10
    SHOT_LOOKAHEAD_SECONDS: float = 0.5
    MIN_SHOT_SPEED_KMH: float = 10.0
    MAX_SHOT_SPEED_KMH: float = 120.0
    NEAR_GOAL_CM_THRESHOLD: float = 300.0  # cm
    MAX_INTERPOLATION_GAP: int = 5
    STATIC_SPEED_THRESHOLD: float = 0.5

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or spark_session

    def fetch_goals(self, match_id: int, session: Session) -> DataFrame:
        """Carga los goles/postes de la BD y los convierte a DataFrame."""
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._goal_schema())
        data = []
        for g in rows:
            cx, cy = compute_centroid(g.x1, g.y1, g.x2, g.y2)
            area = compute_area(g.x1, g.y1, g.x2, g.y2)
            data.append({
                "id": g.id,
                "match_id": g.match_id,
                "frame_number": g.frame_number,
                "timestamp": float(g.timestamp or 0.0),
                "confidence": float(g.confidence or 0.0),
                "x1": float(g.x1 or 0.0),
                "y1": float(g.y1 or 0.0),
                "x2": float(g.x2 or 0.0),
                "y2": float(g.y2 or 0.0),
                "goal_cx": cx,
                "goal_cy": cy,
                "goal_area": area,
            })
        return self.spark.createDataFrame(data, schema=self._goal_schema())

    def fetch_ball_states(self, match_id: int, session: Session) -> DataFrame:
        """Carga los estados del balón y los convierte a DataFrame con métricas."""
        stmt = (
            select(BallState)
            .where(BallState.match_id == match_id)
            .order_by(modelcol(BallState.frame_number))
        )
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._ball_schema())
        data = []
        for b in rows:
            cx, cy = compute_centroid(b.x1, b.y1, b.x2, b.y2)
            area = compute_area(b.x1, b.y1, b.x2, b.y2)
            data.append({
                "id": b.id,
                "match_id": b.match_id,
                "frame_number": b.frame_number,
                "timestamp": float(b.timestamp or 0.0),
                "confidence": float(b.confidence or 0.0),
                "x1": float(b.x1 or 0.0),
                "y1": float(b.y1 or 0.0),
                "x2": float(b.x2 or 0.0),
                "y2": float(b.y2 or 0.0),
                "ball_cx": cx,
                "ball_cy": cy,
                "ball_area": area,
                "ball_mx": cx * self.DEFAULT_METER_PER_PIXEL,
                "ball_my": cy * self.DEFAULT_METER_PER_PIXEL,
                "dx": float(b.dx or 0.0),
                "dy": float(b.dy or 0.0),
                "dx_meters": float(b.dx_meters or 0.0),
                "dy_meters": float(b.dy_meters or 0.0),
                "distance_meters": float(b.distance_meters or 0.0),
                "speed_kmh": float(b.speed_kmh or 0.0),
                "vx": float(b.vx or 0.0),
                "vy": float(b.vy or 0.0),
                "ax": float(b.ax or 0.0),
                "ay": float(b.ay or 0.0),
                "acceleration": float(b.acceleration or 0.0),
            })
        return self.spark.createDataFrame(data, schema=self._ball_schema())

    def fetch_player_states(self, match_id: int, session: Session) -> DataFrame:
        """Carga los estados de jugadores y los combina con PlayerModel."""
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(PlayerModel.match_id == match_id)
            .order_by(modelcol(PlayerState.frame_number))
        )
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._player_schema())
        data = []
        for state, player in rows:
            cx, cy = compute_centroid(state.x1, state.y1, state.x2, state.y2)
            area = compute_area(state.x1, state.y1, state.x2, state.y2)
            data.append({
                "player_state_id": state.id,
                "player_id": player.id,
                "track_id": player.track_id,
                "team_id": player.team_id,
                "team_color": player.team_color,
                "shirt_number": player.shirt_number,
                "frame_number": state.frame_number,
                "timestamp": float(state.timestamp or 0.0),
                "player_x1": float(state.x1 or 0.0),
                "player_y1": float(state.y1 or 0.0),
                "player_x2": float(state.x2 or 0.0),
                "player_y2": float(state.y2 or 0.0),
                "player_cx": cx,
                "player_cy": cy,
                "player_area": area,
                "player_confidence": float(state.confidence or 0.0),
                "has_ball": state.has_ball or False,
                "player_speed": float(state.speed_kmh or 0.0),
                "vx": float(state.vx or 0.0),
                "vy": float(state.vy or 0.0),
                "ball_x": float(state.ball_x or 0.0),
                "ball_y": float(state.ball_y or 0.0),
            })
        return self.spark.createDataFrame(data, schema=self._player_schema())

    def fetch_possessions(self, match_id: int, session: Session) -> DataFrame:
        """
        Carga posesiones explícitas (donde has_ball=True) de PlayerState.
        Similar a fetch_player_states pero solo con posesión.
        """
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(PlayerModel.match_id == match_id)
            .where(PlayerState.has_ball == True)
            .order_by(modelcol(PlayerState.frame_number))
        )
        rows = session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._possession_schema())
        data = []
        for state, player in rows:
            cx, cy = compute_centroid(state.x1, state.y1, state.x2, state.y2)
            data.append({
                "possession_id": state.id,
                "player_id": player.id,
                "track_id": player.track_id,
                "team_id": player.team_id,
                "team_color": player.team_color,
                "shirt_number": player.shirt_number,
                "frame_number": state.frame_number,
                "timestamp": float(state.timestamp or 0.0),
                "player_x1": float(state.x1 or 0.0),
                "player_y1": float(state.y1 or 0.0),
                "player_x2": float(state.x2 or 0.0),
                "player_y2": float(state.y2 or 0.0),
                "player_cx": cx,
                "player_cy": cy,
                "player_confidence": float(state.confidence or 0.0),
                "ball_x": float(state.ball_x or 0.0),
                "ball_y": float(state.ball_y or 0.0),
            })
        return self.spark.createDataFrame(data, schema=self._possession_schema())


    @staticmethod
    def _goal_schema() -> StructType:
        return StructType([
            StructField("id", IntegerType(), True),
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

    @staticmethod
    def _ball_schema() -> StructType:
        return StructType([
            StructField("id", IntegerType(), True),
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
            StructField("dx", FloatType(), True),
            StructField("dy", FloatType(), True),
            StructField("dx_meters", FloatType(), True),
            StructField("dy_meters", FloatType(), True),
            StructField("distance_meters", FloatType(), True),
            StructField("speed_kmh", FloatType(), True),
            StructField("vx", FloatType(), True),
            StructField("vy", FloatType(), True),
            StructField("ax", FloatType(), True),
            StructField("ay", FloatType(), True),
            StructField("acceleration", FloatType(), True),
        ])

    @staticmethod
    def _player_schema() -> StructType:
        return StructType([
            StructField("player_state_id", IntegerType(), True),
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
            StructField("player_area", FloatType(), True),
            StructField("player_confidence", FloatType(), True),
            StructField("has_ball", BooleanType(), True),
            StructField("player_speed", FloatType(), True),
            StructField("vx", FloatType(), True),
            StructField("vy", FloatType(), True),
            StructField("ball_x", FloatType(), True),
            StructField("ball_y", FloatType(), True),
        ])

    @staticmethod
    def _possession_schema() -> StructType:
        return StructType([
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
            StructField("player_confidence", FloatType(), True),
            StructField("ball_x", FloatType(), True),
            StructField("ball_y", FloatType(), True),
        ])

    def add_ball_goal_distances(
        self,
        df: DataFrame,
        goal_line: Optional[Dict] = None,
        goal_centroid: Optional[Tuple[float, float]] = None
    ) -> DataFrame:
        """
        Añade columnas de distancia a la línea de gol y al centro de la portería.
        Se espera que el DataFrame tenga columnas 'ball_mx' y 'ball_my' (metros).
        """
        if goal_line is not None:
            df = add_distance_to_goal_line(df, goal_line, "ball_mx", "ball_my",
                                           output_col="dist_to_goal_line")
        if goal_centroid is not None:
            gx, gy = goal_centroid
            # Convertir centro de portería a metros si no lo está
            gx_m = gx * self.DEFAULT_METER_PER_PIXEL
            gy_m = gy * self.DEFAULT_METER_PER_PIXEL
            df = df.withColumn(
                "dist_to_goal_center",
                sqrt(pow(col("ball_mx") - lit(gx_m), 2) + pow(col("ball_my") - lit(gy_m), 2))
            )
        return df

    def add_player_ball_features(self, player_df: DataFrame, ball_df: DataFrame) -> DataFrame:
        """
        Combina jugadores y balón por frame y añade distancia, ángulo, etc.
        Retorna un DataFrame con columnas adicionales.
        """
        # Renombrar para evitar conflictos
        ball_renamed = ball_df.select(
            col("frame_number"),
            col("ball_cx").alias("bx"),
            col("ball_cy").alias("by"),
            col("ball_mx").alias("bmx"),
            col("ball_my").alias("bmy"),
            col("vx").alias("bvx"),
            col("vy").alias("bvy"),
            col("speed_kmh").alias("ball_speed"),
            col("confidence").alias("ball_conf")
        )
        joined = player_df.join(ball_renamed, on="frame_number", how="inner")
        joined = joined.withColumn(
            "dist_pb",
            sqrt(pow(col("player_cx") - col("bx"), 2) + pow(col("player_cy") - col("by"), 2))
        )
        joined = joined.withColumn(
            "dist_pb_m",
            sqrt(pow(col("player_cx")*self.DEFAULT_METER_PER_PIXEL - col("bmx"), 2) +
                 pow(col("player_cy")*self.DEFAULT_METER_PER_PIXEL - col("bmy"), 2))
        )
        # Ángulo entre velocidad del jugador y vector al balón
        joined = joined.withColumn(
            "to_ball_x", col("bmx") - col("player_cx")*self.DEFAULT_METER_PER_PIXEL
        ).withColumn(
            "to_ball_y", col("bmy") - col("player_cy")*self.DEFAULT_METER_PER_PIXEL
        ).withColumn(
            "player_speed_norm", sqrt(pow(col("vx"), 2) + pow(col("vy"), 2))
        ).withColumn(
            "dot_to_ball",
            col("vx") * col("to_ball_x") + col("vy") * col("to_ball_y")
        ).withColumn(
            "angle_cos",
            when(
                (col("player_speed_norm") > 0) & (col("dist_pb_m") > 0),
                col("dot_to_ball") / (col("player_speed_norm") * col("dist_pb_m"))
            ).otherwise(0.0)
        )
        return joined


    def update_ball_states(
        self,
        match_id: int,
        ball_df: DataFrame,
        session: Session,
        delete_non_winners: bool = False,
        winner_ids: Optional[List[int]] = None
    ) -> None:
        """
        Sincroniza el DataFrame de balón con la BD.
        Si delete_non_winners es True, elimina los estados no presentes en winner_ids.
        """
        # Actualizar estados existentes
        rows = ball_df.collect()
        for row in rows:
            state = session.get(BallState, row.id) if row.id is not None else None
            if state:
                # Actualizar campos
                for field in ["x1", "y1", "x2", "y2", "confidence", "dx", "dy",
                              "dx_meters", "dy_meters", "distance_meters",
                              "speed_kmh", "vx", "vy", "ax", "ay", "acceleration"]:
                    if hasattr(row, field):
                        setattr(state, field, float(getattr(row, field)))
                session.add(state)

        # Insertar nuevos estados (id is None)
        new_rows = ball_df.filter(col("id").isNull()).collect()
        for row in new_rows:
            new_state = BallState(
                match_id=match_id,
                frame_number=row.frame_number,
                timestamp=row.timestamp if row.timestamp is not None else 0.0,
                confidence=row.confidence if row.confidence is not None else 0.5,
                x1=float(row.x1),
                y1=float(row.y1),
                x2=float(row.x2),
                y2=float(row.y2),
                dx=float(row.dx),
                dy=float(row.dy),
                dx_meters=float(row.dx_meters),
                dy_meters=float(row.dy_meters),
                distance_meters=float(row.distance_meters),
                speed_kmh=float(row.speed_kmh),
                vx=float(row.vx),
                vy=float(row.vy),
                ax=float(row.ax),
                ay=float(row.ay),
                acceleration=float(row.acceleration),
            )
            session.add(new_state)

        if delete_non_winners and winner_ids is not None:
            # Eliminar estados que no están en winner_ids
            stmt = select(BallState).where(BallState.match_id == match_id)
            all_states = session.exec(stmt).all()
            for state in all_states:
                if state.id not in winner_ids:
                    session.delete(state)

        session.commit()
        logfire.info(f"Ball states updated for match {match_id}")

    def update_player_stats(
        self,
        match_id: int,
        session: Session,
        possession_time_df: Optional[DataFrame] = None,
        shot_counts_df: Optional[DataFrame] = None,
        goal_counts: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Actualiza las estadísticas de jugadores (tiempo de posesión, disparos, goles).
        """
        # Obtener jugadores del partido
        stmt = select(PlayerModel).where(PlayerModel.match_id == match_id)
        players = session.exec(stmt).all()
        if not players:
            return

        # Convertir DataFrames a diccionarios
        pos_dict = {}
        if possession_time_df is not None:
            for row in possession_time_df.collect():
                pos_dict[row.player_id] = row.possession_time

        shot_dict = {}
        if shot_counts_df is not None:
            for row in shot_counts_df.collect():
                shot_dict[row.player_id] = row.shots

        goal_dict = goal_counts or {}

        for player in players:
            if player.id in pos_dict:
                player.ball_possession_time = int(pos_dict[player.id])
            if player.id in shot_dict:
                player.shots = int(shot_dict[player.id])
            if player.id in goal_dict:
                player.goals = int(goal_dict[player.id])
            session.add(player)

        session.commit()
        logfire.info(f"Player stats updated for match {match_id}")

    def update_goals(self, match_id: int, goals_df: DataFrame, session: Session) -> None:
        """
        Reemplaza la lista de goles/postes por los registros del DataFrame.
        (Elimina los antiguos y crea nuevos).
        """
        # Eliminar goles existentes
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        existing = session.exec(stmt).all()
        for g in existing:
            session.delete(g)

        # Insertar nuevos
        rows = goals_df.collect()
        for row in rows:
            goal = GoalModel(
                match_id=match_id,
                frame_number=row.frame_number,
                timestamp=row.timestamp if row.timestamp is not None else 0.0,
                confidence=row.confidence if row.confidence is not None else 0.5,
                x1=float(row.x1),
                y1=float(row.y1),
                x2=float(row.x2),
                y2=float(row.y2),
            )
            session.add(goal)

        session.commit()
        logfire.info(f"Goals updated for match {match_id}")

    @abstractmethod
    def predict(self, match_id: int, total_frames: int, fps: float, session: Session) -> None:
        """
        Método principal que debe implementar cada procesador.
        """
        pass