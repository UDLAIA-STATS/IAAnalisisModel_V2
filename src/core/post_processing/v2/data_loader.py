from typing import Tuple
from pyspark.sql import DataFrame
from sqlmodel import Session, select, col as modelcol
import logfire

from src.entities.models.soccer.ball_model import BallState
from src.entities.models.soccer.player_model import PlayerModel, PlayerState
from src.entities.models.soccer.goal_model import GoalModel
from src.entities.utils.spark_instance import spark


class DataLoader:
    """Carga todos los datos necesarios en DataFrames de PySpark."""

    def __init__(self, session: Session):
        self.session = session
        self.spark = spark

    def load_all(
        self, match_id: int
    ) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """Carga todos los DataFrames y los devuelve en un diccionario."""
        logfire.info(f"[DataLoader] Loading data for match {match_id}")
        return (
            self.load_ball_states(match_id),
            self.load_player_states(match_id),
            self.load_goal_posts(match_id),
            self.load_players(match_id),
        )

    def load_ball_states(self, match_id: int) -> DataFrame:
        stmt = (
            select(BallState)
            .where(BallState.match_id == match_id)
            .order_by(modelcol(BallState.frame_number), modelcol(BallState.timestamp))
        )
        rows = self.session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._ball_schema())
        data = [
            {
                "id": b.id,
                "match_id": b.match_id,
                "frame_number": b.frame_number,
                "timestamp": float(b.timestamp or 0.0),
                "confidence": float(b.confidence or 0.0),
                "x1": float(b.x1 or 0.0),
                "y1": float(b.y1 or 0.0),
                "x2": float(b.x2 or 0.0),
                "y2": float(b.y2 or 0.0),
                "cx": (
                    (b.x1 + b.x2) / 2.0
                    if b.x1 is not None and b.x2 is not None
                    else 0.0
                ),
                "cy": (
                    (b.y1 + b.y2) / 2.0
                    if b.y1 is not None and b.y2 is not None
                    else 0.0
                ),
                "area": max(0.0, (b.x2 or 0) - (b.x1 or 0))
                * max(0.0, (b.y2 or 0) - (b.y1 or 0)),
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
            }
            for b in rows
        ]
        schema = self._ball_schema()
        df = self.spark.createDataFrame(data, schema=schema)
        logfire.info(f"[DataLoader] Loaded {df.count()} ball states")
        return df

    def load_player_states(self, match_id: int) -> DataFrame:
        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(modelcol(PlayerModel.match_id) == match_id)
            .order_by(
                modelcol(PlayerState.frame_number), modelcol(PlayerState.timestamp)
            )
        )
        rows = self.session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._player_schema())
        data = []
        for state, player in rows:
            cx = (
                (state.x1 + state.x2) / 2.0
                if state.x1 is not None and state.x2 is not None
                else 0.0
            )
            cy = (
                (state.y1 + state.y2) / 2.0
                if state.y1 is not None and state.y2 is not None
                else 0.0
            )
            area = max(0.0, (state.x2 or 0) - (state.x1 or 0)) * max(
                0.0, (state.y2 or 0) - (state.y1 or 0)
            )
            data.append(
                {
                    "player_state_id": state.id,
                    "player_id": player.id,
                    "track_id": player.track_id,
                    "team_id": player.team_id,
                    "frame_number": state.frame_number,
                    "timestamp": float(state.timestamp or 0.0),
                    "x1": float(state.x1 or 0.0),
                    "y1": float(state.y1 or 0.0),
                    "x2": float(state.x2 or 0.0),
                    "y2": float(state.y2 or 0.0),
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                    "confidence": float(state.confidence or 0.0),
                    "has_ball": state.has_ball if state.has_ball is not None else False,
                    "vx": float(state.vx or 0.0),
                    "vy": float(state.vy or 0.0),
                    "speed_kmh": float(state.speed_kmh or 0.0),
                    "ball_x": float(state.ball_x or 0.0),
                    "ball_y": float(state.ball_y or 0.0),
                    "shirt_number": player.shirt_number,
                }
            )
        schema = self._player_schema()
        df = self.spark.createDataFrame(data, schema=schema)
        logfire.info(f"[DataLoader] Loaded {df.count()} player states")
        return df

    def load_goal_posts(self, match_id: int) -> DataFrame:
        stmt = select(GoalModel).where(GoalModel.match_id == match_id)
        rows = self.session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._goal_schema())
        data = []
        for g in rows:
            cx = (g.x1 + g.x2) / 2.0 if g.x1 is not None and g.x2 is not None else 0.0
            cy = (g.y1 + g.y2) / 2.0 if g.y1 is not None and g.y2 is not None else 0.0
            area = max(0.0, (g.x2 or 0) - (g.x1 or 0)) * max(
                0.0, (g.y2 or 0) - (g.y1 or 0)
            )
            data.append(
                {
                    "id": g.id,
                    "match_id": g.match_id,
                    "frame_number": g.frame_number,
                    "timestamp": float(g.timestamp or 0.0),
                    "confidence": float(g.confidence or 0.0),
                    "x1": float(g.x1 or 0.0),
                    "y1": float(g.y1 or 0.0),
                    "x2": float(g.x2 or 0.0),
                    "y2": float(g.y2 or 0.0),
                    "cx": cx,
                    "cy": cy,
                    "area": area,
                }
            )
        schema = self._goal_schema()
        df = self.spark.createDataFrame(data, schema=schema)
        logfire.info(f"[DataLoader] Loaded {df.count()} goal posts")
        return df

    def load_players(self, match_id: int) -> DataFrame:
        stmt = select(PlayerModel).where(PlayerModel.match_id == match_id)
        rows = self.session.exec(stmt).all()
        if not rows:
            return self.spark.createDataFrame([], self._player_info_schema())
        data = [
            {
                "id": p.id,
                "track_id": p.track_id,
                "team_id": p.team_id,
                "shirt_number": p.shirt_number,
                "goals": p.goals or 0,
                "shots": p.shots or 0,
                "ball_possession_time": p.ball_possession_time or 0.0,
                "team_goals": p.team_goals or 0,
            }
            for p in rows
        ]
        schema = self._player_info_schema()
        df = self.spark.createDataFrame(data, schema=schema)
        logfire.info(f"[DataLoader] Loaded {df.count()} player infos")
        return df

    @staticmethod
    def _ball_schema():
        from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

        return StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("match_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("confidence", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("cx", FloatType(), True),
                StructField("cy", FloatType(), True),
                StructField("area", FloatType(), True),
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
            ]
        )

    @staticmethod
    def _player_schema():
        from pyspark.sql.types import (
            StructType,
            StructField,
            IntegerType,
            FloatType,
            BooleanType,
        )

        return StructType(
            [
                StructField("player_state_id", IntegerType(), True),
                StructField("player_id", IntegerType(), True),
                StructField("track_id", IntegerType(), True),
                StructField("team_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("cx", FloatType(), True),
                StructField("cy", FloatType(), True),
                StructField("area", FloatType(), True),
                StructField("confidence", FloatType(), True),
                StructField("has_ball", BooleanType(), True),
                StructField("vx", FloatType(), True),
                StructField("vy", FloatType(), True),
                StructField("speed_kmh", FloatType(), True),
                StructField("ball_x", FloatType(), True),
                StructField("ball_y", FloatType(), True),
                StructField("shirt_number", IntegerType(), True),
            ]
        )

    @staticmethod
    def _goal_schema():
        from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

        return StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("match_id", IntegerType(), True),
                StructField("frame_number", IntegerType(), True),
                StructField("timestamp", FloatType(), True),
                StructField("confidence", FloatType(), True),
                StructField("x1", FloatType(), True),
                StructField("y1", FloatType(), True),
                StructField("x2", FloatType(), True),
                StructField("y2", FloatType(), True),
                StructField("cx", FloatType(), True),
                StructField("cy", FloatType(), True),
                StructField("area", FloatType(), True),
            ]
        )

    @staticmethod
    def _player_info_schema():
        from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

        return StructType(
            [
                StructField("id", IntegerType(), True),
                StructField("track_id", IntegerType(), True),
                StructField("team_id", IntegerType(), True),
                StructField("shirt_number", IntegerType(), True),
                StructField("goals", IntegerType(), True),
                StructField("shots", IntegerType(), True),
                StructField("ball_possession_time", FloatType(), True),
                StructField("team_goals", IntegerType(), True),
            ]
        )
