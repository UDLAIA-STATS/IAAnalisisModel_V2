
from typing import Dict, List
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    lag,
    when,
    row_number,
    avg,
    abs as sql_abs,
    lit,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    BooleanType,
    StructType,
    StructField,
    StringType,
)
from sqlmodel import Session, select, col as modelcol

from src.entities.services.goal_scorer_detector_base import GoalScorerDetectorBase
from src.entities.models.soccer.goal_model import GoalModel
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.pyspark.udfs import (
    euclidean_distance_udf,
)

class GoalScorerDetector(GoalScorerDetectorBase):
    """
    Main goal scorer detector using PySpark for distributed processing.

    PlayerModel.goals is treated as "goals_shots" — incremented for both:
      - Actual goals (ball enters goal)
      - Near-goal shots (ball passes within NEAR_GOAL_CM_THRESHOLD of goal line)
    """

    def detect(self, match_id: int, total_frames: int, session: Session) -> None:
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

        goal_links = []
        if goals:
            goal_links = self._link_goals_to_players(goals_df, ball_df, possession_df)

        near_goal_shots = self._detect_near_goal_shots(
            goals_df, ball_df, possession_df, match_id, session
        )

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
        from src.entities.models.soccer.player_model import PlayerState, PlayerModel

        stmt = (
            select(PlayerState, PlayerModel)
            .join(PlayerModel, modelcol(PlayerState.player_id) == PlayerModel.id)
            .where(modelcol(PlayerState.player.match_id) == match_id)
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
            })
        return results

    # ==================================================================
    # Stage 2: Build DataFrames
    # ==================================================================

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

            # Use meter coordinates if available, otherwise fallback to pixels
            has_meters = b.dx_meters is not None and b.dy_meters is not None
            ball_mx = b.dx_meters if has_meters else cx * self.DEFAULT_METER_PER_PIXEL
            ball_my = b.dy_meters if has_meters else cy * self.DEFAULT_METER_PER_PIXEL

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
                "has_meters": has_meters,
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
        ])
        return self.spark.createDataFrame(rows, schema=schema)


    def _link_goals_to_players(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
    ) -> List[Dict]:
        if goals_df.count() == 0 or possession_df.count() == 0:
            return []

        candidates = goals_df.crossJoin(possession_df)

        filtered = (
            candidates
            .filter(col("frame_number") < col("goal_frame_number"))
            .filter(
                col("goal_frame_number") - col("frame_number") <= self.POSSESSION_LOOKBACK_FRAMES
            )
            .filter(
                col("goal_timestamp") - col("timestamp") <= self.POSSESSION_LOOKBACK_SECONDS
            )
        )

        filtered = filtered.withColumn(
            "ball_to_goal_dist_px",
            euclidean_distance_udf(col("ball_x"), col("ball_y"), col("goal_cx"), col("goal_cy")),
        )

        ball_at_goal = ball_df.select(
            col("frame_number").alias("ball_goal_frame"),
            col("ball_cx").alias("ball_at_goal_cx"),
            col("ball_cy").alias("ball_at_goal_cy"),
            col("ball_mx").alias("ball_at_goal_mx"),
            col("ball_my").alias("ball_at_goal_my"),
            col("speed_kmh").alias("ball_speed_at_goal"),
        )

        filtered = filtered.join(
            ball_at_goal,
            filtered.goal_frame_number == ball_at_goal.ball_goal_frame,
            "left",
        )

        filtered = filtered.withColumn(
            "ball_goal_proximity_m",
            when(
                col("ball_at_goal_mx").isNotNull() & col("ball_at_goal_my").isNotNull(),
                euclidean_distance_udf(
                    col("ball_at_goal_mx"), col("ball_at_goal_my"),
                    col("goal_cx") * self.DEFAULT_METER_PER_PIXEL,
                    col("goal_cy") * self.DEFAULT_METER_PER_PIXEL,
                ),
            ).otherwise(
                col("ball_to_goal_dist_px") * self.DEFAULT_METER_PER_PIXEL
            ),
        )

        # Score: lower = better match
        filtered = filtered.withColumn(
            "frame_gap", col("goal_frame_number") - col("frame_number"),
        ).withColumn(
            "possession_score",
            col("frame_gap") / self.POSSESSION_LOOKBACK_FRAMES * 0.30
            + col("ball_to_goal_dist_px") / 1000.0 * 0.25
            + col("ball_goal_proximity_m") / 10.0 * 0.25
            + (1.0 - col("confidence")) * 0.20,
        )

        w_goal = Window.partitionBy("goal_id").orderBy(col("possession_score").asc())
        ranked = filtered.withColumn("rn", row_number().over(w_goal))
        best = ranked.filter(col("rn") == 1).drop("rn")

        results = []
        for row in best.collect():
            results.append({
                "goal_id": row.goal_id,
                "player_id": row.player_id,
                "track_id": row.track_id,
                "team_id": row.team_id,
                "shirt_number": row.shirt_number,
                "possession_frame": row.frame_number,
                "goal_frame": row.goal_frame_number,
                "frame_gap": row.frame_gap,
                "ball_goal_proximity_m": row.ball_goal_proximity_m,
                "possession_score": row.possession_score,
                "is_goal": True,
                "event_type": "goal",
            })

        logfire.info(f"[GoalScorerDetector] Linked {len(results)} goals to players")
        return results


    def _detect_near_goal_shots(
        self,
        goals_df: DataFrame,
        ball_df: DataFrame,
        possession_df: DataFrame,
        match_id: int,
        session: Session,
    ) -> List[Dict]:
        """
        Detect shots where the ball passes within NEAR_GOAL_CM_THRESHOLD (centimeters)
        of the goal line/post, but no goal was scored.

        Uses meter coordinates when available; falls back to pixel scaling.
        """
        if possession_df.count() == 0 or ball_df.count() == 0:
            return []

        goal_frames = set()
        if goals_df.count() > 0:
            goal_frames = {row.frame_number for row in goals_df.select("frame_number").collect()}

        w_ball = Window.partitionBy("match_id").orderBy("frame_number")
        ball_traj = (
            ball_df
            .withColumn("prev_frame", lag("frame_number", 1).over(w_ball))
            .withColumn("prev_mx", lag("ball_mx", 1).over(w_ball))
            .withColumn("prev_my", lag("ball_my", 1).over(w_ball))
            .withColumn("prev_cx", lag("ball_cx", 1).over(w_ball))
            .withColumn("prev_cy", lag("ball_cy", 1).over(w_ball))
            .withColumn("frame_gap", col("frame_number") - col("prev_frame"))
        )

        goal_line_y_meters = self._estimate_goal_line_y(goals_df, ball_df)

        candidates = possession_df.crossJoin(ball_traj)

        filtered = (
            candidates
            .filter(col("ball_frame_number") > col("frame_number"))
            .filter(
                col("ball_frame_number") - col("frame_number") <= self.SHOT_LOOKAHEAD_FRAMES
            )
            .filter(
                col("ball_timestamp") - col("timestamp") <= self.SHOT_LOOKAHEAD_SECONDS
            )
            .filter(col("speed_kmh") >= self.MIN_SHOT_SPEED_KMH)
            .filter(col("speed_kmh") <= self.MAX_SHOT_SPEED_KMH)
        )

        if goal_frames:
            filtered = filtered.filter(~col("ball_frame_number").isin(list(goal_frames)))

        threshold_m = self.NEAR_GOAL_CM_THRESHOLD / 100.0

        filtered = filtered.withColumn(
            "dist_to_goal_line_m",
            when(
                col("has_meters") == True,
                sql_abs(col("ball_my") - lit(goal_line_y_meters)),
            ).otherwise(
                sql_abs(
                    col("ball_cy") * self.DEFAULT_METER_PER_PIXEL - lit(goal_line_y_meters)
                ),
            ),
        )

        near_goal = filtered.filter(col("dist_to_goal_line_m") <= threshold_m)

        near_goal = near_goal.withColumn(
            "moving_toward_goal",
            when(
                col("ball_my") < goal_line_y_meters,
                col("ball_my") - col("prev_my") > 0,
            ).when(
                col("ball_my") > goal_line_y_meters,
                col("ball_my") - col("prev_my") < 0,
            ).otherwise(False),
        ).filter(col("moving_toward_goal") == True)

        w_player = Window.partitionBy("player_id").orderBy(
            col("dist_to_goal_line_m").asc(),
            col("speed_kmh").desc(),
        )
        ranked = near_goal.withColumn("rn", row_number().over(w_player))
        best_shots = ranked.filter(col("rn") == 1).drop("rn")

        results = []
        for row in best_shots.collect():
            dist_cm = row.dist_to_goal_line_m * 100.0
            results.append({
                "player_id": row.player_id,
                "track_id": row.track_id,
                "team_id": row.team_id,
                "shirt_number": row.shirt_number,
                "possession_frame": row.frame_number,
                "shot_frame": row.ball_frame_number,
                "ball_cx": row.ball_cx,
                "ball_cy": row.ball_cy,
                "ball_mx": row.ball_mx,
                "ball_my": row.ball_my,
                "speed_kmh": row.speed_kmh,
                "dist_to_goal_cm": dist_cm,
                "is_goal": False,
                "event_type": "near_goal_shot",
            })
            logfire.info(
                f"[GoalScorerDetector] Near-goal shot by player {row.player_id} "
                f"at frame {row.ball_frame_number}: {dist_cm:.1f}cm from goal line"
            )

        logfire.info(f"[GoalScorerDetector] Detected {len(results)} near-goal shots")
        return results

    def _estimate_goal_line_y(
        self, goals_df: DataFrame, ball_df: DataFrame
    ) -> float:
        """
        Estimate the goal line Y position in meters.
        Uses the average ball_y (in meters) at goal frames.
        Fallback to standard field width ~68m.
        """
        if goals_df.count() == 0:
            return 68.0

        goals_with_ball = goals_df.join(
            ball_df.select(
                col("frame_number").alias("ball_frame"),
                col("ball_my"),
            ),
            goals_df.frame_number == ball_df.frame_number,
            "left",
        )

        avg_y = goals_with_ball.select(avg("ball_my")).collect()[0][0]
        if avg_y is not None and avg_y > 0:
            return float(avg_y)

        return 68.0


    def _update_player_goals_shots(
        self,
        match_id: int,
        goal_links: List[Dict],
        near_goal_shots: List[Dict],
        session: Session,
    ) -> None:
        """
        Update PlayerModel.goals (goals_shots) for both actual goals and near-goal shots.
        Mark PlayerState.is_goal=True for actual goal possessions.
        """
        from src.entities.models.soccer.player_model import PlayerModel, PlayerState

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

        total_goals = 0
        total_shots = 0

        for player_id, counts in player_event_counts.items():
            player = session.get(PlayerModel, player_id)
            if player:
                increment = counts["goals"] + counts["shots"]
                player.goals += increment
                total_goals += counts["goals"]
                total_shots += counts["shots"]
                logfire.info(
                    f"[GoalScorerDetector] Player {player_id} (track {player.track_id}) "
                    f"+{increment} goals_shots (goals={counts['goals']}, shots={counts['shots']}) "
                    f"-> total={player.goals}"
                )

        for link in goal_links:
            if not link.get("is_goal"):
                continue
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

        session.commit()
        logfire.info(
            f"[GoalScorerDetector] DB updated: {total_goals} goals + {total_shots} near-goal shots "
            f"assigned across {len(player_event_counts)} players"
        )


goal_scorer_detector_cls = GoalScorerDetector()