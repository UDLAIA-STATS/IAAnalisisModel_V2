from typing import List
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col,
    lag,
    lead,
    when,
    row_number,
    count,
    abs as sql_abs,
    lit,
    explode,
    expr,
)
from pyspark.sql.types import (
    FloatType,
    IntegerType,
    StructType,
    StructField,
)
from sqlmodel import Session, select, col as modelcol

from src.entities.predictors.ball_predictor_base import PredictorBase
from src.core.repository.ball_repository import BallRepository
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.pyspark.udfs import (
    euclidean_distance_udf,
    hypot_udf,
)



class BallPredictor(PredictorBase):
    """
    Main ball predictor using PySpark for distributed processing.

    Pipeline:
        1. Load ball states for match
        2. Compute movement metrics (dx, dy, dx_meters, dy_meters, speed, accel, velocity)
        3. Convert to Spark DataFrame
        4. Detect multi-ball frames and classify detections (active vs static vs replacement)
        5. Select the most probable real ball per frame
        6. Interpolate missing frames (up to MAX_INTERPOLATION_GAP)
        7. Sync back to database (delete losers, insert interpolated, update metrics)
    """

    def predict(self, match_id: int, total_frames: int, session: Session) -> None:
        logfire.info(f"[BallPredictor] Starting prediction for match {match_id}")

        states = list(BallRepository.get_balls_by_match_id(match_id, session))
        if not states:
            logfire.info("[BallPredictor] No ball states found for match")
            return

        states = self.compute_movement_metrics(states, match_id, session)
        logfire.info(f"[BallPredictor] Computed metrics for {len(states)} states")

        ball_df = self._build_ball_dataframe(states)
        if ball_df.count() == 0:
            logfire.info("[BallPredictor] Empty DataFrame after build")
            return

        classified = self._classify_ball_detections(ball_df)

        winners = self._select_winners_per_frame(classified)
        winner_count = winners.count()
        logfire.info(f"[BallPredictor] Selected {winner_count} winner balls")

        interpolated = self._interpolate_gaps(winners, total_frames)
        interp_count = interpolated.count() - winner_count
        logfire.info(f"[BallPredictor] Interpolated {interp_count} missing frames")

        self._sync_to_database(match_id, states, winners, interpolated, session)

        logfire.info(f"[BallPredictor] Completed for match {match_id}")


    def _build_ball_dataframe(self, states: List[BallState]) -> DataFrame:
        """Convert computed BallState objects to a Spark DataFrame."""
        rows = []
        for s in states:
            cx, cy = self._compute_centroid(s)
            area = max(0.0, (s.x2 or 0) - (s.x1 or 0)) * max(0.0, (s.y2 or 0) - (s.y1 or 0))
            rows.append({
                "id": s.id,
                "match_id": s.match_id,
                "frame_number": s.frame_number,
                "timestamp": s.timestamp,
                "confidence": s.confidence,
                "x1": s.x1,
                "y1": s.y1,
                "x2": s.x2,
                "y2": s.y2,
                "cx": cx,
                "cy": cy,
                "area": area,
                "dx": s.dx or 0.0,
                "dy": s.dy or 0.0,
                "dx_meters": s.dx_meters or 0.0,
                "dy_meters": s.dy_meters or 0.0,
                "distance_meters": s.distance_meters or 0.0,
                "speed_kmh": s.speed_kmh or 0.0,
                "vx": s.vx or 0.0,
                "vy": s.vy or 0.0,
                "ax": s.ax or 0.0,
                "ay": s.ay or 0.0,
                "acceleration": s.acceleration or 0.0,
            })

        schema = StructType([
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
        ])

        return self.spark.createDataFrame(rows, schema=schema)


    def _classify_ball_detections(self, ball_df: DataFrame) -> DataFrame:
        """
        Add classification columns:
            - is_static: ball not moving
            - is_replacement: sudden jump from predicted trajectory
            - is_erratic: speed/accel beyond physical limits
        """
        w = Window.partitionBy("match_id").orderBy("frame_number")

        df = (
            ball_df
            .withColumn("prev_cx", lag("cx", 1).over(w))
            .withColumn("prev_cy", lag("cy", 1).over(w))
            .withColumn("prev_vx", lag("vx", 1).over(w))
            .withColumn("prev_vy", lag("vy", 1).over(w))
            .withColumn("prev_frame", lag("frame_number", 1).over(w))
            .withColumn("frame_gap", col("frame_number") - col("prev_frame"))
        )

        df = df.withColumn(
            "pred_cx",
            when(
                col("prev_cx").isNotNull() & col("prev_vx").isNotNull(),
                col("prev_cx") + col("prev_vx") * col("frame_gap") / self.FRAME_RATE,
            ).otherwise(col("cx")),
        ).withColumn(
            "pred_cy",
            when(
                col("prev_cy").isNotNull() & col("prev_vy").isNotNull(),
                col("prev_cy") + col("prev_vy") * col("frame_gap") / self.FRAME_RATE,
            ).otherwise(col("cy")),
        )

        df = df.withColumn(
            "trajectory_dist",
            euclidean_distance_udf(col("cx"), col("cy"), col("pred_cx"), col("pred_cy")),
        )

        df = df.withColumn(
            "position_jump",
            euclidean_distance_udf(col("cx"), col("cy"), col("prev_cx"), col("prev_cy")),
        )

        df = (
            df.withColumn(
                "is_static",
                (col("speed_kmh") < self.STATIC_SPEED_THRESHOLD)
                & (sql_abs(col("dx")) < 1.0)
                & (sql_abs(col("dy")) < 1.0),
            )
            .withColumn(
                "is_replacement",
                (col("position_jump") > self.MAX_POSITION_JUMP_PX)
                & (col("trajectory_dist") > self.MAX_POSITION_JUMP_PX * 0.5),
            )
            .withColumn(
                "is_erratic",
                (col("speed_kmh") > self.MAX_BALL_SPEED_KMH)
                | (col("acceleration") > self.MAX_BALL_ACCEL_MSS),
            )
        )

        return df


    def _select_winners_per_frame(self, classified_df: DataFrame) -> DataFrame:
        """
        For each frame with multiple balls, select the most probable real one.
        Ranking: non-erratic > non-replacement > non-static > highest confidence > closest to trajectory
        """
        w_frame = Window.partitionBy("match_id", "frame_number").orderBy(
            col("is_erratic").asc(),      # False first
            col("is_replacement").asc(),  # False first
            col("is_static").asc(),       # False first (prefer moving balls)
            col("confidence").desc(),     # Highest confidence
            col("trajectory_dist").asc(), # Closest to predicted path
        )

        ranked = classified_df.withColumn("rn", row_number().over(w_frame))
        winners = ranked.filter(col("rn") == 1).drop("rn")

        multi_ball_frames = (
            classified_df.groupBy("match_id", "frame_number")
            .agg(count("*").alias("detection_count"))
            .filter(col("detection_count") > 1)
        )
        multi_count = multi_ball_frames.count()
        if multi_count > 0:
            logfire.info(f"[BallPredictor] Found {multi_count} multi-ball frames")

        return winners


    def _interpolate_gaps(self, winners_df: DataFrame, total_frames: int) -> DataFrame:
        """
        Identify gaps in the frame sequence and interpolate missing ball states.
        Only interpolates gaps <= MAX_INTERPOLATION_GAP.
        """
        w = Window.partitionBy("match_id").orderBy("frame_number")

        df = (
            winners_df
            .withColumn("next_frame", lead("frame_number", 1).over(w))
            .withColumn("next_cx", lead("cx", 1).over(w))
            .withColumn("next_cy", lead("cy", 1).over(w))
            .withColumn("next_x1", lead("x1", 1).over(w))
            .withColumn("next_y1", lead("y1", 1).over(w))
            .withColumn("next_x2", lead("x2", 1).over(w))
            .withColumn("next_y2", lead("y2", 1).over(w))
            .withColumn("next_timestamp", lead("timestamp", 1).over(w))
            .withColumn("next_vx", lead("vx", 1).over(w))
            .withColumn("next_vy", lead("vy", 1).over(w))
        )

        df = df.withColumn("gap_size", col("next_frame") - col("frame_number") - 1)

        df = df.filter(
            (col("gap_size").isNull()) | (col("gap_size") <= self.MAX_INTERPOLATION_GAP)
        )

        df_with_interp = df.withColumn(
            "interp_frames",
            when(
                (col("gap_size") > 0) & (col("gap_size").isNotNull()),
                expr(f"sequence(frame_number + 1, next_frame - 1)"),
            ).otherwise(expr("array()")),
        )

        exploded = df_with_interp.withColumn("interp_frame", explode(col("interp_frames")))

        exploded = exploded.withColumn(
            "ratio",
            (col("interp_frame") - col("frame_number")) / (col("next_frame") - col("frame_number")),
        )

        interpolated = (
            exploded
            .withColumn("interp_cx", col("cx") + (col("next_cx") - col("cx")) * col("ratio"))
            .withColumn("interp_cy", col("cy") + (col("next_cy") - col("cy")) * col("ratio"))
            .withColumn("interp_x1", col("x1") + (col("next_x1") - col("x1")) * col("ratio"))
            .withColumn("interp_y1", col("y1") + (col("next_y1") - col("y1")) * col("ratio"))
            .withColumn("interp_x2", col("x2") + (col("next_x2") - col("x2")) * col("ratio"))
            .withColumn("interp_y2", col("y2") + (col("next_y2") - col("y2")) * col("ratio"))
            .withColumn("interp_timestamp", col("timestamp") + (col("next_timestamp") - col("timestamp")) * col("ratio"))
            .withColumn("interp_vx", (col("next_cx") - col("cx")) * self.FRAME_RATE / (col("next_frame") - col("frame_number")))
            .withColumn("interp_vy", (col("next_cy") - col("cy")) * self.FRAME_RATE / (col("next_frame") - col("frame_number")))
        )

        interp_rows = (
            interpolated
            .select(
                lit(None).cast(IntegerType()).alias("id"),  # new record, no ID yet
                col("match_id"),
                col("interp_frame").alias("frame_number"),
                col("interp_timestamp").alias("timestamp"),
                # Interpolated confidence: average of neighbors * 0.9
                ((col("confidence") + lead("confidence", 1).over(w)) / 2.0 * 0.9).alias("confidence"),
                col("interp_x1").alias("x1"),
                col("interp_y1").alias("y1"),
                col("interp_x2").alias("x2"),
                col("interp_y2").alias("y2"),
                col("interp_cx").alias("cx"),
                col("interp_cy").alias("cy"),
                # Area from interpolated bbox
                ((col("interp_x2") - col("interp_x1")) * (col("interp_y2") - col("interp_y1"))).alias("area"),
                # dx, dy from displacement
                (col("interp_cx") - col("cx")).alias("dx"),
                (col("interp_cy") - col("cy")).alias("dy"),
                # dx_meters, dy_meters: scale by same ratio as pixels (approximation)
                (col("dx_meters") * col("ratio")).alias("dx_meters"),
                (col("dy_meters") * col("ratio")).alias("dy_meters"),
                lit(0.0).alias("distance_meters"),
                # Speed from interpolated velocity
                (hypot_udf(col("interp_vx"), col("interp_vy")) * 3.6).alias("speed_kmh"),
                col("interp_vx").alias("vx"),
                col("interp_vy").alias("vy"),
                lit(0.0).alias("ax"),
                lit(0.0).alias("ay"),
                lit(0.0).alias("acceleration"),
            )
        )

        # Union original winners with interpolated
        # First, select same columns from winners to match schema
        original_cols = [
            "id", "match_id", "frame_number", "timestamp", "confidence",
            "x1", "y1", "x2", "y2", "cx", "cy", "area",
            "dx", "dy", "dx_meters", "dy_meters", "distance_meters",
            "speed_kmh", "vx", "vy", "ax", "ay", "acceleration",
        ]
        original = winners_df.select(*[col(c) for c in original_cols])

        # Ensure interpolated has same schema
        for c in original_cols:
            if c not in interp_rows.columns:
                interp_rows = interp_rows.withColumn(c, lit(None))

        interp_rows = interp_rows.select(*[col(c) for c in original_cols])

        result = original.unionByName(interp_rows)
        return result.orderBy("frame_number")


    def _sync_to_database(
        self,
        match_id: int,
        original_states: List[BallState],
        winners_df: DataFrame,
        interpolated_df: DataFrame,
        session: Session,
    ) -> None:
        """
        Sync results back to database:
        - Delete non-winner ball states
        - Insert interpolated ball states
        - Update metrics on retained states
        """
        # Collect winner IDs
        winner_ids = {row.id for row in winners_df.filter(col("id").isNotNull()).select("id").collect()}
        original_ids = {s.id for s in original_states}
        to_delete = original_ids - winner_ids

        if to_delete:
            logfire.info(f"[BallPredictor] Deleting {len(to_delete)} non-winner ball states")
            for bid in to_delete:
                state = session.get(BallState, bid)
                if state:
                    session.delete(state)

        # Insert interpolated states
        interp_rows = interpolated_df.filter(col("id").isNull()).collect()
        inserted = 0
        for row in interp_rows:
            new_state = BallState(
                match_id=match_id,
                frame_number=row.frame_number,
                timestamp=row.timestamp,
                confidence=row.confidence if row.confidence is not None else 0.5,
                x1=row.x1,
                y1=row.y1,
                x2=row.x2,
                y2=row.y2,
                dx=row.dx,
                dy=row.dy,
                dx_meters=row.dx_meters,
                dy_meters=row.dy_meters,
                distance_meters=row.distance_meters,
                speed_kmh=row.speed_kmh,
                vx=row.vx,
                vy=row.vy,
                ax=row.ax,
                ay=row.ay,
                acceleration=row.acceleration,
            )
            session.add(new_state)
            inserted += 1

        # Update metrics on retained winner states
        winner_rows = {row.id: row for row in winners_df.filter(col("id").isNotNull()).collect()}
        updated = 0
        for state in original_states:
            if state.id in winner_rows:
                row = winner_rows[state.id]
                state.dx = row.dx
                state.dy = row.dy
                state.dx_meters = row.dx_meters
                state.dy_meters = row.dy_meters
                state.distance_meters = row.distance_meters
                state.speed_kmh = row.speed_kmh
                state.vx = row.vx
                state.vy = row.vy
                state.ax = row.ax
                state.ay = row.ay
                state.acceleration = row.acceleration
                updated += 1

        session.commit()
        logfire.info(
            f"[BallPredictor] Sync complete: {len(to_delete)} deleted, "
            f"{inserted} inserted, {updated} updated"
        )


ball_predictor_cls = BallPredictor()
