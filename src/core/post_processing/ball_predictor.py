from typing import List, Optional, Tuple
import logfire

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, lit, when, row_number, count, lag, lead, 
    abs as sql_abs, expr, explode, sqrt, pow, coalesce,
)
from pyspark.sql.types import (
    FloatType, IntegerType, StructType, StructField
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml import Pipeline, PipelineModel

from sqlmodel import Session

from src.entities.predictors.ball_predictor_base import PredictorBase
from src.core.repository.ball_repository import BallRepository
from src.entities.models.soccer.ball_model import BallState
from src.entities.models.pyspark.udfs import (
    euclidean_distance_udf,
)


class BallPredictor(PredictorBase):
    """
    Ball predictor with ML-enhanced trajectory estimation and multi-ball cleaning.
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

        traj_model_cx, traj_model_cy = self._train_trajectory_model(ball_df)
        if traj_model_cx is not None and traj_model_cy is not None:
            traj_df = self._predict_trajectory(traj_model_cx, traj_model_cy, total_frames, match_id)
            logfire.info("[BallPredictor] Trajectory model trained and applied.")
        else:
            traj_df = None
            logfire.warning("[BallPredictor] Trajectory model not available; falling back to heuristics.")

        classified = self._classify_ball_detections(ball_df)

        if traj_df is not None:
            winners = self._select_winners_per_frame(classified, traj_df)
        else:
            winners = self._select_winners_per_frame(classified, None)

        winner_count = winners.count()
        logfire.info(f"[BallPredictor] Selected {winner_count} winner balls")

        interpolated = self._interpolate_gaps(winners, total_frames, traj_df)
        interp_count = interpolated.count() - winner_count
        logfire.info(f"[BallPredictor] Interpolated {interp_count} missing frames")

        self._sync_to_database(match_id, states, winners, interpolated, session)

        logfire.info(f"[BallPredictor] Completed for match {match_id}")


    def _train_trajectory_model(self, ball_df: DataFrame) -> Tuple[Optional[PipelineModel], Optional[PipelineModel]]:
        """
        Train two RandomForestRegressor models (for cx and cy) using frame_number and frame_number^2.
        Returns (model_cx, model_cy) or (None, None) if insufficient data.
        """
        train_df = ball_df.select("frame_number", "cx", "cy").na.drop()
        if train_df.count() < 10:
            logfire.warning("[BallPredictor] Not enough data to train trajectory model (need at least 10 frames).")
            return None, None

        train_df = train_df.withColumn("frame2", col("frame_number") ** 2)
        feature_cols = ["frame_number", "frame2"]

        def _train_single(label_col: str) -> PipelineModel:
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            rf = RandomForestRegressor(
                featuresCol="features",
                labelCol=label_col,
                numTrees=50,
                maxDepth=6,
                seed=42,
                subsamplingRate=0.8
            )
            pipeline = Pipeline(stages=[assembler, rf])
            return pipeline.fit(train_df)

        model_cx = _train_single("cx")
        model_cy = _train_single("cy")
        logfire.info("[BallPredictor] Trajectory models trained successfully.")
        return model_cx, model_cy

    def _predict_trajectory(self, model_cx: PipelineModel, model_cy: PipelineModel,
                            total_frames: int, match_id: int) -> DataFrame:
        """
        Generate predictions for all frames from 0 to total_frames-1.
        Returns DataFrame with columns: match_id, frame_number, pred_cx, pred_cy.
        """
        frames = [(match_id, i, i * i) for i in range(total_frames)]
        schema = StructType([
            StructField("match_id", IntegerType(), True),
            StructField("frame_number", IntegerType(), True),
            StructField("frame2", IntegerType(), True),
        ])
        all_frames_df = self.spark.createDataFrame(frames, schema)

        pred_cx = model_cx.transform(all_frames_df).select(
            col("match_id"),
            col("frame_number"),
            col("prediction").alias("pred_cx")
        )
        pred_cy = model_cy.transform(all_frames_df).select(
            col("match_id"),
            col("frame_number"),
            col("prediction").alias("pred_cy")
        )

        traj_df = pred_cx.join(pred_cy, on=["match_id", "frame_number"], how="inner")
        return traj_df

    def _build_ball_dataframe(self, states: List[BallState]) -> DataFrame:
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

        # Predicted position based on previous velocity (linear extrapolation)
        df = df.withColumn(
            "pred_cx_from_prev",
            when(
                col("prev_cx").isNotNull() & col("prev_vx").isNotNull(),
                col("prev_cx") + col("prev_vx") * col("frame_gap") / self.FRAME_RATE,
            ).otherwise(col("cx"))
        ).withColumn(
            "pred_cy_from_prev",
            when(
                col("prev_cy").isNotNull() & col("prev_vy").isNotNull(),
                col("prev_cy") + col("prev_vy") * col("frame_gap") / self.FRAME_RATE,
            ).otherwise(col("cy"))
        )

        df = df.withColumn(
            "trajectory_dist",
            euclidean_distance_udf(col("cx"), col("cy"), col("pred_cx_from_prev"), col("pred_cy_from_prev"))
        )

        df = df.withColumn(
            "position_jump",
            euclidean_distance_udf(col("cx"), col("cy"), col("prev_cx"), col("prev_cy"))
        )

        df = (
            df.withColumn(
                "is_static",
                (col("speed_kmh") < self.STATIC_SPEED_THRESHOLD)
                & (sql_abs(col("dx")) < 1.0)
                & (sql_abs(col("dy")) < 1.0)
            )
            .withColumn(
                "is_replacement",
                (col("position_jump") > self.MAX_POSITION_JUMP_PX)
                & (col("trajectory_dist") > self.MAX_POSITION_JUMP_PX * 0.5)
            )
            .withColumn(
                "is_erratic",
                (col("speed_kmh") > self.MAX_BALL_SPEED_KMH)
                | (col("acceleration") > self.MAX_BALL_ACCEL_MSS)
            )
        )

        return df

    def _select_winners_per_frame(self, classified_df: DataFrame, traj_df: Optional[DataFrame]) -> DataFrame:
        """
        For each frame, choose the best ball candidate.
        If traj_df is provided, add distance to predicted trajectory as a criterion.
        """
        if traj_df is not None:
            df = classified_df.join(traj_df, on=["match_id", "frame_number"], how="left")
            df = df.withColumn("pred_cx", coalesce(col("pred_cx"), col("cx"))) \
                   .withColumn("pred_cy", coalesce(col("pred_cy"), col("cy")))
            df = df.withColumn(
                "dist_to_traj",
                euclidean_distance_udf(col("cx"), col("cy"), col("pred_cx"), col("pred_cy"))
            )
        else:
            df = classified_df.withColumn("dist_to_traj", lit(0.0))

        w_frame = Window.partitionBy("match_id", "frame_number").orderBy(
            col("is_erratic").asc(),
            col("is_replacement").asc(),
            col("is_static").asc(),
            col("confidence").desc(),
            col("dist_to_traj").asc()
        )

        ranked = df.withColumn("rn", row_number().over(w_frame))
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

    # --------------------------------------------------------------------
    #  Interpolation with trajectory for large gaps
    # --------------------------------------------------------------------

    def _interpolate_gaps(self, winners_df: DataFrame, total_frames: int,
                          traj_df: Optional[DataFrame]) -> DataFrame:
        """
        Fill missing frames using linear interpolation for small gaps,
        and using trajectory predictions for large gaps (if available).
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

        # For gaps <= MAX_INTERPOLATION_GAP, interpolate linearly
        df_small_gap = df.filter(
            (col("gap_size").isNull()) | (col("gap_size") <= self.MAX_INTERPOLATION_GAP)
        )

        df_small_gap = df_small_gap.withColumn(
            "interp_frames",
            when(
                (col("gap_size") > 0) & (col("gap_size").isNotNull()),
                expr(f"sequence(frame_number + 1, next_frame - 1)")
            ).otherwise(expr("array()"))
        )

        exploded = df_small_gap.withColumn("interp_frame", explode(col("interp_frames")))

        exploded = exploded.withColumn(
            "ratio",
            (col("interp_frame") - col("frame_number")) / (col("next_frame") - col("frame_number"))
        )

        interpolated_small = (
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

        # Prepare interpolated rows for small gaps
        interp_small_rows = self._prepare_interpolated_rows(interpolated_small, is_linear=True)

        # For large gaps, if trajectory model is available, use predictions
        if traj_df is not None:
            # Identify gaps > MAX_INTERPOLATION_GAP
            df_large_gap = df.filter(
                (col("gap_size") > self.MAX_INTERPOLATION_GAP) & (col("gap_size").isNotNull())
            )
            # We need to generate all frames between start and end, and use trajectory prediction for each.
            # This requires expanding the range. We'll use sequence again.
            df_large_gap = df_large_gap.withColumn(
                "interp_frames",
                expr(f"sequence(frame_number + 1, next_frame - 1)")
            )
            exploded_large = df_large_gap.withColumn("interp_frame", explode(col("interp_frames")))

            # Join with trajectory predictions to get predicted cx, cy for those frames
            interp_large = exploded_large.select("match_id", "interp_frame", "frame_number", "next_frame")
            interp_large = interp_large.join(
                traj_df,
                (interp_large["match_id"] == traj_df["match_id"]) &
                (interp_large["interp_frame"] == traj_df["frame_number"]),
                how="left"
            )
            # Also get the start and end values to set confidence, etc.
            # We'll use the predictions directly, but we need to fill other fields (x1,y1, etc.)
            # We can approximate bbox from cx,cy using average size from nearby real detections.
            # For simplicity, we'll create a synthetic bbox with a default size (e.g., 20x20 px)
            # and set confidence to 0.5 (medium).
            # Better: use the size from the nearest real detection.

            # We'll join with the original winners to get the size (x1,y1,x2,y2) of the start/end.
            start_cols = df_large_gap.select(
                col("match_id"), col("frame_number"), col("x1"), col("y1"), col("x2"), col("y2"),
                col("cx"), col("cy"), col("timestamp"), col("confidence"), col("id")
            ).withColumnRenamed("frame_number", "start_frame")
            end_cols = df_large_gap.select(
                col("match_id"), col("next_frame"), col("next_x1").alias("x1"),
                col("next_y1").alias("y1"), col("next_x2").alias("x2"),
                col("next_y2").alias("y2"), col("next_cx").alias("cx"),
                col("next_cy").alias("cy"), col("next_timestamp").alias("timestamp"),
                col("next_vx").alias("vx"), col("next_vy").alias("vy")
            ).withColumnRenamed("next_frame", "end_frame")

            # Combine start and end into a single row per gap, then join with interp_large to assign sizes.
            # This gets complicated. A simpler approach: use the trajectory prediction as the centroid,
            # and compute bbox as a fixed size (e.g., 20x20) for interpolation rows.
            # This is a simplification acceptable for filling large gaps.

            # We'll just use the predicted cx,cy and create a bbox of 20x20 around it.
            interp_large_rows = interp_large.withColumn(
                "interp_x1", col("pred_cx") - 10
            ).withColumn(
                "interp_y1", col("pred_cy") - 10
            ).withColumn(
                "interp_x2", col("pred_cx") + 10
            ).withColumn(
                "interp_y2", col("pred_cy") + 10
            ).withColumn(
                "interp_timestamp", lit(0.0)  # We don't have real timestamps; can approximate later
            ).withColumn(
                "interp_vx", lit(0.0)
            ).withColumn(
                "interp_vy", lit(0.0)
            ).withColumn(
                "confidence", lit(0.5)
            ).withColumn(
                "dx", lit(0.0)
            ).withColumn(
                "dy", lit(0.0)
            ).withColumn(
                "dx_meters", lit(0.0)
            ).withColumn(
                "dy_meters", lit(0.0)
            ).withColumn(
                "distance_meters", lit(0.0)
            ).withColumn(
                "speed_kmh", lit(0.0)
            ).withColumn(
                "vx", lit(0.0)
            ).withColumn(
                "vy", lit(0.0)
            ).withColumn(
                "ax", lit(0.0)
            ).withColumn(
                "ay", lit(0.0)
            ).withColumn(
                "acceleration", lit(0.0)
            )

            # Prepare the row format for union
            interp_large_rows = interp_large_rows.select(
                lit(None).cast(IntegerType()).alias("id"),
                col("match_id"),
                col("interp_frame").alias("frame_number"),
                col("interp_timestamp").alias("timestamp"),
                col("confidence"),
                col("interp_x1").alias("x1"),
                col("interp_y1").alias("y1"),
                col("interp_x2").alias("x2"),
                col("interp_y2").alias("y2"),
                col("pred_cx").alias("cx"),
                col("pred_cy").alias("cy"),
                ((col("interp_x2") - col("interp_x1")) * (col("interp_y2") - col("interp_y1"))).alias("area"),
                col("dx"),
                col("dy"),
                col("dx_meters"),
                col("dy_meters"),
                col("distance_meters"),
                col("speed_kmh"),
                col("vx"),
                col("vy"),
                col("ax"),
                col("ay"),
                col("acceleration")
            )
        else:
            interp_large_rows = self.spark.createDataFrame([], self._interpolated_schema())

        original_cols = [
            "id", "match_id", "frame_number", "timestamp", "confidence",
            "x1", "y1", "x2", "y2", "cx", "cy", "area",
            "dx", "dy", "dx_meters", "dy_meters", "distance_meters",
            "speed_kmh", "vx", "vy", "ax", "ay", "acceleration"
        ]
        original = winners_df.select(*[col(c) for c in original_cols])

        interp_small_rows = interp_small_rows.select(*[col(c) for c in original_cols])

        if traj_df is not None:
            interp_large_rows = interp_large_rows.select(*[col(c) for c in original_cols])

        result = original
        if interp_small_rows.count() > 0:
            result = result.unionByName(interp_small_rows)
        if traj_df is not None and interp_large_rows.count() > 0:
            result = result.unionByName(interp_large_rows)

        return result.orderBy("frame_number")

    def _prepare_interpolated_rows(self, interpolated_df: DataFrame, is_linear: bool) -> DataFrame:
        """Helper to format interpolated rows with the same schema."""
        return interpolated_df.select(
            lit(None).cast(IntegerType()).alias("id"),
            col("match_id"),
            col("interp_frame").alias("frame_number"),
            col("interp_timestamp").alias("timestamp"),

            ((col("confidence") + lead("confidence", 1).over(Window.partitionBy("match_id").orderBy("frame_number"))) / 2.0 * 0.9).alias("confidence")
            if is_linear else lit(0.5).alias("confidence"),
            col("interp_x1").alias("x1"),
            col("interp_y1").alias("y1"),
            col("interp_x2").alias("x2"),
            col("interp_y2").alias("y2"),
            col("interp_cx").alias("cx"),
            col("interp_cy").alias("cy"),
            ((col("interp_x2") - col("interp_x1")) * (col("interp_y2") - col("interp_y1"))).alias("area"),
            col("dx"),
            col("dy"),
            col("dx_meters"),
            col("dy_meters"),
            col("distance_meters"),
            col("speed_kmh"),
            col("vx"),
            col("vy"),
            col("ax"),
            col("ay"),
            col("acceleration")
        )

    def _interpolated_schema(self) -> StructType:
        """Schema for interpolated rows (same as ball state)."""
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
        winner_ids = {row.id for row in winners_df.filter(col("id").isNotNull()).select("id").collect()}
        original_ids = {s.id for s in original_states}
        to_delete = original_ids - winner_ids

        if to_delete:
            logfire.info(f"[BallPredictor] Deleting {len(to_delete)} non-winner ball states")
            for bid in to_delete:
                state = session.get(BallState, bid)
                if state:
                    session.delete(state)

        interp_rows = interpolated_df.filter(col("id").isNull()).collect()
        inserted = 0
        for row in interp_rows:
            new_state = BallState(
                match_id=match_id,
                frame_number=row.frame_number,
                timestamp=row.timestamp if row.timestamp is not None else 0.0,
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