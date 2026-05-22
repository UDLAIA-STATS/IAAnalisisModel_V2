from pathlib import Path
import traceback

import logfire
import tqdm

from src.core.database import connection_manager
from src.core.reporter.time_reporter import ProcessTimeReporter
from src.core.repository.task_repository import TaskRepository
from src.core.tasks.steps.analysis_steps import NumberAndColorRecognition, ObjectDetection
from src.core.trackers import BallTracker, GoalTracker, PlayerTracker, TrackerManager
from src.core.video.video_manager import VideoManager

from src.config.configuration import settings

from src.entities.models.app.analyze_request import AnalyzeRequest
from src.entities.models.app.detector_base import TrackManagerItem
from src.entities.models.app.queue_model import Task, TaskStep
from src.entities.types.states import StatesModel
from src.core.reporter.detections_reporter import reporter as detection_reporter

# S3 Connection Manager


class Orchestrator:
    def __init__(self):
        super().__init__()

    def _init_trackers(self) -> TrackerManager:
        track_manager = TrackerManager(trackers=None)
        ball_tracker = BallTracker(tracker_config_file=None)
        goal_tracker = GoalTracker(tracker_config_file=None)
        player_tracker = PlayerTracker()
        track_manager.add_tracker(TrackManagerItem(tracker=ball_tracker, object_ids=[0]))
        track_manager.add_tracker(TrackManagerItem(tracker=goal_tracker, object_ids=[0]))
        track_manager.add_tracker(TrackManagerItem(tracker=player_tracker, object_ids=[0]))

        return track_manager

    def run_tasks(self, request: AnalyzeRequest, task_id: str):
        with connection_manager.create_session() as session:
            task = TaskRepository.get_task(task_id, session)

            if task is None:
                task = Task(match_id=request.match_id, video_name=request.video_name, user_id=request.user_id)
                TaskRepository.upsert_task(task, session)

            video_batching_step = TaskStep(task_id=task.id, name="Video Batching", message="Dividiendo en batch el video", step_number=1)
            video_batching_step = TaskRepository.upsert_task_step(video_batching_step, session)

            video_manager = VideoManager(match_id=request.match_id, video_path=Path(request.video_name), show=True)
            time_reporter = ProcessTimeReporter(match_id=request.match_id)
            time_reporter.start("Video Analysis (General)")
            batches = video_manager.read_video(int(settings.BATCH_SIZE), request.match_id)

            video_batching_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(video_batching_step, session)

            video_batching_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(video_batching_step, session)

            tracker_manager = self._init_trackers()
            object_detection = ObjectDetection()
            color_number_recognizer = NumberAndColorRecognition()

            processing_batch_step = TaskStep(
                task_id=task.id,
                name="Processing Batch",
                message="Procesando batch, detectando objetos, reconociendo color, números y calculando distancias y velocidades",
                step_number=2,
            )

            try:
                for batch in tqdm.tqdm(batches, total=video_manager.get_total_frames() // int(settings.BATCH_SIZE)):
                    for video_item in batch:
                        logfire.info("[Orchestator] Detecting items")
                        time_reporter.start("Object Detection")
                        object_detection.execute(session=session, video_item=video_item, track_manager=tracker_manager)
                        session.commit()
                        time_reporter.stop("Object Detection")
                        logfire.info("[Orchestator] Color and number recognition")
                        time_reporter.start("Color and Number Recognition")
                        color_number_recognizer.execute(session=session, video_item=video_item)
                        time_reporter.stop("Color and Number Recognition")
                        session.commit()

            except Exception as e:
                processing_batch_step.state = StatesModel.FAILED
                processing_batch_step.message = f"Error procesando batch: {str(e)}"
                TaskRepository.upsert_task_step(processing_batch_step, session)
                logfire.error(f"Error processing batch: {traceback.format_exc()}")
                raise e

            reporter_step = TaskStep(
                task_id=task.id,
                name="Generating Report",
                message="Generando reporte",
                step_number=3,
            )
            TaskRepository.upsert_task_step(reporter_step, session)
            detection_reporter.generate_report(request.match_id, session)
            reporter_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(reporter_step, session)

            processing_batch_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(processing_batch_step, session)

            time_reporter.stop("Video Analysis (General)")
            time_reporter.publish()
