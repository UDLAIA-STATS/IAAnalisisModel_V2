from pathlib import Path
import traceback

import logfire
import tqdm

from src.config.routes import INPUT_VIDEOS_DIR
from src.core.tasks.steps.post_process_steps import ValidationProcess
from src.core.database import connection_manager
from src.core.reporter.time_reporter import ProcessTimeReporter
from src.core.repository.task_repository import TaskRepository
from src.core.tasks.steps.analysis_steps import NumberAndColorRecognition, ObjectDetection, VideoDownload
from src.core.trackers import BallTracker, GoalTracker, PlayerTracker, TrackerManager
from src.core.video.video_manager import VideoManager

from src.config.configuration import settings

from src.entities.models.requests.analyze_request import AnalyzeRequest
from src.entities.models.app.detector_base import TrackManagerItem
from src.entities.models.requests.queue_model import Task, TaskStep
from src.entities.types.states import StatesModel
from src.core.reporter.detections_reporter import reporter as detection_reporter
from src.core.tasks.steps.conversion_steps import ConversionCalculatorSteps
from src.core.vision.camera_scale import scale_motion_detector

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
            time_reporter = ProcessTimeReporter(match_id=request.match_id)

            if task is None:
                task = Task(match_id=request.match_id, video_name=request.video_name, nickname=request.nickname)
                TaskRepository.upsert_task(task, session)

            video_path = INPUT_VIDEOS_DIR / request.video_name
            
            if not request.video_name.startswith(r"C:\Users"):
                time_reporter.start("Video Download")
                VideoDownload().execute(session, video_name=request.video_name, destination=video_path.as_posix(), task_id=task.id)
                time_reporter.stop("Video Download")

            video_batching_step = TaskStep(task_id=task.id, name="Video Batching", message="Dividiendo en batch el video", step_number=1)
            video_batching_step = TaskRepository.upsert_task_step(video_batching_step, session)

            video_manager = VideoManager(match_id=request.match_id, video_path=video_path, show=True)
            time_reporter.start("Video Analysis (General)")
            batches = video_manager.read_video(int(settings.BATCH_SIZE), request.match_id)
            total_frames = video_manager.get_total_frames()
            fps = video_manager.get_fps()
            scale_motion_detector.start(video_manager.get_first_frame())

            video_batching_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(video_batching_step, session)

            video_batching_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(video_batching_step, session)

            tracker_manager = self._init_trackers()
            object_detection = ObjectDetection()
            color_number_recognizer = NumberAndColorRecognition()
            constant_calculator_steps = ConversionCalculatorSteps()

            processing_batch_step = TaskStep(
                task_id=task.id,
                name="Processing Batch",
                message="Procesando batch, detectando objetos, reconociendo color, números y calculando distancias y velocidades",
                step_number=2,
            )
            batches_count = 1

            try:
                for batch in tqdm.tqdm(batches, total=total_frames // int(settings.BATCH_SIZE), postfix=f"Actual batch: {batches_count}"):
                    for video_item in batch:
                        time_reporter.start("Object Detection")
                        object_detection.execute(session=session, video_item=video_item, track_manager=tracker_manager)
                        time_reporter.stop("Object Detection")
                        time_reporter.start("Color and Number Recognition")
                        color_number_recognizer.execute(session=session, video_item=video_item)
                        time_reporter.stop("Color and Number Recognition")
                        video_manager.write(video_item.annotated_frame, video_item.frame_num, save_frame=True)

                        time_reporter.start("Calculating Constants")
                        constant_calculator_steps.execute(session=session, video_item=video_item)
                        time_reporter.stop("Calculating Constants")

                    batches_count += 1

            except Exception as e:
                processing_batch_step.state = StatesModel.FAILED
                processing_batch_step.message = f"Error procesando batch: {traceback.format_exc(500)}"
                TaskRepository.upsert_task_step(processing_batch_step, session)
                task.general_state = StatesModel.FAILED
                TaskRepository.upsert_task(task, session)
                logfire.fatal(f"Error processing batch: {traceback.format_exc()}")
                raise e

            session.commit()
            processing_batch_step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(processing_batch_step, session)

            time_reporter.start("Validando Jugadores")
            try:
                ValidationProcess().execute(session=session, task=task, request=request, total_frames=total_frames, fps=fps)
            except Exception as e:
                logfire.fatal(f"Error validating players: {traceback.format_exc(500)}")
                task.general_state = StatesModel.FAILED
                TaskRepository.upsert_task(task, session)
                raise e
            finally:
                time_reporter.stop("Validando Jugadores")
            
            reporter_step = TaskStep(
                task_id=task.id,
                name="Generating Report",
                message="Generando reporte",
                step_number=4,
            )
            TaskRepository.upsert_task_step(reporter_step, session)

            try:
                detection_reporter.generate_report(request.match_id, session)
                reporter_step.state = StatesModel.COMPLETED
                TaskRepository.upsert_task_step(reporter_step, session)


                time_reporter.stop("Video Analysis (General)")
                time_reporter.publish()
            except Exception as e:
                logfire.fatal(f"Error generating report: {traceback.format_exc(500)}")
                detection_reporter.generate_report(request.match_id, session)
                reporter_step.state = StatesModel.FAILED
                TaskRepository.upsert_task_step(reporter_step, session)
                task.general_state = StatesModel.FAILED
                TaskRepository.upsert_task(task, session)

                raise e

            task.general_state = StatesModel.COMPLETED
            TaskRepository.upsert_task(task, session)
