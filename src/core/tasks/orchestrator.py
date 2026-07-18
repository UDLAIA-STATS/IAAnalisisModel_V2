from datetime import datetime
from pathlib import Path
import traceback

import logfire
import tqdm

from src.core.vision.calibrators.pnl_calibrator import PnLCalibrator
from src.core.vision.correction.scale_corrector import ScaleCorrector
from src.core.vision.detectors.pnl_detector import PnLFeatureDetector

from src.config.routes import INPUT_VIDEOS_DIR
from src.config.configuration import settings

from src.core.database import connection_manager
from src.core.reporter.time_reporter import ProcessTimeReporter
from src.core.repository.task_repository import TaskRepository
from src.core.trackers import BallTracker, GoalTracker, PlayerTracker, TrackerManager
from src.core.video.video_manager import VideoManager

from src.entities.models.requests.analyze_request import AnalyzeRequest
from src.entities.models.app.detector_base import TrackManagerItem
from src.entities.models.requests.queue_model import Task, TaskStep
from src.entities.types.states import StatesModel

from src.core.reporter.detections_reporter import reporter as detection_reporter
from src.core.tasks.steps import (
    ConversionCalculatorSteps,
    ValidationProcess,
    NumberAndColorRecognition,
    ObjectDetection,
    VideoDownload,
)
from src.core.vision import scale_motion_detector, tilt_detector, pitch_homography
from src.core.post_processing import resume_generator


class Orchestrator:
    def __init__(self):
        super().__init__()

    def _init_trackers(self) -> TrackerManager:
        track_manager = TrackerManager(trackers=None)
        ball_tracker = BallTracker(tracker_config_file=None)
        goal_tracker = GoalTracker(tracker_config_file=None)
        player_tracker = PlayerTracker()
        track_manager.add_tracker(
            TrackManagerItem(tracker=ball_tracker, object_ids=[0])
        )
        track_manager.add_tracker(
            TrackManagerItem(tracker=goal_tracker, object_ids=[0])
        )
        track_manager.add_tracker(
            TrackManagerItem(tracker=player_tracker, object_ids=[0])
        )

        return track_manager

    def run_tasks(self, request: AnalyzeRequest, task_id: str):
        with connection_manager.create_session() as session:
            task = TaskRepository.get_task(task_id, session)
            time_reporter = ProcessTimeReporter(match_id=request.match_id)

            if task is None:
                task = Task(
                    match_id=request.match_id,
                    video_name=request.video_name,
                    nickname=request.nickname,
                )
                TaskRepository.upsert_task(task, session)

            video_path = INPUT_VIDEOS_DIR / request.video_name

            if not request.video_name.startswith(r"C:\Users"):
                time_reporter.start("Video Download")
                VideoDownload().execute(
                    session,
                    video_name=request.video_name,
                    destination=video_path.as_posix(),
                    task_id=task.id,
                )
                time_reporter.stop("Video Download")

            video_batching_step = TaskStep(
                task_id=task.id,
                name="Video Batching",
                message="Dividiendo en batch el video",
                step_number=1,
            )
            video_batching_step = TaskRepository.upsert_task_step(
                video_batching_step, session
            )

            video_manager = VideoManager(
                match_id=request.match_id, video_path=video_path, show=True
            )
            time_reporter.start("Video Analysis (General)")
            batches = video_manager.read_video(
                int(settings.BATCH_SIZE), request.match_id
            )
            total_frames = video_manager.get_total_frames()
            fps = video_manager.get_fps()
            first_frame = video_manager.get_first_frame()
            frame_w, frame_h = video_manager.get_frame_size()

            scale_motion_detector.start(first_frame)
            tilt_detector.start(first_frame)
            pitch_homography.set_reference_frame_size(int(frame_w), int(frame_h))
            pitch_homography.set_correctors(
                scale_corrector=ScaleCorrector(session),
                calibrator=PnLCalibrator(
                    image_width=int(frame_w), image_height=int(frame_h)
                ),
                detector=PnLFeatureDetector(image_size=(int(frame_w), int(frame_h))),
            )

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
            total_batches = total_frames // int(settings.BATCH_SIZE)

            validation_step = TaskStep(
                task_id=task.id,
                name="Validating Players",
                message="Validando jugadores detectados",
                step_number=3,
            )

            reporter_step = TaskStep(
                task_id=task.id,
                name="Generating Report",
                message="Generando reporte",
                step_number=4,
            )

            current_step = None

            try:
                current_step = processing_batch_step
                time_reporter.start("Video Loop")
                with tqdm.tqdm(total=total_batches, desc="Processing Video") as pbar:
                    for batch in batches:
                        for video_item in batch:
                            time_reporter.start("Object Detection")
                            object_detection.execute(
                                session=session,
                                video_item=video_item,
                                track_manager=tracker_manager,
                            )
                            time_reporter.stop("Object Detection")

                            time_reporter.start("Color and Number Recognition")
                            color_number_recognizer.execute(
                                session=session, video_item=video_item
                            )
                            time_reporter.stop("Color and Number Recognition")
                            video_manager.write(
                                video_item.annotated_frame,
                                video_item.frame_num,
                                save_frame=True,
                            )

                            time_reporter.start("Calculating Constants")
                            constant_calculator_steps.execute(
                                session=session,
                                video_item=video_item,
                                video_manager=video_manager,
                            )
                            time_reporter.stop("Calculating Constants")

                        pbar.set_postfix({"Actual batch": batches_count})
                        pbar.update(1)
                        batches_count += 1

                time_reporter.stop("Video Loop")
                session.commit()
                processing_batch_step.state = StatesModel.COMPLETED
                TaskRepository.upsert_task_step(processing_batch_step, session)

                current_step = validation_step
                time_reporter.start("PostProcesamiento")
                ValidationProcess().execute(
                    session=session,
                    task=task,
                    request=request,
                    total_frames=total_frames,
                    fps=fps,
                )
                time_reporter.stop("PostProcesamiento")
                validation_step.state = StatesModel.COMPLETED
                TaskRepository.upsert_task_step(validation_step, session)

                current_step = reporter_step
                detection_reporter.generate_report(request.match_id, session)
                logfire.info(f"Enviando resumen, color {request.color}")
                resume_generator.send_resumes(
                    request.match_id, request.color, True, session
                )
                reporter_step.state = StatesModel.COMPLETED
                TaskRepository.upsert_task_step(reporter_step, session)

                time_reporter.stop("Video Analysis (General)")
                time_reporter.publish()

                task.general_state = StatesModel.COMPLETED
                task.completed_at = datetime.now()
                TaskRepository.upsert_task(task, session)

            except Exception as e:
                error_msg = traceback.format_exc()
                resume_generator.send_resumes(
                    request.match_id, request.color, False, session
                )

                logfire.fatal(
                    f"Error durante '{current_step.name if current_step else 'run_tasks'}': {error_msg}"
                )

                if current_step:
                    current_step.state = StatesModel.FAILED
                    current_step.message = f"Error: {error_msg[:400]}"
                    TaskRepository.upsert_task_step(current_step, session)

                task.general_state = StatesModel.FAILED
                task.completed_at = datetime.now()
                TaskRepository.upsert_task(task, session)
                raise e
