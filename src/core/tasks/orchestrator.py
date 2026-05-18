

from pathlib import Path

from sqlmodel import Session

from core.repository.task_repository import TaskRepository
from core.trackers import BallTracker, GoalTracker, PlayerTracker, TrackerManager, tracker_manager
from core.video.video_manager import VideoManager
from entities.models.app.analyze_request import AnalyzeRequest
from config.configuration import settings
from entities.models.app.detector_base import TrackManagerItem
from entities.models.app.queue_model import Task, TaskStep
from entities.types.states import States

class Orchestrator():
    
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


    
    def run_task(self, request: AnalyzeRequest, session: Session, task_id: str):
        task = TaskRepository.get_task(task_id, session)

        if task is None:
            task = Task(
                match_id=request.match_id,
                video_name=request.video_name,
                user_id=request.user_id
            )
            TaskRepository.upsert_task(task, session)

        video_batching_step = TaskStep(
            task_id=task.id,
            name="Video Batching",
            message="Iniciando análisis",
            step_number=1
        )

        video_batching_step = TaskRepository.upsert_task_step(video_batching_step, session)

        video_manager = VideoManager(
            match_id=request.match_id,
            video_path=Path(request.video_name),
            show=True)
        batches = video_manager.read_video(int(settings.BATCH_SIZE))

        video_batching_step.state = States.COMPLETED
        TaskRepository.upsert_task_step(video_batching_step, session)

        tracker_manager = self._init_trackers()

        
        for batch in batches:
            for video_item in batch:
                pass
