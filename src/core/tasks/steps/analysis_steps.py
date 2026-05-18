from sqlmodel import Session

from core.trackers import TrackerManager
from entities.interfaces.app import AnalysisStepHandler
from entities.models.app.video_item import VideoItem

### Object detection --> Video Frame
### Number and color recognition --> Video Frame
### Physics computation --> Video Frame

### Team assigment --> DB
### Ball assignment --> Db
### Goal interaction --> DB

### Heatmap --> DB
### Data post processing
### Document uplaod --> Post



class ObjectDetection(AnalysisStepHandler):
    name = "Object Detection"
    number_step = 1

    def execute(self, session: Session, **kwargs) -> bool:
        # track_manager = TrackerManager(trackers=None)
        # ball_tracker = BallTracker(tracker_config_file=None)
        # goal_tracker = GoalTracker(tracker_config_file=None)
        # player_tracker = PlayerTracker()

        track_manager: TrackerManager = kwargs["track_manager"]
        video_item: VideoItem = kwargs["video_item"]
        # ball_tracker = kwargs["ball_tracker"]
        # goal_tracker = kwargs["goal_tracker"]
        # player_tracker = kwargs["player_tracker"]
        # track_manager.add_tracker(TrackManagerItem(tracker=ball_tracker, object_ids=[0]))
        # track_manager.add_tracker(TrackManagerItem(tracker=goal_tracker, object_ids=[0]))
        # track_manager.add_tracker(TrackManagerItem(tracker=player_tracker, object_ids=[0]))

        track_manager.execute_trackers(video_item, session=session)
        
        return True


