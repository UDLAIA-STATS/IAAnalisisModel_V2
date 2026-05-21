from sqlmodel import Session

from src.core.trackers import TrackerManager
from src.core.vision.color_recognizer import ColorRecognizer
from src.entities.interfaces.app import AnalysisStepHandler
from src.entities.models.app.video_item import VideoItem
from src.core.video.annotators import player_annotator
from src.core.repository import PlayerStatesRepository 

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
        """
        Execute the step and return the results.
        Args:
            session:
            video_item: the video item type VideoItem
            track_manager: the tracker manager type TrackerManager
        """
        track_manager: TrackerManager = kwargs["track_manager"]
        video_item: VideoItem = kwargs["video_item"]
        track_manager.execute_trackers(video_item, session=session)
        session.commit()

        return True


class NumberAndColorRecognition(AnalysisStepHandler):
    name = "Number and Color Recognition"
    number_step = 2

    def execute(self, session: Session, **kwargs) -> bool:
        video_item: VideoItem = kwargs["video_item"]
        states = PlayerStatesRepository.get_states_by_frame(
            video_item.match_id,
            video_item.frame_num,
            session=session)
        
        for state in states:
            x1, y1, x2, y2 = state.x1, state.y1, state.x2, state.y2
            crop = video_item.frame.copy()
            crop = crop[y1:y2, x1:x2]
            rgb, hex = ColorRecognizer.extract_color(crop)
            rgb_str = f"{rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}"

            state.player.team_color = rgb_str
            label = f"ID: {state.player.track_id} | {hex} | Conf: {state.confidence}"

            video_item.annotated_frame = player_annotator.annotate(
                annotated_frame=video_item.annotated_frame,
                detections=None,
                label=label
            )

        return True