import logfire
from sqlmodel import Session

from src.core.trackers import TrackerManager
from src.core.vision.color_recognizer import ColorRecognizer
from src.entities.interfaces.app import AnalysisStepHandler
from src.entities.models.app.video_item import VideoItem
from src.core.video.annotators import player_annotator
from src.core.repository import PlayerStatesRepository

# Object detection --> Video Frame
# Number and color recognition --> Video Frame
# Physics computation --> Video Frame

# Team assigment --> DB
# Ball assignment --> Db
# Goal interaction --> DB

# Heatmap --> DB
# Data post processing
# Document uplaod --> Post


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

        try:
            track_manager.execute_trackers(video_item, session)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e


class NumberAndColorRecognition(AnalysisStepHandler):
    name = "Number and Color Recognition"
    number_step = 2

    def execute(self, session: Session, **kwargs) -> bool:
        video_item: VideoItem = kwargs["video_item"]
        states = PlayerStatesRepository.get_states_by_frame(video_item.match_id, video_item.frame_num, session=session)
        logfire.info(f"[NumberAndColorRecognition] Number of states: {len(states)}")
        labels = []

        if len(states) == 0:
            return True

        try:
            for state in states:
                x1, y1, x2, y2 = state.x1, state.y1, state.x2, state.y2

                if x1 is None or y1 is None or x2 is None or y2 is None:
                    logfire.error(
                        f"[NumberAndColorRecognition] No coordinates or coordinates incompleted for "
                        f"player {state.player.track_id} in frame {video_item.frame_num} in match {video_item.match_id}"
                    )
                    continue

                crop = video_item.frame.copy()
                crop = crop[int(y1) : int(y2), int(x1) : int(x2)]
                rgb, hex = ColorRecognizer.extract_color(crop)
                rgb_str = f"{rgb[0]:.0f},{rgb[1]:.0f},{rgb[2]:.0f}"

                state.player.team_color = rgb_str
                label = f"ID: {state.player.track_id} |{hex}| Conf: {state.confidence:.2f}"
                labels.append(label)

                session.add(state)
                session.flush()

            video_item.annotated_frame = player_annotator.annotate(
                annotated_frame=video_item.annotated_frame, detections=None, labels=labels)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
