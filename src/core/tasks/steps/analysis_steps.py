import traceback
from typing import List
import uuid

import cv2
import logfire
import numpy as np
from sqlmodel import Session

from src.entities.models.requests.queue_model import TaskStep
from src.entities.types.states import StatesModel
from src.core.repository.player_repository import PlayerRepository
from src.core.trackers import TrackerManager
from src.core.vision.color_recognizer import jersey_color_extractor
from src.entities.interfaces.app import AnalysisStepHandler
from src.entities.models.app.video_item import VideoItem
from src.core.video.annotators import player_annotator
from src.core.repository import PlayerStatesRepository, files_repository, TaskRepository
from src.core.vision.number_recognizer import number_predictor

# Object detection --> Video Frame
# Number and color recognition --> Video Frame
# Physics computation --> Video Frame

# Team assigment --> DB
# Ball assignment --> Db
# Goal interaction --> DB

# Heatmap --> DB
# Data post processing
# Document uplaod --> Post


class VideoDownload(AnalysisStepHandler):
    name = "Download Video"
    number_step = 1

    def execute(self, session: Session, **kwargs) -> bool:
        video_name: str = kwargs["video_name"]
        destination: str = kwargs["destination"]
        task_id: str = kwargs["task_id"]

        step = TaskStep(
            task_id=task_id,
            name="Video Download",
            message="Descargando el video",
            step_number=1,
        )
        TaskRepository.upsert_task_step(step, session)
        try:
            files_repository.steam_download(video_name, destination)
            step.state = StatesModel.COMPLETED
            TaskRepository.upsert_task_step(step, session)
            return True
        except Exception as e:
            logfire.exception(traceback.format_exc())
            step.state = StatesModel.FAILED
            TaskRepository.upsert_task_step(step, session)
            raise e


class ObjectDetection(AnalysisStepHandler):
    name = "Object Detection"
    number_step = 2

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
    number_step = 3

    def execute(self, session: Session, **kwargs) -> bool:
        video_item: VideoItem = kwargs["video_item"]
        # logfire.info(f"[NumberAndColorRecognition] Number of states: {len(states)}")
        try:
            session.commit()
            labels = jersey_color_extractor.recognize(video_item, session)
            numbers = number_predictor.predict(video_item.match_id, video_item, session)

            for number, label in zip(numbers, labels):
                id = label.split("|")[0].split(":")[1].strip()
                
                if id != number.player_id:
                    continue

                label += f" | Number: {number.number}"

            if len(labels) == len(player_annotator.get_detections()):
                video_item.annotated_frame = player_annotator.annotate(
                    annotated_frame=video_item.annotated_frame,
                    detections=None,
                    labels=labels,
                )

            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
