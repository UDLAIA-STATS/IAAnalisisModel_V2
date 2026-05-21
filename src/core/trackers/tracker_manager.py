from typing import List, Optional

from sqlmodel import Session

from src.entities.models.app.detector_base import TrackManagerItem
from src.entities.models.app.video_item import VideoItem
from src.entities.utils.singleton import Singleton


class TrackerManager(metaclass=Singleton):
    def __init__(self, trackers: Optional[List[TrackManagerItem]]) -> None:
        self.trackers: List[TrackManagerItem] = []

        if trackers is not None:
            self.trackers.extend(trackers)

    def add_tracker(self, item: TrackManagerItem) -> None:
        self.trackers.append(item)
    
    def execute_trackers(self, video_item: VideoItem, session: Session):
        for item in self.trackers:
            item.tracker.get_tracks(video_item, item.object_ids, session)
