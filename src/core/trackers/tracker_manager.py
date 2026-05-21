import asyncio
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from sqlmodel import Session

from src.entities.models.app.detector_base import TrackManagerItem
from src.entities.models.app.video_item import VideoItem
from src.entities.utils.singleton import Singleton

executor = ThreadPoolExecutor(max_workers=3)

class TrackerManager(metaclass=Singleton):
    def __init__(self, trackers: Optional[List[TrackManagerItem]]) -> None:
        self.trackers: List[TrackManagerItem] = []

        if trackers is not None:
            self.trackers.extend(trackers)

    def add_tracker(self, item: TrackManagerItem) -> None:
        self.trackers.append(item)

    async def execute_trackers(self, video_item: VideoItem):
        for item in self.trackers:
            item.tracker.get_tracks(video_item=video_item, object_ids=item.object_ids)
        # loop = asyncio.get_running_loop()

        # tasks = [
        #     loop.run_in_executor(
        #         executor,
        #         item.tracker.get_tracks,
        #         video_item,
        #         item.object_ids
        #     ) for item in self.trackers
        # ]

        # await asyncio.gather(*tasks)

