from typing import List, Union
from cv2.typing import MatLike
import logfire

from src.entities.services.video_manager_base import VideoManagerBase
from src.core.services.global_value_store import value_store


class VideoManager(VideoManagerBase):
    def read_video(self, batch_size: int, match_id: int):
        if not self.check_video_state():
            logfire.info(f"[VideoManager] Cap was not opened, opening video {self.video_path}")
            self.cap.open(self.video_path.as_posix())

        logfire.info(f"[VideoManager] Start reading video of match {self.match_id} with {batch_size} frames per batch")

        frame_rate = self.get_fps()
        total_frames = self.get_total_frames()
        self.frame_size = self.get_frame_size()
        value_store.set("frame_rate", frame_rate)
        value_store.set("total_frames", total_frames)
        value_store.set("frame_size", self.frame_size)

        frame_count = 0

        while frame_count < total_frames:
            to_read = min(batch_size, total_frames - frame_count)
            batch = self.get_batch(to_read, match_id)

            if not batch or len(batch) == 0:
                break

            frame_count += len(batch)
            yield batch

        self.close()

    def write(self, frames: Union[List[MatLike], MatLike], frame_num: int, save_frame: bool = False):
        if isinstance(frames, List):
            for frame in frames:
                self.writer.write(frame)
                self.preview_frame(frame)
                if save_frame:
                    self._save_frame_as_image(frame_num, frame)
            return

        logfire.info(f"""shape={frames.shape} | expected={(self.writing_width, self.writing_height)} |received={(frames.shape[1], frames.shape[0])}""")

        self.writer.write(frames)
        self.preview_frame(frames)

        if save_frame:
            self._save_frame_as_image(frame_num, frames)
