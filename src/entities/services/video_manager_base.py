from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator, List, Union
import uuid
from cv2 import VideoCapture
import cv2
from cv2.typing import MatLike
import logfire

from src.config.routes import ANOTATED_OUTPUT_IMAGES, ANOTATED_VIDEOS_DIR
from src.entities.models.app.video_item import VideoItem


class VideoManagerBase(ABC):
    def __init__(self, match_id: int, video_path: Path, show: bool = False):
        if self.validate_video(video_path):
            self.cap: VideoCapture = VideoCapture(video_path.as_posix())
        else:
            logfire.error(f"Video no encontrado: {video_path}")
            raise FileNotFoundError("Video no encontrado")

        self.video_path = video_path
        self.output_video = ANOTATED_VIDEOS_DIR / f"{match_id}_{uuid.uuid4()}.mp4"
        self.match_id = match_id
        self.ouput_images_dir = ANOTATED_OUTPUT_IMAGES / f"{match_id}"
        self.ouput_images_dir.mkdir(parents=True, exist_ok=True)
        self.show = show

        first_frame = self.get_first_frame()
        h, w = first_frame.shape[:2]
        self.writing_width = w
        self.writing_height = h

        self.writer = cv2.VideoWriter(
            self.output_video.as_posix(),
            cv2.VideoWriter.fourcc(*"mp4v"),
            self.get_fps(),
            (int(w), int(h)),
        )
 
        if show:
            self.named_window = f"Annotated {self.video_path.name} - Match {self.match_id}"
            cv2.namedWindow(self.named_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.named_window, int(w * 2), int(h * 1.8))

    def validate_video(self, video_path: Path):
        return video_path.exists() and video_path.is_file()

    def get_frame_size(self):
        return self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_total_frames(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_first_frame(self):
        _, frame = self.cap.read()
        return frame

    def check_video_state(self):
        return self.cap.isOpened()

    def preview_frame(self, frame):
        """
        Muestra frame en pantalla.
        Retorna False si usuario pide salir.
        """
        if not self.show:
            return True

        cv2.imshow(self.named_window, frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q")):
            return False

        return True

    def get_batch(self, batch_size: int, match_id: int):
        """
        Get a batch of frames from the video.
        params:
            batch_size: the batch size
        returns:
            a list of frames with shape (frame, timestamp, frame number)
        """
        batch: List[VideoItem] = []

        for _ in range(batch_size):
            frame_exists, frame = self.cap.read()

            if not frame_exists:
                break

            dt = float(self.cap.get(cv2.CAP_PROP_POS_MSEC)) * 0.001
            frame_num = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            batch.append(VideoItem(frame=frame, annotated_frame=frame, timestamp=dt, match_id=match_id, frame_num=frame_num))

        return batch

    @abstractmethod
    def read_video(self, batch_size: int, match_id: int) -> Generator[List[VideoItem]]:
        pass

    @abstractmethod
    def write(self, frames: Union[List[MatLike], MatLike], frame_num: int, save_frame: bool = False):
        """
        Write annotated frames in the video.
        params:
            video_path: the video path
            frames: the frames to write
            frame_num: the frame number
            save_frame: save the frame as an image if True
        """
        pass

    def _save_frame_as_image(self, frame_num: int, frame: MatLike):
        image_path = self.ouput_images_dir / f"{frame_num}_{uuid.uuid4()}.jpg"
        cv2.imwrite(image_path.as_posix(), frame)

    def close(self):
        self.cap.release()
        self.writer.release()
        cv2.destroyAllWindows()

    def __exit__(self, exc_type, exc, tb):
        self.close()
