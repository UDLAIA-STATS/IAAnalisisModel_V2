from pydantic import BaseModel

from cv2.typing import MatLike

class VideoItem(BaseModel):
    frame: MatLike
    frame_num: int
    timestamp: float

    class Config:
        arbitrary_types_allowed = True