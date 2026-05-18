from typing import Union

import numpy as np
from pydantic import BaseModel, ConfigDict

from cv2.typing import MatLike

class VideoItem(BaseModel):
    match_id: int
    frame: Union[MatLike, np.ndarray]
    annotated_frame: Union[MatLike, np.ndarray]
    frame_num: int
    timestamp: float

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

