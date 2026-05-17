from typing import Tuple

from pydantic import BaseModel, ConfigDict


class TrackData(BaseModel):
    xyxy: Tuple[int, int, int, int]
    track_id: int
    confidence: float

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)