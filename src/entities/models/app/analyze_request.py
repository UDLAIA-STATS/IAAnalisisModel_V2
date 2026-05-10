from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    video_name: str
    match_id: int
    color: str = Field(examples=["123,111,202"])
    user_id: int
