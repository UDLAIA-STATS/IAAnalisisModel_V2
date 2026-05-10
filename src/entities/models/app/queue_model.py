from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel
from sqlmodel import Field, SQLModel
from src.entities.types.states import States

class Task(SQLModel, table=True):
    id: str = Field(primary_key=True, index=True, default_factory=lambda: str(uuid4()))
    match_id: int
    video_name: str
    state: States
    create_at: datetime
    finished_at: datetime
    updated_at: datetime

class AnalysisStep(BaseModel):
    id: int
    create_at: datetime
    completed_at: datetime
    state: States

class TaskDetail(SQLModel, table=True):
    id: str = Field(primary_key=True, index=True, default_factory=lambda: str(uuid4()))
    task_id: str = Field(foreign_key="task.id", index=True)
    step: AnalysisStep
    message: str
    create_at: datetime
