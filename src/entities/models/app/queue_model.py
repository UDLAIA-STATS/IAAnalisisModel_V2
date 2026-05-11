from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel
from sqlmodel import Field, Relationship, SQLModel, Column, JSON
from pydantic import Field as PyField
from entities.models.base_models import (
    AuditTableCompletedTable,
    AuditTable,
    UUIDModel)
from src.entities.types.states import States


class AnalysisStep(AuditTableCompletedTable, table=True):
    id: int
    name: str = "Analysis Step"
    state: States = PyField(default=States.PENDING)

class TaskDetail(UUIDModel, AuditTable, table=True):
    task_id: str = Field(foreign_key="task.id", index=True)
    step: AnalysisStep = Field(sa_column=Column(JSON))
    message: str | None = Field(default=None, nullable=True)
    task: 'Task' = Relationship(back_populates="details")

class Task(UUIDModel, AuditTableCompletedTable, table=True):
    match_id: int = Field(index=True)
    video_name: str = Field(index=True)
    state: States = Field(index=True, default=States.PENDING)
    details: list[TaskDetail] = Relationship(
        back_populates="task",
        sa_relationship_kwargs={"cascade": "all, delete-orphan"})