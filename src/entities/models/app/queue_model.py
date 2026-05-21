from typing import List

from sqlmodel import Field, Relationship
from entities.models.base_models import AuditTableCompletedTable, NumericIdModel, UUIDModel
from src.entities.types.states import States


class TaskStep(AuditTableCompletedTable, NumericIdModel, table=True):
    __tablename__ = "task_steps"  # type: ignore
    step_number: int = Field(index=True)
    name: str = Field(max_length=100)
    message: str = Field(max_length=500)
    state: States = Field(default=States.PENDING)

    task_id: str = Field(foreign_key="task.id", index=True)
    task: "Task" = Relationship(back_populates="details")


class Task(UUIDModel, AuditTableCompletedTable, table=True):
    match_id: int = Field(index=True)
    video_name: str = Field(index=True)
    state: States = Field(index=True, default=States.PENDING)
    user_id: int

    steps: List[TaskStep] = Relationship(back_populates="task_steps", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
