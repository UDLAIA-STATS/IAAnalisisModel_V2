from typing import List

from sqlmodel import Field, Relationship
from src.entities.models.base_models import AuditTableCompletedTable, NumericIdModel, UUIDModel
from src.entities.types.states import StatesModel


class TaskStep(AuditTableCompletedTable, NumericIdModel, table=True):
    __tablename__ = "task_steps"  # type: ignore
    step_number: int = Field(index=True)
    name: str = Field(max_length=100)
    message: str = Field(max_length=500)
    state: StatesModel = Field(default=StatesModel.PENDING)

    task_id: str = Field(foreign_key="tasks.id", index=True)
    task: "Task" = Relationship(back_populates="steps")


class Task(UUIDModel, AuditTableCompletedTable, table=True):
    __tablename__ = "tasks"  # type: ignore
    match_id: int = Field(index=True)
    video_name: str = Field(index=True)
    general_state: StatesModel = Field(index=True, default=StatesModel.PENDING)
    user_id: int

    steps: List[TaskStep] = Relationship(back_populates="task", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
