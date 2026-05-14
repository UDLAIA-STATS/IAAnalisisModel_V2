from typing import List, Optional

from sqlmodel import Field, Relationship
from entities.models.base_models import AuditTableCompletedTable, AuditTable, NumericIdModel, UUIDModel
from src.entities.types.states import States


class AnalysisStep(AuditTableCompletedTable, NumericIdModel, table=True):
    __tablename__ = "analysisstep"  # type: ignore
    name: str = "Analysis Step"
    state: States = Field(default=States.PENDING)

    task_detail_id: str = Field(foreign_key="taskdetail.id", index=True, nullable=False)
    task_detail: Optional["TaskDetail"] = Relationship(back_populates="steps")


class TaskDetail(UUIDModel, AuditTable, table=True):
    __tablename__ = "taskdetail"  # type: ignore
    task_id: str = Field(foreign_key="task.id", index=True)
    message: str | None = Field(default=None, nullable=True)

    task: "Task" = Relationship(back_populates="details")
    steps: List[AnalysisStep] = Relationship(back_populates="task_detail", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class Task(UUIDModel, AuditTableCompletedTable, table=True):
    match_id: int = Field(index=True)
    video_name: str = Field(index=True)
    state: States = Field(index=True, default=States.PENDING)

    details: list[TaskDetail] = Relationship(back_populates="task", sa_relationship_kwargs={"cascade": "all, delete-orphan"})
