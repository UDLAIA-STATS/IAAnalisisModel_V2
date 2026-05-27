from sqlmodel import Session, select
from entities.models.requests.queue_model import TaskStep, Task


class TaskRepository:
    @staticmethod
    def get_task(task_id: str, session: Session) -> Task | None:
        query = select(Task).where(Task.id == task_id)
        return session.exec(query).first()

    @staticmethod
    def upsert_task(task: Task, session: Session):
        session.add(task)
        session.commit()
        session.refresh(task)
        return task

    @staticmethod
    def upsert_task_step(task_step: TaskStep, session: Session):
        session.add(task_step)
        session.commit()
        session.refresh(task_step)
        return task_step

    @staticmethod
    def get_task_step(task_id: str, step_number: int, session: Session) -> TaskStep | None:
        query = select(TaskStep).where(TaskStep.task_id == task_id and TaskStep.step_number == step_number)
        return session.exec(query).first()
