from datetime import date

from sqlmodel import Session, asc, desc, func, select
from sqlalchemy.orm import selectinload
from src.entities.types.states import StatesModel
from src.entities.models.requests.queue_model import TaskStep, Task


class TaskRepository:
    @staticmethod
    def get_task(task_id: str, session: Session) -> Task | None:
        query = select(Task).where(Task.id == task_id)
        return session.exec(query).first()
    
    @staticmethod
    def get_tasks(
        session: Session,
        analysis_date: date | None = None,
        order: str = "desc",
    ) -> list[Task]:
        statement = select(Task)

        if analysis_date is not None:
            statement = statement.where(func.date(Task.created_at) == analysis_date)

        order_column = Task.created_at
        statement = statement.order_by(
            asc(order_column) if order == "asc" else desc(order_column)
        )

        return list(session.exec(statement).all())

    @staticmethod
    def upsert_task(task: Task, session: Session):
        db_task = session.get(Task, task.id)

        if db_task is not None:
            db_task.sqlmodel_update(task.model_dump(exclude_unset=True))
            session.add(db_task)
            session.commit()
            session.refresh(db_task)

        session.add(task)
        session.commit()
        session.refresh(task)
        return task

    @staticmethod
    def upsert_task_step(task_step: TaskStep, session: Session):
        db_step = session.get(TaskStep, task_step.id)

        if db_step is not None:
            db_step.sqlmodel_update(task_step.model_dump(exclude_unset=True))
            session.add(db_step)
            session.commit()
            session.refresh(db_step)
            return db_step

        session.add(task_step)
        session.commit()
        session.refresh(task_step)
        return task_step

    @staticmethod
    def get_task_step(task_id: str, step_number: int, session: Session) -> TaskStep | None:
        query = select(TaskStep).where(TaskStep.task_id == task_id and TaskStep.step_number == step_number)
        return session.exec(query).first()

    @staticmethod
    def cancel_pending_tasks(session: Session):
        query = select(Task).where(Task.general_state == StatesModel.PENDING or Task.general_state == StatesModel.PROCESSING)
        tasks = session.exec(query).all()
        for task in tasks:
            task.state = StatesModel.CANCELLED
            session.add(task)

            for step in task.steps:
                step.state = StatesModel.CANCELLED
                session.add(step)

        session.commit()
