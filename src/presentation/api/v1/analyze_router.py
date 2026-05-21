from fastapi import APIRouter, Depends
from sqlmodel import Session

from core.repository.task_repository import TaskRepository
from core.tasks.orchestrator import Orchestrator
from src.core.database import connection_manager
from src.entities.models.app.analyze_request import AnalyzeRequest
from src.entities.models.app.queue_model import Task
from src.entities.types.states import States

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/run", status_code=202)
async def run_analysis(
    body: AnalyzeRequest,
    session: Session = Depends(connection_manager.create_session),
):
    task = Task(
        match_id=body.match_id,
        video_name=body.video_name,
        state=States.PENDING,
        user_id=body.user_id
    )

    TaskRepository.upsert_task(task, session)
    body.video_name = r"C:\Users\Usuario\Desktop\temp\res\Partido corto 4.mp4"

    orchestrator = Orchestrator()
    orchestrator.run_tasks(body, session, task.id)

    return {
        "status": "queued",
        "task_id": str(task.id),
        "video_name": task.video_name,
        "match_id": task.match_id,
    }
