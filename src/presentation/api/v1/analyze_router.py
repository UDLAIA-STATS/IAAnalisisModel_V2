from fastapi import APIRouter, Depends
import logfire
from sqlmodel import Session

from src.core.repository.task_repository import TaskRepository
from src.core.tasks.orchestrator import Orchestrator
from src.core.database import connection_manager
from entities.models.requests.analyze_request import AnalyzeRequest
from entities.models.requests.queue_model import Task
from src.entities.types.states import StatesModel

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/run", status_code=202)
async def run_analysis(
    body: AnalyzeRequest,
    session: Session = Depends(connection_manager.create_session),
):
    task = Task(match_id=body.match_id, video_name=body.video_name, general_state=StatesModel.PENDING, user_id=body.user_id)

    TaskRepository.upsert_task(task, session)
    logfire.info(f"[Router] Task {task.id} created, response submitted: {body.model_dump()}")
    body.video_name = r"C:\Users\Usuario\Desktop\temp\res\Partido corto 4.mp4"

    orchestrator = Orchestrator()
    orchestrator.run_tasks(body, task.id)
    # asyncio.create_task(
    #     orchestrator.run_tasks(body, task.id)
    # )

    return {
        "status": "queued",
        "task_id": str(task.id),
        "video_name": task.video_name,
        "match_id": task.match_id,
    }
