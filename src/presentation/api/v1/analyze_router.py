import asyncio

from fastapi import APIRouter, Depends
import logfire
from sqlmodel import Session

from src.core.repository.task_repository import TaskRepository
from src.core.tasks.orchestrator import Orchestrator
from src.core.database import connection_manager
from src.entities.models.requests.analyze_request import AnalyzeRequest
from src.entities.models.requests.queue_model import Task
from src.entities.types.states import StatesModel

router = APIRouter(prefix="/analyze", tags=["analyze"])

async def execute_analysis(body: AnalyzeRequest, task_id: str):
    try:
        orchestrator = Orchestrator()

        await asyncio.to_thread(
            orchestrator.run_tasks,
            body,
            task_id
        )

    except Exception as e:
        logfire.exception(
            f"[BackgroundTask] Error processing task {task_id}: {e}"
        )


@router.post("/run", status_code=202)
async def run_analysis(
    body: AnalyzeRequest,
    session: Session = Depends(connection_manager.create_session),
):
    task = Task(match_id=body.match_id, video_name=body.video_name, general_state=StatesModel.PENDING, nickname=body.nickname)

    TaskRepository.upsert_task(task, session)
    logfire.info(f"[Router] Task {task.id} created, response submitted: {body.model_dump()}")
    body.video_name = r"C:\Users\Usuario\Desktop\temp\res\Partido corto 4.mp4"

    orchestrator = Orchestrator()
    asyncio.create_task(execute_analysis(body, str(task.id)))

    return {
        "status": "queued",
        "task_id": str(task.id),
        "video_name": task.video_name,
        "match_id": task.match_id,
    }
