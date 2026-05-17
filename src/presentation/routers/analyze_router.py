from fastapi import APIRouter, Depends
from sqlmodel import Session

from src.core.database import get_session
from src.entities.models.app.analyze_request import AnalyzeRequest
from src.entities.models.app.queue_model import Task
from src.entities.types.states import States

router = APIRouter(prefix="/analyze", tags=["analyze"])


@router.post("/run", status_code=202)
async def run_analysis(
    body: AnalyzeRequest,
    session: Session = Depends(get_session),
):
    task = Task(
        match_id=body.match_id,
        video_name=body.video_name,
        state=States.PENDING,
    )
    session.add(task)
    session.commit()
    session.refresh(task)

    return {
        "status": "queued",
        "task_id": str(task.id),
        "video_name": task.video_name,
        "match_id": task.match_id,
    }