from contextlib import asynccontextmanager
import signal

import cv2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logfire
import uvicorn

from src.core.repository.task_repository import TaskRepository
from src.entities.utils.spark_instance import graceful_shutdown
from src.core.database import connection_manager
from src.presentation.api.v1.analyze_router import router as analyze_router
from src.config.routes import ensure_directories, validate_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting...")
    print("Creating tables...")
    connection_manager.create_database()
    ensure_directories()
    validate_model()
    logfire.configure(scrubbing=False, service_name="pnl_analyzer", service_version="2.0.0")
    logfire.instrument_fastapi(app)
    # logfire.info(cv2.getBuildInformation())
    logfire.notice("Application started, ready to receive requests")
    yield
    session = connection_manager.create_session()
    TaskRepository.cancel_pending_tasks(session)
    connection_manager.dispose()
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    print("Application is shutting down...")


def run_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(analyze_router)
    return app


app = run_app()


def main():
    try:
        logfire.info("Application starting")
        uvicorn.run("main:app", host="0.0.0.0", port=6070, reload=True)
    except Exception:
        logfire.error("Error starting application shutting down logfire")
    finally:
        logfire.shutdown()


if __name__ == "__main__":
    main()
