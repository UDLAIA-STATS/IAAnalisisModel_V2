from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logfire
import uvicorn

from src.core.database import connection_manager
from src.presentation.api.v1.analyze_router import router as analyze_router
from src.config.routes import ensure_directories

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting...")
    print("Creating tables...")
    connection_manager.create_database()
    ensure_directories()
    yield
    connection_manager.dispose()
    print("Application is shutting down...")


def run_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    logfire.configure()
    logfire.instrument_fastapi(app)
    
    app.include_router(analyze_router)
    return app


app = run_app()


def main():
    try:
        logfire.info("Application starting")
        uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)
    except Exception as e:
        logfire.error(f"Error starting application shutting down logfire")
    finally:
        logfire.shutdown()

if __name__ == "__main__":
    main()
