from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logfire
from core.database import connection_manager
import uvicorn


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application is starting...")
    print("Creating tables...")
    connection_manager.create_database()
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

    return app

app = run_app()

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=4)

    # load_dotenv(".env.local")

    # from src.tinybird.client import tinybird

    # now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

    # # Ingest data using the Events API
    # tinybird.page_views.ingest(
    #     {
    #         "timestamp": now,
    #         "session_id": "abc123",
    #         "pathname": "/home",
    #         "referrer": "https://google.com",
    #     }
    # )

    # # Query the endpoint
    # result = tinybird.top_pages.query(
    #     {
    #         "start_date": "2026-01-01 00:00:00",
    #         "end_date": now,
    #         "limit": 5,
    #     }
    # )

    # for row in result["data"]:
    #     print(row["pathname"], row["views"])


if __name__ == "__main__":
    main()
