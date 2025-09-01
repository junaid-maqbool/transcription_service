import multiprocessing
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel
from fastapi import FastAPI
import structlog
from torch import multiprocessing

from src.transcription_service.settings import settings
multiprocessing.set_start_method('spawn', force=True)

class AppState(BaseModel):
    bg_workers_process_pool: Any


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger = structlog.get_logger()
    logger.info("Starting audio transcription service")

    try:
        pool = multiprocessing.Pool(
            processes=settings.background_workers
        )
        app_state = AppState(
            bg_workers_process_pool=pool,
        )
        yield {"app_state": app_state}
        # On app shutdown
        pool.close()
        pool.join()

    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down audio transcription service")

