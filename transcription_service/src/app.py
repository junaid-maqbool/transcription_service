import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.lifespan import lifespan
from src.transcription_service.routers import router
from src.transcription_service.settings import settings
from src.utils.logging_config import setup_logging


# Initialize settings and logging
setup_logging(settings.log_level)

# Create FastAPI app
app = FastAPI(
    title="Audio Transcription Service",
    description="A microservice that separates vocals from background noise and transcribes speech using APIs",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger = structlog.get_logger()
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        error_type=type(exc).__name__,
    )

    return JSONResponse(
        status_code=500,
        content={
            "request_id": request_id,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "audio-transcription"}


@app.get("/")
async def root():
    return {
        "message": "Audio Transcription Service",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
    }


# Include API routes
app.include_router(router, prefix="/v1")
