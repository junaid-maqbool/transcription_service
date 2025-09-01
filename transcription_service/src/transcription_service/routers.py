import uuid
from pathlib import Path
from typing import Optional
import time

import torch
import structlog
from fastapi import APIRouter, UploadFile, Request, File, Form, HTTPException

from src.lifespan import AppState
from src.transcription_service.schemas import (
    TranscriptionConfig,
    TranscriptionResponse,
    PipelineInfo,
    TimingInfo,
)
from src.transcription_service.settings import settings
from src.transcription_service.transcription_service import TranscriptionService
from src.utils.service_utils import (
    validate_file_name_and_ext,
    validate_file_size,
    get_audio_duration_and_sample_rate,
)

logger = structlog.get_logger(__name__)

router = APIRouter()



def transcribe_audio(audio_fp: Path, transcription_config: TranscriptionConfig, request_id: str, duration: float, sample_rate: int) -> TranscriptionResponse:
    start_time = time.time()
    transcription_service = TranscriptionService()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"{device} will be used.",
        request_id=request_id,
    )
    sep_stime = time.time()
    # Process audio
    try:
        vocals_fp = transcription_service.separate_vocals(
            model=settings.separator_model, audio_fp=audio_fp, device=device
        )
    except Exception as e:
        logger.warning(
            "Vocal separation failed, skipping vocal separation step.",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        vocals_fp = audio_fp

    sep_time = time.time() - sep_stime
    transcription_stime = time.time()
    text, segments, load_time = transcription_service.transcribe_audio(
        vocals_fp=vocals_fp,
        transcription_model=f"{transcription_config.model_size}.{transcription_config.language_hint}",
    )
    transcription_time = time.time() - transcription_stime
    # Calculate total time
    total_time = int((time.time() - start_time))
    timing_info = TimingInfo(
        load=int(load_time * 1000),
        separation=int(sep_time * 1000),
        transcription=int(transcription_time * 1000),
        total=int(total_time * 1000),
    )
    pipeline_info = PipelineInfo(
        separation={
            "enables": transcription_config.enable_separation,
            "method": "demucs",
        },
        transcription={"model": transcription_config.model_size},
    )
    response = TranscriptionResponse(
        request_id=request_id,
        duration_sec=duration,
        sample_rate=sample_rate,
        pipeline=pipeline_info,
        text=text,
        language=transcription_config.language_hint,
        timings_ms=timing_info,
        segments=segments,
    )
    vocals_fp.unlink()
    logger.info(
        "Transcription completed successfully",
        request_id=request_id,
        duration_sec=duration,
        segments=len(response.segments),
        total_time_ms=total_time,
    )
    return response


@router.post("/transcribe/")
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file to transcribe"),
    config: Optional[str] = Form(..., description="JSON configuration"),
) -> TranscriptionResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    state: AppState = request.state.app_state
    pool = state.bg_workers_process_pool
    transcription_config: TranscriptionConfig = (
        TranscriptionConfig.from_json_string(config)
    )
    # Validate file and file extension
    file_extension = validate_file_name_and_ext(file=file)
    # Check file size
    content = await file.read()
    file_size = validate_file_size(content=content)
    logger.info(
        "File validation passed",
        request_id=request_id,
        file_size_mb=file_size / 1024 / 1024,
        format=file_extension,
    )
    # Save file temporarily
    audio_fp = settings.temp_dir / f"{request_id}_{uuid.uuid4()}.{file_extension}"
    with open(str(audio_fp), "wb") as audio_file:
        audio_file.write(content)
    audio_duration, sample_rate = get_audio_duration_and_sample_rate(
        audio_path=audio_fp
    )
    try:
        async_response = pool.apply_async(transcribe_audio,
                                                       kwds=dict(
                                                           audio_fp=audio_fp,
                                                           transcription_config=transcription_config,
                                                           request_id=request_id,
                                                           duration=audio_duration,
                                                           sample_rate=sample_rate,
                                                       ))
        transcription_response = async_response.get()
        return transcription_response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Transcription failed",
            request_id=request_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail={
                "request_id": request_id,
                "error": "processing_failed",
                "detail": str(e),
            },
        )
    finally:
        audio_fp.unlink()
