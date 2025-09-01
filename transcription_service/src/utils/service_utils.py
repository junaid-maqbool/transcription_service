from pathlib import Path

from fastapi import UploadFile, HTTPException
from pydub import AudioSegment
from pydub.exceptions import CouldntDecodeError

from src.transcription_service.settings import settings


def validate_file_name_and_ext(file: UploadFile) -> str:
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file",
                "detail": "No filename provided",
            },
        )
    file_extension = Path(file.filename).suffix.lower().lstrip('.')
    if file_extension not in settings.supported_formats:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_format",
                "detail": f"Unsupported format: {file_extension}. Supported: {settings.supported_formats}",
            },
        )
    return file_extension

def validate_file_size(content) -> float:
    file_size = len(content)
    max_size_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size > max_size_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": f"File size {file_size / 1024 / 1024:.1f}MB exceeds limit of {settings.max_file_size_mb}MB",
            },
        )
    return file_size


def get_audio_duration_and_sample_rate(audio_path: Path) -> tuple[float, int]:
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000.0  # Convert ms to seconds
        sample_rate = audio.frame_rate
        return duration_sec, sample_rate
    except CouldntDecodeError as e:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Unable to decode",
                "detail": f"{e}",
            },
        )
