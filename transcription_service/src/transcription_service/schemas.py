import json
from typing import List, Optional, Dict, Any

import structlog
from pydantic import BaseModel, Field
from fastapi import HTTPException

import uuid


logger = structlog.get_logger(__name__)


class TranscriptionConfig(BaseModel):
    language_hint: Optional[str] = Field(default="en")
    enable_separation: bool = Field(default=True)
    diarize: bool = Field(default=False)
    model_size: str = Field(default="small", description="Whisper model size")
    target_sr: int = Field(default=16000, description="Target sample rate")

    @classmethod
    def from_dict(cls, json_dict: Dict[str, Any]) -> "TranscriptionConfig":
        return cls(**json_dict)

    @classmethod
    def from_json_string(cls, json_string: str) -> "TranscriptionConfig":
        try:
            transcription_config = cls.from_dict(json.loads(json_string))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Invalid configuration", error=str(e))
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_config",
                    "detail": f"Invalid JSON configuration: {str(e)}",
                },
            )
        return transcription_config


class PipelineInfo(BaseModel):
    separation: Dict[str, Any]
    transcription: Dict[str, Any]


class TranscriptionSegment(BaseModel):
    start: float
    end: float
    text: str

    @staticmethod
    def get_segments_from_whisper_results(result_dict: Dict[str, Any]) -> List["TranscriptionSegment"]:
        segments = []
        for segment in result_dict["segments"]:
            segments.append(TranscriptionSegment(start=segment["start"], end=segment["end"], text=segment["text"]))
        return segments


class TimingInfo(BaseModel):
    load: int
    separation: int
    transcription: int
    total: int


class TranscriptionResponse(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    duration_sec: float
    sample_rate: int
    pipeline: PipelineInfo
    segments: List[TranscriptionSegment]
    text: str
    language: str
    timings_ms: TimingInfo


class ErrorResponse(BaseModel):
    request_id: str = Field(description="Request ID")
    error: str = Field(description="Error type")
    detail: str = Field(description="Error details")
    status_code: int = Field(description="HTTP status code")
