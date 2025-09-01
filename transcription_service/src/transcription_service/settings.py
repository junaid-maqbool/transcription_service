from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application settings
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    # API settings
    max_file_size_mb: int = Field(default=100)
    supported_formats: List[str] = Field(default=["wav", "mp3", "m4a", "flac", "ogg"])
    background_workers: int = Field(default=4)
    # File paths
    temp_dir: Path = Field(default=Path(__file__).parent.resolve())
    # Performance settings
    request_timeout_sec: int = Field(default=300)
    separator_model: str = Field(default="htdemucs_ft")


# Global settings instance
settings = Settings()
settings.temp_dir.mkdir(parents=True, exist_ok=True)
