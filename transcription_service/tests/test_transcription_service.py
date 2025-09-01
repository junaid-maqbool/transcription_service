import json
import multiprocessing
import tempfile
from pathlib import Path

from fastapi import Request
from unittest.mock import Mock, patch
import pytest
from fastapi.testclient import TestClient

from src.app import app
from src.lifespan import lifespan, AppState
from src.transcription_service.schemas import TranscriptionConfig, TranscriptionSegment
from src.transcription_service.transcription_service import TranscriptionService
from src.utils.service_utils import (
    validate_file_name_and_ext,
    validate_file_size,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_wav_content():
    """Create minimal valid WAV file content."""
    # Minimal WAV header + silent audio data
    wav_header = (
        b"RIFF"
        + (1000).to_bytes(4, "little")  # file size - 8
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")  # fmt chunk size
        + (1).to_bytes(2, "little")  # audio format (PCM)
        + (1).to_bytes(2, "little")  # num channels
        + (16000).to_bytes(4, "little")  # sample rate
        + (32000).to_bytes(4, "little")  # byte rate
        + (2).to_bytes(2, "little")  # block align
        + (16).to_bytes(2, "little")  # bits per sample
        + b"data"
        + (960).to_bytes(4, "little")  # data chunk size
        + b"\x00" * 960  # silence data
    )
    return wav_header


class TestHealthAndStatus:
    """Test basic service health endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestTranscriptionConfig:
    """Test configuration handling."""

    def test_default_config(self):
        """Test default configuration."""
        config = TranscriptionConfig()
        assert config.language_hint == "en"
        assert config.enable_separation is True
        assert config.model_size == "small"

    def test_config_from_json(self):
        """Test config creation from JSON."""
        json_str = '{"language_hint": "es", "model_size": "large"}'
        config = TranscriptionConfig.from_json_string(json_str)
        assert config.language_hint == "es"
        assert config.model_size == "large"

    def test_invalid_json_config(self):
        """Test handling of invalid JSON config."""
        with pytest.raises(Exception):
            TranscriptionConfig.from_json_string("invalid json")


class TestFileValidation:
    """Test file validation utilities."""

    def test_valid_audio_formats(self):
        """Test validation of supported audio formats."""
        valid_formats = ["wav", "mp3", "m4a", "flac", "ogg"]

        for fmt in valid_formats:
            mock_file = Mock(filename=f"test.{fmt}")
            ext = validate_file_name_and_ext(mock_file)
            assert ext == fmt

    def test_invalid_audio_format(self):
        """Test rejection of unsupported formats."""
        mock_file = Mock(filename="test.txt")
        with pytest.raises(Exception):
            validate_file_name_and_ext(mock_file)

    def test_file_size_validation(self):
        """Test file size validation."""
        # Valid size
        small_content = b"x" * 1024  # 1KB
        size = validate_file_size(small_content)
        assert size == 1024

        # Too large
        large_content = b"x" * (100 * 1024 * 1024 + 1)  # > 100MB
        with pytest.raises(Exception):
            validate_file_size(large_content)


class TestTranscriptionAPI:
    """Test main transcription API endpoint."""

    def test_transcribe_invalid_file_format(self, client):
        """Test transcription with invalid file format."""
        files = {"file": ("test.txt", b"not audio", "text/plain")}
        data = {"config": json.dumps({})}
        pool = multiprocessing.Pool(8)
        with patch.object(
            Request,
            "state",
            new_callable=lambda: type(
                "obj", (object,), {"app_state": AppState(bg_workers_process_pool=pool)}
            )(),
        ):
            response = client.post("/v1/transcribe/", files=files, data=data)
        assert response.status_code == 400

    def test_transcribe_oversized_file(self, client):
        """Test transcription with file too large."""
        # Create content larger than max size
        large_content = b"x" * (101 * 1024 * 1024)  # 101MB
        files = {"file": ("large.wav", large_content, "audio/wav")}
        data = {"config": json.dumps({})}
        pool = multiprocessing.Pool(8)
        with patch.object(
            Request,
            "state",
            new_callable=lambda: type(
                "obj", (object,), {"app_state": AppState(bg_workers_process_pool=pool)}
            )(),
        ):
            response = client.post("/v1/transcribe/", files=files, data=data)
        assert response.status_code == 413

    def test_transcribe_invalid_config(self, client, sample_wav_content):
        """Test transcription with malformed config."""
        files = {"file": ("test.wav", sample_wav_content, "audio/wav")}
        data = {"config": "invalid json string"}
        pool = multiprocessing.Pool(8)
        with patch.object(
            Request,
            "state",
            new_callable=lambda: type(
                "obj", (object,), {"app_state": AppState(bg_workers_process_pool=pool)}
            )(),
        ):
            response = client.post("/v1/transcribe/", files=files, data=data)
        assert response.status_code == 400


class TestTranscriptionSegments:
    """Test transcription segment handling."""

    def test_segment_creation(self):
        """Test creating transcription segments."""
        segment = TranscriptionSegment(start=0.0, end=5.0, text="test speech")
        assert segment.start == 0.0
        assert segment.end == 5.0
        assert segment.text == "test speech"

    def test_segments_from_whisper_results(self):
        """Test parsing Whisper results into segments."""
        whisper_result = {
            "segments": [
                {"start": 0.0, "end": 2.5, "text": "first part"},
                {"start": 2.5, "end": 5.0, "text": "second part"},
            ]
        }

        segments = TranscriptionSegment.get_segments_from_whisper_results(
            whisper_result
        )
        assert len(segments) == 2
        assert segments[0].text == "first part"
        assert segments[1].text == "second part"


class TestServiceIntegration:
    """High-level integration tests with mocked ML models."""

    @patch("whisper.load_model")
    @patch("demucs.api.Separator")
    @patch("demucs.api.save_audio")
    def test_transcription_service_workflow(
        self, mock_save_audio, mock_separator_class, mock_whisper
    ):
        """Test the complete transcription workflow."""
        # Mock Demucs separator
        mock_separator = Mock()
        mock_separator.samplerate = 16000
        mock_separator.separate_audio_file.return_value = (
            "origin",
            {"vocals": "vocals_tensor"},
        )
        mock_separator_class.return_value = mock_separator

        # Mock Whisper model
        mock_model = Mock()
        mock_model.transcribe.return_value = {
            "text": "this is a test",
            "segments": [{"start": 0.0, "end": 3.0, "text": "this is a test"}],
        }
        mock_whisper.return_value = mock_model

        # Test the service
        service = TranscriptionService()

        # Test vocal separation
        test_audio_path = Path("test.wav")
        vocals_path = service.separate_vocals(test_audio_path, "htdemucs_ft", "cpu")
        assert vocals_path.name == "vocals.wav"

        # Test transcription
        text, segments, load_time = service.transcribe_audio(
            vocals_path, "small.en", "cpu"
        )
        assert text == "this is a test"
        assert len(segments) == 1
        assert isinstance(load_time, float)

    def test_device_selection_cpu(self):
        """Test CPU device selection when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            # Device selection logic would be tested here
            # In actual implementation, check if "cpu" is passed to models
            pass

    def test_device_selection_gpu(self):
        """Test GPU device selection when CUDA available."""
        with patch("torch.cuda.is_available", return_value=True):
            # Device selection logic would be tested here
            # In actual implementation, check if "cuda" is passed to models
            pass


# Configuration for pytest
def pytest_configure():
    """Configure pytest settings."""
    pytest.main_test_dir = Path(__file__).parent


# Run tests with: pytest test_transcription_service.py -v
