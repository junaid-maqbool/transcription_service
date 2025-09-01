# Audio Transcription Service

A containerized microservice that separates vocals from background noise and transcribes speech using state-of-the-art machine learning models. Built with FastAPI, Whisper, and Demucs for high-quality audio processing.

## Features

- **Vocal Separation**: Uses Demucs to isolate vocals from background music/noise
- **Speech Transcription**: Powered by OpenAI Whisper for accurate speech-to-text
- **Multiple Audio Formats**: Supports WAV, MP3, M4A, FLAC, and OGG
- **GPU/CPU Support**: Automatic device detection with fallback to CPU
- **RESTful API**: Clean HTTP API with comprehensive error handling
- **Dockerized**: Fully containerized with Docker and docker-compose
- **Structured Logging**: Request tracking and performance metrics

## Quick Start

### Prerequisites

- Docker
- Git
- Python 3.9

### Running with Docker

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd transcription_service
   ```

2. **Build Docker image**
   ```bash
   sudo DOCKER_BUILDKIT=1 docker build -f transcription_service/Dockerfile -t transcription_service ./
   ```

3. **Run the container**
   ```bash
   sudo docker run  --gpus all -p 8000:8000 transcription_service
   ```

4. **Service will be available at**
   - API: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Running Locally (Development)

1. **Install the service**
   ```bash
   # Install the package (includes all dependencies and Demucs)
   # From repository root dir run following commands
   pip install -e transcription_service
   cd transcription_service/src/external
   git clone https://github.com/adefossez/demucs.git
   pip install -e .
   ```

2. **Run the service**
   ```bash
   python transcription_service/src/main.py
   ```

## API Usage

### Transcribe Audio

**Endpoint**: `POST /v1/transcribe/`

**Request**:
```bash
curl -X POST "http://localhost:8000/v1/transcribe/" \
  -F "file=@your_audio.wav" \
  -F 'config={"language_hint": "en", "model_size": "tiny", "enable_separation": true}'
```

**Response**:
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "duration_sec": 31.2,
  "sample_rate": 16000,
  "pipeline": {
    "separation": {"enabled": true, "method": "demucs"},
    "transcription": {"model": "small"}
  },
  "segments": [
    {"start": 0.0, "end": 3.1, "text": "hello world"}
  ],
  "text": "hello world",
  "language": "en",
  "timings_ms": {
    "load": 420,
    "separation": 1800,
    "transcription": 4100,
    "total": 6400
  }
}
```

### Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `language_hint` | string | "en" | Target language for transcription |
| `enable_separation` | boolean | true | Enable vocal separation |
| `diarize` | boolean | false | Enable speaker diarization (future) |
| `model_size` | string | "small" | Whisper model size (tiny, small, medium, large) |
| `target_sr` | integer | 16000 | Target sample rate |

### Error Responses

- **400**: Invalid file format or configuration
- **413**: File too large (>100MB)
- **422**: Audio decode failure
- **500**: Internal processing error

## Architecture Decision Record (ADR)

### Pipeline Design

The service implements a multi-stage audio processing pipeline:

```
Input Audio → Validation → Normalization → Vocal Separation → Transcription → Response
```

**Stage Details**:
1. **Decode & Validate**: Accept multiple formats, validate size/type
2. **Separation**: Use Demucs to isolate vocals from background noise
3. **Transcription**: Apply Whisper ASR to separated vocals
4. **Format**: Structure response with segments and metadata

### Model Selection Rationale

#### Vocal Separation: Demucs (`htdemucs_ft`)
- **Choice**: Hybrid Transformer Demucs Fine-Tuned
- **Rationale**: Best balance of quality and speed for vocal isolation
- **Trade-offs**: 
  - ✅ High-quality separation, handles music well
  - ❌ Slower than simpler models (~1-2s processing time)
  - ✅ Works on CPU and GPU

#### Speech Recognition: OpenAI Whisper
- **Choice**: Whisper with configurable model sizes
- **Rationale**: State-of-the-art accuracy with multilingual support
- **Model Size Trade-offs**:
  - `tiny`: Fast (~39 MB), lower accuracy
  - `small`: Balanced (~244 MB), good accuracy/speed ratio
  - `medium`: High accuracy (~769 MB), slower
  - `large`: Best accuracy (~1550 MB), slowest

### GPU/CPU Strategy

- **Auto-detection**: Uses `torch.cuda.is_available()` for device selection
- **Fallback**: Graceful degradation to CPU if GPU unavailable
- **Memory Management**: Models loaded per-request (trade-off: cold start vs memory)
- **Future Optimization**: Model caching and warm-up for production

### Concurrency Model

- **FastAPI + Uvicorn**: Async request handling
- **Process Pool**: Background workers for each request
- **Worker Count**: Configurable (default: 8 processes)
- **Isolation**: Each request processed in separate worker process

### Observability

- **Structured Logging**: JSON logs with request IDs and timing
- **Request Tracking**: UUID-based request identification
- **Performance Metrics**: Load, separation, transcription, and total timing
- **Error Context**: Detailed error messages with request correlation

### Failure Modes & Fallbacks

1. **Separation Failure**: 
   - Fallback: Skip separation, process original audio
   - Logging: Clear indication of fallback usage

2. **Model Loading Failure**:
   - Error: Return 500 with specific error message

3. **GPU Memory Issues**:
   - Fallback: Automatic CPU processing
   - Logging: Device fallback notification

4. **File Processing Errors**:
   - Validation: Early rejection of invalid files
   - Cleanup: Automatic temporary file removal

## File Structure

```
├── transcription_service/
│   ├── src/
│   │   ├── external\         # directory to clone any external repositories/models
│   │   ├── tmp\              # directory to save any temporary files during execution
│   │   ├── transcription_service\
│   │   │   ├── routers.py        # API route handlers
│   │   │   ├── schemas.py        # Pydantic models
│   │   │   ├── settings.py       # Configuration management
│   │   │   └── transcription_service.py  # Core ML processing
│   │   ├── utils\      
│   │   │   ├── logging_config.py # Structured logging setup
│   │   │   └── service_utils.py  # Utility functions       
│   │   ├── lifespan.py       # Application lifecycle management
│   │   ├── app.py            # FastApi App
│   │   ├── main.py           # Application entry point
│   ├── tests/
│   │   ├── sample_audio/     # Test audio files
│   ├── └── test_transcription_service.py  # Test suite 
│   ├── Dockerfile        # Multi-stage container definition
│   ├── setup.py          # Package installation with custom Demucs setup
│   ├── README.md         # This file
│   ├── .gitignore        # git ignore paths/patterns
│   └── .dockerignore     # docker ignore paths/patterns    
```

## Sample Audio Files

The `tests/test_audios/` directory contains test files with varying noise conditions:

1. **harvard.wav** 
   - **Noise Level:** Clear speech, no background noise, 18 secs
   - **Performance:** Service transcribed audio perfectly without any error 
2. **song.mp3**\
   - **Noise Level:** Song with loud background instruments/music, 10 secs 
   - **Performance:** Despite loud music noise service transcribed audio with perfectly without any error
3. **poor-audio.ogg** 
   - **Noise Level:** Very high background noise, even difficult to understand vocals for a human, 1 min 45 secs 
   - **Performance:** Despite such a high level noise most of the vocals are transcribed with good accuracy

## Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest transcription_service/tests/test_transcription_service.py -
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **API Tests**: HTTP endpoint validation
- **Error Handling**: Edge cases and failure modes

## Configuration

### Model Configuration

Models are downloaded automatically on first use:
- **Whisper models**: Cached in `~/.cache/whisper/`
- **Demucs models**: Cached in `~/.cache/torch/hub/`

## Performance Considerations

### Latency Expectations
- **Model Loading**: ~500 ms (model loading)
- **Separation**: ~ 4925 seconds per 10s audio with ft version
- **Transcription**: ~ 862 per 10s audio with tiny model
- **Total**: 5000 ms for 10s audio

### Memory Requirements
- **Minimum**: 4GB RAM for small models
- **Recommended**: 8GB+ RAM for medium/large models
- **GPU**: 2GB+ VRAM recommended for faster processing

### Optimization Strategies
- Use smaller Whisper models for faster processing
- Disable separation for clean audio to reduce latency
- Scale horizontally with multiple service instances
- Consider model caching for production deployments

## Development

### Adding New Features

1. **Audio Formats**: Add to `supported_formats` in `settings.py`
2. **Models**: Update model configurations in service classes
3. **API Parameters**: Extend `TranscriptionConfig` schema
4. **Processing Steps**: Modify pipeline in `transcription_service.py`

### Code Quality

The project uses:
- **Formatting**: Black for code formatting
- **Testing**: Pytest with comprehensive coverage

```bash
# Format code
black src/ tests/
```

## Deployment

### Docker Production Build

The service uses a multi-stage Dockerfile optimized for production:

```bash
# Build the image
docker build -t audio-transcription:latest .

# Run container with GPU support (if available)
docker run --gpus all -p 8000:8000 \
  -e LOG_LEVEL=WARNING \
  -e BACKGROUND_WORKERS=8 \
  audio-transcription:latest

# Run container CPU-only
docker run -p 8000:8000 \
  -e LOG_LEVEL=WARNING \
  -e BACKGROUND_WORKERS=4 \
  audio-transcription:latest
```

### Docker Architecture

The Dockerfile uses a multi-stage build with micromamba for efficient dependency management:

1. **Base Stage**: Sets up Ubuntu focal with essential packages
2. **Dependencies Stage**: Installs Python 3.9 and all ML dependencies
3. **Runtime Stage**: Copies built environment and adds FFmpeg
4. **Service Stage**: Final stage with entry point configuration

**Benefits**:
- Smaller final image size
- Efficient layer caching
- Separates build dependencies from runtime
- Uses micromamba for faster conda environment setup

### Scaling Considerations

- **Horizontal**: Deploy multiple instances behind load balancer
- **Vertical**: Increase worker processes and memory allocation
- **GPU**: Use GPU-enabled containers for faster processing
- **Storage**: Mount shared volume for model caching across instances

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce model size (use "tiny" or "small")
   - Decrease worker count
   - Ensure adequate system RAM

2. **Slow Processing**
   - Enable GPU support
   - Use smaller audio files for testing
   - Check CPU/memory utilization

3. **Model Download Failures**
   - Ensure internet connectivity
   - Check disk space for model cache
   - Verify firewall settings

### Logs and Debugging

Enable debug logging:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

Check logs for detailed processing information and error traces.

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI Spec**: `http://localhost:8000/openapi.json`

## License

This project is created for assessment purposes. Please ensure compliance with model licenses:
- **Whisper**: MIT License
- **Demucs**: MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

---

For questions or issues, please open a GitHub issue with detailed information about your environment and the problem encountered.