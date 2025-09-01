from setuptools import find_packages, setup
from pathlib import Path


ROOT_DIR = Path(__file__).parent.resolve()


requirements = [
    "injector",
    "pydantic",
    "pydantic_settings",
    "fastapi",
    "python-multipart",
    "pydub",
    "aiofiles",
    "httpx",
    "structlog",
    "uvicorn",
    "pytest",
    "pytest-asyncio",
    "torch@https://download.pytorch.org/whl/cu118/torch-2.1.0%2Bcu118-cp39-cp39-linux_x86_64.whl",
    "torchvision@https://download.pytorch.org/whl/cu118/torchvision-0.16.0%2Bcu118-cp39-cp39-linux_x86_64.whl",
    "torchaudio@https://download.pytorch.org/whl/cu118/torchaudio-2.1.0%2Bcu118-cp39-cp39-linux_x86_64.whl",
    "dora-search",
    "einops",
    "julius>=0.2.3",
    "lameenc>=1.2",
    "openunmix==1.1.2",
    "pyyaml",
    "tqdm",
    "openai-whisper",
    "numpy==1.22.0",
]
setup(
    name="transcription_service",
    python_requires="==3.9.*",
    packages=find_packages(include=["transcription_service"]),
    install_requires=requirements,
)
