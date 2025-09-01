import time
from pathlib import Path
from typing import List, Tuple

import demucs.api
import whisper

from src.transcription_service.schemas import TranscriptionSegment
from src.transcription_service.settings import settings


class TranscriptionService:

    def separate_vocals(self, audio_fp: Path, model: str, device: str = "cuda") -> Path:
        separator = demucs.api.Separator(model=model, device=device)
        origin, separated = separator.separate_audio_file(file=audio_fp)
        vocals = separated["vocals"]
        vocals_fp = settings.temp_dir / "vocals.wav"
        demucs.api.save_audio(vocals, vocals_fp, samplerate=separator.samplerate)
        return vocals_fp

    def transcribe_audio(
        self,
        vocals_fp: Path,
        transcription_model: str = "tiny.en",
        device: str = "cuda",
    ) -> Tuple[str, List[TranscriptionSegment], float]:
        load_stime = time.time()
        transcriber = whisper.load_model(transcription_model, device=device)
        load_time = time.time() - load_stime
        result = transcriber.transcribe(audio=str(vocals_fp))
        segments = TranscriptionSegment.get_segments_from_whisper_results(
            result_dict=result
        )
        return result["text"], segments, load_time
