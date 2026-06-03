from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx


@dataclass(slots=True)
class TranscriptionResult:
    text: str
    confidence: float


class VoiceInputService:
    def __init__(self, *, backend: str, model_size: str, groq_api_key: str = '', groq_model: str = 'whisper-large-v3-turbo') -> None:
        self.backend = backend
        self.model_size = model_size
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model
        self._local_model = None

    async def transcribe(self, file_path: str) -> TranscriptionResult:
        path = Path(file_path)
        try:
            if self.backend == 'groq':
                return await self._transcribe_groq(path)
            return await self._transcribe_local(path)
        finally:
            if path.exists():
                path.unlink(missing_ok=True)

    async def _transcribe_local(self, path: Path) -> TranscriptionResult:
        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError('Local Whisper backend is not installed') from exc
        if self._local_model is None:
            self._local_model = whisper.load_model(self.model_size)
        result = self._local_model.transcribe(str(path))
        text = (result.get('text') or '').strip()
        confidence = 1.0 if text else 0.0
        return TranscriptionResult(text=text, confidence=confidence)

    async def _transcribe_groq(self, path: Path) -> TranscriptionResult:
        if not self.groq_api_key:
            raise RuntimeError('GROQ_API_KEY is required for groq voice input backend')
        async with httpx.AsyncClient(timeout=30.0) as client:
            with path.open('rb') as handle:
                response = await client.post(
                    'https://api.groq.com/openai/v1/audio/transcriptions',
                    headers={'Authorization': f'Bearer {self.groq_api_key}'},
                    files={'file': (path.name, handle, 'audio/ogg')},
                    data={'model': self.groq_model},
                )
            response.raise_for_status()
        payload = response.json()
        text = payload.get('text', '').strip()
        confidence = 1.0 if text else 0.0
        return TranscriptionResult(text=text, confidence=confidence)
