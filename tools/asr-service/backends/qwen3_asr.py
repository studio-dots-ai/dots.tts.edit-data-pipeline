"""HTTP backend adapter for the standalone Qwen3-ASR vLLM service."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple, Union

import numpy as np
import requests
import soundfile as sf


@dataclass
class ASRResult:
    """Single ASR result."""

    text: str
    language: Optional[str] = None


class Qwen3ASRHTTPBackend:
    """Delegate ASR requests to a ``qwen-asr-serve`` HTTP endpoint."""

    def __init__(
        self,
        *,
        backend_name: str = "qwen3-asr",
        service_url: str = "http://127.0.0.1:8033",
        timeout: float = 300.0,
        max_tokens: int = 512,
    ) -> None:
        self._backend_name = backend_name
        self.service_url = service_url.rstrip("/")
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._ready = False

    def load(self) -> None:
        response = requests.get(f"{self.service_url}/health", timeout=min(self.timeout, 30.0))
        response.raise_for_status()
        self._ready = True

    @property
    def is_loaded(self) -> bool:
        return self._ready

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def inference_name(self) -> str:
        return "vllm-http"

    @property
    def model_path(self) -> str:
        return self.service_url

    def transcribe_batch(
        self,
        audios: List[Union[str, Tuple[np.ndarray, int]]],
        languages: Optional[List[Optional[str]]] = None,
    ) -> List[ASRResult]:
        if languages is None:
            languages = [None] * len(audios)
        return [
            self._transcribe_one(audio=audio, language=language)
            for audio, language in zip(audios, languages)
        ]

    def transcribe_single(
        self,
        audio: Union[str, Tuple[np.ndarray, int]],
        language: Optional[str] = None,
    ) -> ASRResult:
        return self.transcribe_batch([audio], [language])[0]

    def _transcribe_one(
        self,
        *,
        audio: Union[str, Tuple[np.ndarray, int]],
        language: Optional[str],
    ) -> ASRResult:
        audio_url = self._audio_url(audio)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": audio_url},
                    }
                ],
            }
        ]
        forced_language = _forced_language_name(language)
        if forced_language:
            messages.append(
                {
                    "role": "assistant",
                    "content": f"language {forced_language}<asr_text>",
                }
            )
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        response = requests.post(
            f"{self.service_url}/v1/chat/completions",
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = _extract_chat_content(response.json())
        parsed_language, text = _parse_qwen_output(raw)
        return ASRResult(text=text, language=parsed_language or forced_language or language)

    @staticmethod
    def _audio_url(audio: Union[str, Tuple[np.ndarray, int]]) -> str:
        if isinstance(audio, str):
            if audio.startswith(("http://", "https://", "data:")):
                return audio
            path = Path(audio)
            audio_format = path.suffix.lstrip(".") or "wav"
            data = path.read_bytes()
        else:
            wav, sample_rate = audio
            audio_format = "wav"
            with io.BytesIO() as buffer:
                sf.write(buffer, np.asarray(wav, dtype=np.float32), int(sample_rate), format="WAV")
                data = buffer.getvalue()
        encoded = base64.b64encode(data).decode("ascii")
        return f"data:audio/{audio_format};base64,{encoded}"


def _extract_chat_content(payload: dict) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"Invalid Qwen3-ASR response: {payload}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        raise RuntimeError(f"Invalid Qwen3-ASR response: {payload}")
    return str(message.get("content") or "")


def _forced_language_name(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    if not normalized:
        return None
    lower = normalized.lower().replace("_", "-")
    if lower in {"zh", "zho", "chi", "cmn", "cn", "chinese", "mandarin", "zh-cn", "zh-hans"}:
        return "Chinese"
    if lower in {"en", "eng", "english", "en-us", "en-gb"}:
        return "English"
    return normalized


def _parse_qwen_output(raw: str) -> Tuple[str, str]:
    text = str(raw or "").strip()
    if not text:
        return "", ""
    if "<asr_text>" not in text:
        return "", text
    meta, transcript = text.split("<asr_text>", 1)
    if "language none" in meta.lower() and not transcript.strip():
        return "", ""
    match = re.search(r"(?im)^\s*language\s+(.+?)\s*$", meta)
    language = match.group(1).strip() if match else ""
    return language, transcript.strip()
