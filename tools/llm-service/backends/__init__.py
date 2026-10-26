"""
LLM Backend base class and common types.

All LLM backends (Gemini, local vLLM, etc.) implement the LLMBackend interface.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standard response from an LLM backend."""

    text: str
    model: str
    usage: Dict[str, int] = field(
        default_factory=dict
    )  # prompt_tokens, completion_tokens, etc.
    raw: Optional[Any] = None  # Backend-specific raw response


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (e.g., 'gemini', 'local')."""

    @property
    @abstractmethod
    def available_models(self) -> List[str]:
        """List of models available through this backend."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Whether the backend is initialized and ready to serve."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        audio_base64: Optional[str] = None,
        audio_mime_type: Optional[str] = None,
    ) -> LLMResponse:
        """Generate text from a single prompt, optionally with audio input.

        Args:
            prompt: The input prompt string.
            model: Model name override (backend-specific default if None).
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            json_mode: If True, constrain output to valid JSON.
            stop_sequences: Optional stop strings.
            audio_base64: Base64-encoded audio data for a multimodal backend.
            audio_mime_type: MIME type of the audio (e.g. 'audio/wav').
                Defaults to 'audio/wav' if audio_base64 is provided.

        Returns:
            LLMResponse with generated text.
        """

    def generate_batch(
        self,
        prompts: List[str],
        *,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        audio_base64_list: Optional[List[Optional[str]]] = None,
        audio_mime_type: Optional[str] = None,
    ) -> List[LLMResponse]:
        """Generate text for multiple prompts (batch).

        Default implementation calls generate() sequentially.
        Backends may override for efficient batching.

        Args:
            audio_base64_list: Per-prompt base64 audio data (None entries = no audio).
            audio_mime_type: Shared MIME type for all audio inputs.
        """
        audio_list = audio_base64_list or [None] * len(prompts)
        return [
            self.generate(
                p,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                stop_sequences=stop_sequences,
                audio_base64=a,
                audio_mime_type=audio_mime_type,
            )
            for p, a in zip(prompts, audio_list)
        ]
