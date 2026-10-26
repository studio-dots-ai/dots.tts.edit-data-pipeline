"""
Gemini API backend for the unified LLM service.

Uses a configurable Gemini-compatible endpoint with retries and connection pooling.
"""

import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from . import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class GeminiAPIError(RuntimeError):
    """Raised when the Gemini API responds with an error."""


class GeminiTokenLimitError(GeminiAPIError):
    """Raised when the API rejects a request due to token limits (retriable rate limit)."""


class GeminiTokenBudgetExhausted(GeminiAPIError):
    """Raised when thinking model exhausts token budget before producing output (non-retriable)."""


VALID_MODELS = ["3flash", "2_5pro"]

class GeminiBackend(LLMBackend):
    """Gemini API backend with automatic retries and connection pooling."""

    def __init__(
        self,
        model: str = "3flash",
        url: Optional[str] = None,
        timeout: float = 120.0,
        max_retries: int = 100,
        max_retry_cycles: int = 100,
        retry_backoff: Tuple[float, float] = (1.0, 3.0),
        retry_cycle_sleep_sec: float = 180.0,
    ):
        if model not in VALID_MODELS:
            raise ValueError(f"Unknown Gemini model '{model}'. Valid: {VALID_MODELS}")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if max_retry_cycles < 1:
            raise ValueError("max_retry_cycles must be >= 1")
        self.model = model
        self.api_key = os.environ.get(f"GEMINI_{model.upper()}_API_KEY", os.environ.get("GEMINI_API_KEY", ""))
        self.url = url or os.environ.get("GEMINI_API_URL", "")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_retry_cycles = max_retry_cycles
        self.retry_backoff = retry_backoff
        self.retry_cycle_sleep_sec = retry_cycle_sleep_sec

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=32,
            pool_maxsize=32,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

        logger.info(f"Gemini backend initialized (model: {self.model})")

    # ---- LLMBackend interface ----

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def available_models(self) -> List[str]:
        return [self.model]

    @property
    def is_ready(self) -> bool:
        return bool(self.api_key and self.url)

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
        if not self.is_ready:
            raise GeminiAPIError("Set GEMINI_API_URL and GEMINI_API_KEY (or the model-specific API key)")

        payload = self._build_payload(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
        )
        text, _attempts, raw = self._request(payload)
        return LLMResponse(
            text=text,
            model=self.model,
            usage={},
            raw=raw,
        )

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
        # Gemini doesn't have native batching; sequential calls
        audio_list = audio_base64_list or [None] * len(prompts)
        return [
            self.generate(
                p,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                stop_sequences=stop_sequences,
                audio_base64=a,
                audio_mime_type=audio_mime_type,
            )
            for p, a in zip(prompts, audio_list)
        ]

    # ---- Internal methods ----

    def _build_payload(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        audio_base64: Optional[str] = None,
        audio_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        generation_config: Dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens > 0:
            generation_config["maxOutputTokens"] = max_tokens
        if json_mode:
            generation_config["responseMimeType"] = "application/json"

        parts: List[Dict[str, Any]] = []

        # Audio part (inline_data) — must come before text for Gemini
        if audio_base64:
            mime = audio_mime_type or "audio/wav"
            parts.append(
                {
                    "inline_data": {
                        "data": audio_base64,
                        "mime_type": mime,
                    }
                }
            )

        parts.append({"text": prompt})

        return {
            "contents": [
                {
                    "role": "user",
                    "parts": parts,
                }
            ],
            "generationConfig": generation_config,
        }

    def _parse_response(self, response: Dict[str, Any]) -> str:
        if "Code" in response and "Error" in response:
            message = str(response.get("Error"))
            code = response.get("Code")
            if code == 10001 or "too many tokens" in message.lower():
                raise GeminiTokenLimitError(message)
            raise GeminiAPIError(f"Gemini error {code}: {message}")

        # Check for MAX_TOKENS finish reason with no output content.
        # Gemini 2.5 Pro (thinking model) may consume all tokens for
        # internal reasoning, leaving zero for the actual response.
        try:
            candidate = response["candidates"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiAPIError(
                f"Unexpected response schema: {json.dumps(response)[:512]}"
            ) from exc

        finish_reason = candidate.get("finishReason", "")
        content = candidate.get("content", {})
        parts = content.get("parts")

        if finish_reason == "MAX_TOKENS" and not parts:
            raise GeminiTokenBudgetExhausted(
                "Model exhausted token budget (likely used for thinking) "
                "before producing output. Increase max_tokens."
            )

        if parts is None:
            raise GeminiAPIError(
                f"Unexpected response schema: {json.dumps(response)[:512]}"
            )

        text_parts = [p.get("text", "") for p in parts if isinstance(p, dict)]
        combined = "\n".join(filter(None, text_parts)).strip()
        if not combined:
            raise GeminiAPIError(f"Empty response: {json.dumps(response)[:512]}")
        return combined

    def _request(self, payload: Dict[str, Any]) -> Tuple[str, int, Optional[Dict]]:
        """Make API request with retry logic.

        Returns:
            (response_text, attempts_used, raw_response_json)
        """
        total_attempts = 0
        last_error: Optional[Exception] = None

        for cycle in range(1, self.max_retry_cycles + 1):
            for attempt_in_cycle in range(1, self.max_retries + 1):
                total_attempts += 1
                try:
                    resp = self.session.post(
                        self.url,
                        headers=self.headers,
                        json=payload,
                        timeout=self.timeout,
                    )
                    if resp.status_code >= 400:
                        raise GeminiAPIError(
                            f"HTTP {resp.status_code}: {resp.text[:512]}"
                        )
                    resp_json = resp.json()
                    text = self._parse_response(resp_json)
                    return text, total_attempts, resp_json
                except GeminiTokenBudgetExhausted:
                    raise  # Deterministic failure — retrying won't help
                except Exception as exc:
                    last_error = exc
                    if attempt_in_cycle < self.max_retries:
                        sleep_time = random.uniform(*self.retry_backoff)
                        logger.debug(
                            "Gemini transient failure; sleeping %.1fs before retry: %s",
                            sleep_time,
                            exc,
                        )
                        time.sleep(sleep_time)

            if cycle < self.max_retry_cycles:
                logger.warning(
                    "Gemini request cycle failed; sleeping before the next cycle: %s",
                    last_error,
                )
                time.sleep(self.retry_cycle_sleep_sec)

        raise GeminiAPIError(
            "Request failed after retry budget was exhausted"
        ) from last_error
