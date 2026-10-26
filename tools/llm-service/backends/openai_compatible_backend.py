"""
OpenAI-compatible HTTP backend for the unified LLM service.

This backend proxies requests to an external server that implements
``/v1/chat/completions`` such as vLLM's OpenAI API server. It is intentionally
kept separate from ``LocalVLLMBackend`` so custom vLLM runtimes can live in
their own environments without contaminating the uv-managed ``pipe-llm`` env.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests

from . import LLMBackend, LLMResponse

logger = logging.getLogger(__name__)


class OpenAICompatibleBackendError(RuntimeError):
    """Raised when an OpenAI-compatible backend request fails."""


class OpenAICompatibleBackend(LLMBackend):
    """Proxy backend for OpenAI-compatible chat-completions servers."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        timeout: float = 300.0,
        max_workers: int = 8,
        enable_thinking: Optional[bool] = None,
    ):
        if not name:
            raise ValueError("name must be provided")
        if not base_url:
            raise ValueError("base_url must be provided")
        if not model:
            raise ValueError("model must be provided")
        if max_workers < 1:
            raise ValueError("max_workers must be >= 1")

        self._name = name.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_workers = max_workers
        self.enable_thinking = enable_thinking

        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=max_workers,
            pool_maxsize=max_workers,
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        logger.info(
            "OpenAI-compatible backend initialized: %s -> %s (model=%s)",
            self._name,
            self.chat_completions_url,
            self.model,
        )

    @property
    def chat_completions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    @property
    def name(self) -> str:
        return self._name

    @property
    def available_models(self) -> List[str]:
        return [self.model]

    @property
    def is_ready(self) -> bool:
        return True

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
        requested_model = model or self.model
        payload = self._build_payload(
            prompt,
            model=requested_model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            stop_sequences=stop_sequences,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
        )
        raw = self._request(payload)
        text = self._parse_response(raw)
        return LLMResponse(
            text=text,
            model=raw.get("model") or requested_model,
            usage=raw.get("usage") or {},
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
        audio_list = audio_base64_list or [None] * len(prompts)

        def _generate_one(args: tuple[str, Optional[str]]) -> LLMResponse:
            prompt, audio = args
            return self.generate(
                prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=json_mode,
                stop_sequences=stop_sequences,
                audio_base64=audio,
                audio_mime_type=audio_mime_type,
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(_generate_one, zip(prompts, audio_list)))

    def _build_payload(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        stop_sequences: Optional[List[str]],
        audio_base64: Optional[str] = None,
        audio_mime_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        content: Any = prompt
        if audio_base64 is not None:
            mime_type = audio_mime_type or "audio/wav"
            content = [
                {
                    "type": "audio_url",
                    "audio_url": {
                        "url": f"data:{mime_type};base64,{audio_base64}"
                    },
                },
                {"type": "text", "text": prompt},
            ]
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if stop_sequences:
            payload["stop"] = stop_sequences
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            response = self.session.post(
                self.chat_completions_url,
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise OpenAICompatibleBackendError(
                f"{self.name} request failed: {exc}"
            ) from exc

        if response.status_code != 200:
            raise OpenAICompatibleBackendError(
                f"{self.name} error {response.status_code}: {response.text[:512]}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise OpenAICompatibleBackendError(
                f"{self.name} returned invalid JSON: {response.text[:512]}"
            ) from exc

        if not isinstance(data, dict):
            raise OpenAICompatibleBackendError(
                f"{self.name} returned non-object JSON: {type(data).__name__}"
            )
        return data

    def _parse_response(self, response: Dict[str, Any]) -> str:
        try:
            choice = response["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise OpenAICompatibleBackendError(
                f"Unexpected {self.name} response schema: {str(response)[:512]}"
            ) from exc

        message = choice.get("message") or {}
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") in (None, "text")
            ]
            return "".join(text_parts)
        if "text" in choice:
            return str(choice["text"])
        raise OpenAICompatibleBackendError(
            f"Unexpected {self.name} choice schema: {str(choice)[:512]}"
        )
