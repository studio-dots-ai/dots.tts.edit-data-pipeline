"""
Service Clients: HTTP clients for all TTS/VC services.

Provides unified interfaces for calling various audio processing services.
"""

import base64
import io
import json
import logging
import os
import time
from typing import List, Optional, Dict, Any, Union, Tuple

import numpy as np

import requests

logger = logging.getLogger(__name__)


def encode_audio_bytes(audio_bytes: bytes, audio_format: str = "wav") -> dict:
    """Encode audio bytes to base64 for API transmission."""
    return {
        "data": base64.b64encode(audio_bytes).decode("utf-8"),
        "format": audio_format,
    }


def encode_numpy_tensor(arr: np.ndarray) -> dict:
    """Encode numpy array to base64 for API transmission."""
    return {
        "data": base64.b64encode(arr.tobytes()).decode("utf-8"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def decode_tensor_response(tensor_dict: dict) -> np.ndarray:
    """Decode tensor from API response."""
    data = base64.b64decode(tensor_dict["data"])
    arr = np.frombuffer(data, dtype=np.dtype(tensor_dict["dtype"]))
    return arr.reshape(tensor_dict["shape"])


class ServiceError(Exception):
    """Exception raised when a service call fails."""

    pass


# -----------------------------------------
# Retry helper for transient HTTP failures
# -----------------------------------------


def _request_with_retry(
    method: str,
    url: str,
    *,
    max_attempts: int = 3,
    backoff_base: float = 2.0,
    backoff_max: float = 30.0,
    **kwargs,
):
    """HTTP request with exponential-backoff retry on transient errors.

    Retries on: ConnectionError, ReadTimeout, ConnectionRefusedError,
    and HTTP 502/503/504 responses.  Non-retryable errors (4xx, etc.)
    are raised immediately.

    Args:
        method: HTTP method ('get' or 'post').
        url: Full URL.
        max_attempts: Total attempts (including the first).
        backoff_base: Base for exponential backoff (seconds).
        backoff_max: Cap on sleep between retries.
        **kwargs: Forwarded to requests.get / requests.post.

    Returns:
        requests.Response on success.

    Raises:
        The last exception if all attempts are exhausted.
    """
    last_exc = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = getattr(requests, method)(url, **kwargs)
            # Retry on gateway errors (service restarting)
            if resp.status_code in (502, 503, 504) and attempt < max_attempts:
                logger.warning(
                    f"[retry] {method.upper()} {url} returned {resp.status_code}, "
                    f"attempt {attempt}/{max_attempts}"
                )
                sleep_time = min(backoff_base**attempt, backoff_max)
                time.sleep(sleep_time)
                continue
            return resp
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.ReadTimeout,
            ConnectionRefusedError,
        ) as exc:
            last_exc = exc
            if attempt < max_attempts:
                sleep_time = min(backoff_base**attempt, backoff_max)
                logger.warning(
                    f"[retry] {method.upper()} {url} failed ({type(exc).__name__}), "
                    f"attempt {attempt}/{max_attempts}, sleeping {sleep_time:.1f}s"
                )
                time.sleep(sleep_time)
            else:
                logger.error(
                    f"[retry] {method.upper()} {url} failed after {max_attempts} attempts: {exc}"
                )
    raise last_exc


# -----------------------------------------
# TED-TTS Client
# -----------------------------------------


class TedTTSClient:
    """Client for TED-TTS tagged synthesis service."""

    def __init__(self, base_url: str = "http://localhost:8035"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def adapt_tagged(self, tagged_text: str, timeout: Optional[float] = None) -> dict:
        resp = requests.post(
            f"{self.base_url}/adapt_tagged",
            json={"tagged_text": tagged_text},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"TED-TTS tagged adaptation failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()

    def synthesize_tagged(
        self,
        tagged_text: str,
        spk_audio_bytes: bytes,
        *,
        output_format: str = "wav",
        timeout: Optional[float] = None,
        **kwargs,
    ) -> Union[bytes, dict]:
        payload = {
            "tagged_text": tagged_text,
            "spk_audio_prompt": encode_audio_bytes(spk_audio_bytes),
            "output_format": output_format,
            **kwargs,
        }
        resp = requests.post(
            f"{self.base_url}/synthesize_tagged", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"TED-TTS tagged synthesis failed: {resp.status_code} - {resp.text}"
            )
        if output_format == "json":
            return resp.json()
        return resp.content


# -----------------------------------------
# IndexTTS Client
# -----------------------------------------


class IndexTTSClient:
    """Client for IndexTTS synthesis service."""

    def __init__(self, base_url: str = "http://localhost:8002"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        """Check service health."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _parse_multipart_response(
        content: bytes, content_type: str
    ) -> Tuple[bytes, dict]:
        """Parse multipart/mixed response into (wav_bytes, json_dict).

        The server uses boundary ``----IndexTTSCacheBoundary``.  Each part is
        delimited by ``--<boundary>`` lines.  Part 1 is binary WAV, Part 2 is
        JSON.

        The JSON part may be either:
        - Raw cache dict (legacy: return_cache=True, return_tokens=False)
        - Metadata dict with ``cache`` and/or ``tokens`` keys (new format:
          when return_tokens=True)

        Returns (wav_bytes, json_dict) where json_dict is the parsed JSON.
        """
        import json as _json
        import re

        # Extract boundary from Content-Type header
        m = re.search(r"boundary=(.+)", content_type)
        if not m:
            raise ServiceError("Multipart response missing boundary in Content-Type")
        boundary = m.group(1).strip()

        # Split on boundary bytes
        boundary_bytes = f"--{boundary}".encode()
        parts = content.split(boundary_bytes)
        # parts[0] is empty (before first boundary), last part is '--\r\n'
        # We expect exactly 2 data parts
        wav_bytes = None
        json_dict = None
        for part in parts:
            part = part.strip()
            if not part or part == b"--":
                continue
            # Split headers from body at first double CRLF
            sep = b"\r\n\r\n"
            if sep not in part:
                continue
            header_section, body = part.split(sep, 1)
            header_str = header_section.decode("utf-8", errors="replace").lower()
            if "audio/wav" in header_str:
                wav_bytes = body
            elif "application/json" in header_str:
                json_dict = _json.loads(body.decode("utf-8"))
        if wav_bytes is None:
            raise ServiceError("Multipart response missing WAV part")
        if json_dict is None:
            raise ServiceError("Multipart response missing JSON part")
        return wav_bytes, json_dict

    def synthesize(
        self,
        text: str,
        spk_audio_bytes: bytes,
        *,
        emo_audio_bytes: Optional[bytes] = None,
        emo_text: Optional[str] = None,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: float = 0.8,
        output_format: str = "wav",
        timeout: Optional[float] = None,
        trace_id: Optional[str] = None,
        return_cache: bool = False,
        cache: Optional[dict] = None,
        return_tokens: bool = False,
        skip_vocode: bool = False,
        **kwargs,
    ) -> Union[bytes, np.ndarray, Tuple, dict]:
        """
        Synthesize speech with emotion control.

        Args:
            text: Text to synthesize
            spk_audio_bytes: Speaker reference audio bytes
            emo_audio_bytes: Emotion reference audio bytes (mutually exclusive with emo_text/emo_vector)
            emo_text: Emotion text prompt (mutually exclusive with emo_audio_bytes/emo_vector)
            emo_vector: 8-dim emotion vector (mutually exclusive with emo_audio_bytes/emo_text)
            emo_alpha: Emotion intensity control. Defaults to the service's
                public baseline of 0.8 when no explicit value is supplied.
            output_format: 'wav' or 'numpy'
            timeout: Optional HTTP timeout in seconds
            trace_id: Caller-supplied trace ID for request tracing
            return_cache: If True, include conditioning cache in the result
            cache: Precomputed conditioning cache dict from a previous call.
                If the cache contains a 'continuation' field with tokens/text,
                the service will use them as prefix for continuation mode.
            return_tokens: If True, include generated semantic tokens in result
            skip_vocode: If True, skip audio synthesis and return only tokens.
                Implies return_tokens=True. Returns a dict with 'tokens' and
                'token_rate' keys (and optionally 'cache').
            **kwargs: Additional parameters (top_p, top_k, temperature, etc.)

        Returns:
            The return type depends on the combination of flags:

            - Default: WAV bytes or numpy array
            - return_cache=True: (audio, cache_dict)
            - return_tokens=True: (audio, tokens_dict) where tokens_dict has
              'tokens' (np.ndarray int64) and 'token_rate' (float)
            - return_cache=True + return_tokens=True: (audio, cache_dict, tokens_dict)
            - skip_vocode=True: dict with 'tokens', 'token_rate', and optionally 'cache'
        """
        payload = {
            "text": text,
            "spk_audio_prompt": encode_audio_bytes(spk_audio_bytes),
            "output_format": output_format,
            **kwargs,
        }

        payload["emo_alpha"] = emo_alpha
        if emo_audio_bytes is not None:
            payload["emo_audio_prompt"] = encode_audio_bytes(emo_audio_bytes)
        elif emo_text is not None:
            payload["emo_text"] = emo_text
        elif emo_vector is not None:
            payload["emo_vector"] = emo_vector

        if trace_id is not None:
            payload["trace_id"] = trace_id
        if return_cache:
            payload["return_cache"] = True
        if cache is not None:
            payload["cache"] = cache
        if return_tokens:
            payload["return_tokens"] = True
        if skip_vocode:
            payload["skip_vocode"] = True

        resp = requests.post(
            f"{self.base_url}/synthesize", json=payload, timeout=timeout
        )

        if resp.status_code != 200:
            raise ServiceError(
                f"IndexTTS synthesis failed: {resp.status_code} - {resp.text}"
            )

        # --- skip_vocode: JSON response with tokens only ---
        if skip_vocode:
            data = resp.json()
            result = {
                "tokens": decode_tensor_response(data["tokens"]).astype(np.int64),
                "token_rate": data.get("token_rate", 50.0),
            }
            if "cache" in data:
                result["cache"] = data["cache"]
            return result

        # --- WAV output ---
        if output_format == "wav":
            content_type = resp.headers.get("Content-Type", "")
            if "multipart/mixed" in content_type:
                wav_bytes, json_dict = self._parse_multipart_response(
                    resp.content, content_type
                )
                # Determine format: new metadata format vs legacy raw cache
                has_metadata_wrapper = "tokens" in json_dict or "cache" in json_dict
                if has_metadata_wrapper:
                    # New format: {"cache": {...}, "tokens": {...}, "token_rate": ...}
                    cache_dict = json_dict.get("cache")
                    tokens_dict = None
                    if "tokens" in json_dict:
                        tokens_dict = {
                            "tokens": decode_tensor_response(
                                json_dict["tokens"]
                            ).astype(np.int64),
                            "token_rate": json_dict.get("token_rate", 50.0),
                        }
                    if return_cache and return_tokens:
                        return wav_bytes, cache_dict, tokens_dict
                    elif return_cache:
                        return wav_bytes, cache_dict
                    elif return_tokens:
                        return wav_bytes, tokens_dict
                    return wav_bytes
                else:
                    # Legacy format: JSON is the raw cache dict
                    if return_cache:
                        return wav_bytes, json_dict
                    return wav_bytes
            # No multipart — plain WAV
            if return_cache:
                return resp.content, None
            return resp.content

        # --- numpy output ---
        else:
            data = resp.json()
            audio = decode_tensor_response(data["tensor"])
            cache_dict = data.get("cache")
            tokens_dict = None
            if data.get("tokens") is not None:
                tokens_dict = {
                    "tokens": decode_tensor_response(data["tokens"]).astype(np.int64),
                    "token_rate": data.get("token_rate", 50.0),
                }
            if return_cache and return_tokens:
                return audio, cache_dict, tokens_dict
            elif return_cache:
                return audio, cache_dict
            elif return_tokens:
                return audio, tokens_dict
            return audio

    def extract_tokens(
        self,
        audio_bytes: bytes,
    ) -> Tuple[np.ndarray, float]:
        """
        Extract 50Hz semantic tokens from audio.

        Args:
            audio_bytes: Input audio WAV bytes.

        Returns:
            (codes, token_rate) where codes is 1-D int64 ndarray and token_rate≈50.
        """
        payload = {
            "audio": encode_audio_bytes(audio_bytes),
        }
        resp = requests.post(f"{self.base_url}/extract_tokens", json=payload)
        if resp.status_code != 200:
            raise ServiceError(
                f"IndexTTS extract_tokens failed: {resp.status_code} - {resp.text}"
            )
        data = resp.json()
        codes = decode_tensor_response(data["codes"]).astype(np.int64)
        return codes, data["token_rate"]

    def synthesize_from_tokens(
        self,
        codes: np.ndarray,
        text: str,
        spk_audio_bytes: bytes,
        *,
        emo_text: Optional[str] = None,
        emo_vector: Optional[List[float]] = None,
        emo_alpha: float = 0.8,
        output_format: str = "wav",
        timeout: Optional[float] = None,
        return_cache: bool = False,
        cache: Optional[dict] = None,
        mel_audio_bytes: Optional[bytes] = None,
        style_audio_bytes: Optional[bytes] = None,
        spk_cond_audio_bytes: Optional[bytes] = None,
        seed: Optional[int] = None,
    ) -> Union[bytes, np.ndarray, Tuple[bytes, dict], Tuple[np.ndarray, dict]]:
        """
        Synthesize audio from pre-computed 50Hz semantic tokens.

        Args:
            codes: 1-D int64 numpy array of semantic codes.
            text: Corresponding text.
            spk_audio_bytes: Speaker reference audio bytes (base for all pathways).
            emo_text: Emotion text prompt.
            emo_vector: 8-dim emotion vector.
            emo_alpha: Emotion intensity control.
            output_format: 'wav' or 'numpy'.
            timeout: Optional HTTP timeout in seconds.
            return_cache: If True, return (audio, cache_dict) tuple.
            cache: Precomputed conditioning cache from a previous call.
            mel_audio_bytes: Override audio for mel spectrogram extraction.
                When None, spk_audio_bytes is used.
            style_audio_bytes: Override audio for CAM++ style vector extraction.
                When None, spk_audio_bytes is used.
            spk_cond_audio_bytes: Override audio for w2v-bert spk_cond + s2mel_prompt
                extraction.  spk_cond and s2mel_prompt are always kept together.
                When None, spk_audio_bytes is used.
            seed: Optional random seed for deterministic inference (passed to service).

        Returns:
            If return_cache=False: WAV bytes or numpy array.
            If return_cache=True: (audio, cache_dict) tuple.
        """
        payload = {
            "codes": encode_numpy_tensor(codes.astype(np.int64)),
            "text": text,
            "spk_audio_prompt": encode_audio_bytes(spk_audio_bytes),
            "output_format": output_format,
        }
        payload["emo_alpha"] = emo_alpha
        if emo_text is not None:
            payload["emo_text"] = emo_text
        elif emo_vector is not None:
            payload["emo_vector"] = emo_vector
        if return_cache:
            payload["return_cache"] = True
        if cache is not None:
            payload["cache"] = cache
        if mel_audio_bytes is not None:
            payload["mel_audio_source"] = encode_audio_bytes(mel_audio_bytes)
        if style_audio_bytes is not None:
            payload["style_audio_source"] = encode_audio_bytes(style_audio_bytes)
        if spk_cond_audio_bytes is not None:
            payload["spk_cond_audio_source"] = encode_audio_bytes(spk_cond_audio_bytes)
        if seed is not None:
            payload["seed"] = seed

        resp = requests.post(
            f"{self.base_url}/synthesize_from_tokens", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"IndexTTS synthesize_from_tokens failed: {resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            content_type = resp.headers.get("Content-Type", "")
            if "multipart/mixed" in content_type:
                wav_bytes, json_dict = self._parse_multipart_response(
                    resp.content, content_type
                )
                # synthesize_from_tokens always uses legacy format (raw cache dict)
                return (wav_bytes, json_dict) if return_cache else wav_bytes
            if return_cache:
                return resp.content, None
            return resp.content
        else:
            data = resp.json()
            audio = decode_tensor_response(data["tensor"])
            if return_cache:
                return audio, data.get("cache")
            return audio

    def resynth_with_cco(
        self,
        psola_audio_bytes: bytes,
        original_audio_bytes: bytes,
        text: str,
        *,
        seed: Optional[int] = None,
    ) -> bytes:
        """Resynthesise audio using IndexTTS2 with the CCO pathway configuration.

        CCO means:
          - spk_cond (w2v-bert) + s2mel_prompt: from **C**onverted (psola_audio_bytes)
          - style (CAM++):                       from **C**onverted (psola_audio_bytes)
          - mel reference (S2Mel CFM):           from **O**riginal  (original_audio_bytes)

        Semantic content tokens are always extracted from ``psola_audio_bytes``.
        ``original_audio_bytes`` serves as the base speaker prompt (``spk_audio_prompt``
        in the request) and as the mel reference.

        This configuration was found to best preserve pitch shifts produced by PSOLA
        while maintaining speaker identity close to the original recording.

        Args:
            psola_audio_bytes: WAV bytes of the PSOLA pitch-shifted audio.
                Tokens are extracted from this; spk_cond + style pathways use it.
            original_audio_bytes: WAV bytes of the original (unshifted) reference audio.
                Used as ``spk_audio_prompt`` base and mel pathway source.
            text: Transcript corresponding to the audio content.
            seed: Optional random seed for deterministic inference.

        Returns:
            WAV bytes of the resynthesized audio.
        """
        # Step 1 — extract semantic tokens from the pitch-shifted audio
        codes, _ = self.extract_tokens(psola_audio_bytes)

        # Step 2 — build optional synthesis kwargs
        kwargs: dict = {
            "spk_cond_audio_bytes": psola_audio_bytes,  # C
            "style_audio_bytes": psola_audio_bytes,  # C
            # mel_audio_bytes is omitted → service falls back to spk_audio_prompt
            # (original_audio_bytes) for the mel pathway                          # O
        }
        if seed is not None:
            kwargs["seed"] = seed

        # Step 3 — synthesize
        wav_bytes = self.synthesize_from_tokens(
            codes,
            text,
            original_audio_bytes,  # spk_audio_prompt = original (base + mel source)
            output_format="wav",
            **kwargs,
        )
        return wav_bytes  # type: ignore[return-value]


# -----------------------------------------
# Speech Rate Client
# -----------------------------------------


class SpeechRateClient:
    """Client for the unified speech rate conversion service.

    Supports five methods via a single ``stretch()`` call:
      - auto (default): picks best method based on rate
      - wsola:             SoX WSOLA (signal-level)
      - indextts2:         IndexTTS2 50Hz token stretch + re-synthesis
      - qwen3_tts:         Qwen3-TTS 12Hz token stretch + decode
      - wsola+indextts2:   WSOLA stretch + IndexTTS2 re-synthesis at 1.0x
    """

    def __init__(self, base_url: str = "http://localhost:8004"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def stretch(
        self,
        rate: float,
        method: str = "auto",
        *,
        audio_bytes: Optional[bytes] = None,
        codes: Optional[np.ndarray] = None,
        token_rate: Optional[float] = None,
        text: Optional[str] = None,
        spk_audio_bytes: Optional[bytes] = None,
        output_format: str = "wav",
    ) -> Union[bytes, np.ndarray]:
        """Unified time-stretch call.

        Args:
            rate: Speed factor (>1 = faster, <1 = slower).
            method: 'auto', 'wsola', 'indextts2', 'qwen3_tts', or
                'wsola+indextts2'. Default 'auto' picks the best method
                based on the rate.
            audio_bytes: Input WAV bytes (required for wsola; optional for
                token methods when *codes* is supplied).
            codes: Pre-extracted token array (optional shortcut for token
                methods to skip the extraction step).
            token_rate: Token rate in Hz (required when codes is given).
            text: Text content (required for indextts2/auto/combined).
            spk_audio_bytes: Speaker reference WAV bytes (required for
                indextts2/auto/combined).
            output_format: 'wav' or 'numpy'.

        Returns:
            WAV bytes (default) or numpy array.
        """
        payload: dict = {
            "rate": rate,
            "method": method,
            "output_format": output_format,
        }
        if audio_bytes is not None:
            payload["audio"] = encode_audio_bytes(audio_bytes)
        if codes is not None:
            payload["codes"] = encode_numpy_tensor(codes.astype(np.int64))
        if token_rate is not None:
            payload["token_rate"] = token_rate
        if text is not None:
            payload["text"] = text
        if spk_audio_bytes is not None:
            payload["spk_audio"] = encode_audio_bytes(spk_audio_bytes)

        resp = requests.post(f"{self.base_url}/stretch", json=payload, timeout=300)
        if resp.status_code != 200:
            raise ServiceError(
                f"Speech rate stretch failed ({method}): {resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            return resp.content
        data = resp.json()
        return decode_tensor_response(data["tensor"])


# -----------------------------------------
# F5-TTS Client (for smoothing)
# -----------------------------------------


class F5TTSClient:
    """
    Client for F5-TTS synthesis service.

    Used for boundary smoothing via edit_spans_sec parameter.
    """

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        """Check service health."""
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def sample(
        self,
        cond_waveform: np.ndarray,
        text: List[str],
        duration: Optional[int] = None,
        *,
        duration_sec: Optional[float] = None,
        edit_spans_sec: Optional[List[List[float]]] = None,
        preserve_unmasked_waveform: Optional[bool] = None,
        preserve_spans_sec: Optional[List[List[float]]] = None,
        input_sample_rate: Optional[int] = None,
        source_crossfade_sec: Optional[float] = None,
        regen_crossfade_sec: Optional[float] = None,
        steps: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        seed: Optional[int] = None,
        output_format: str = "wav",
        trace_id: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Union[bytes, np.ndarray, dict]:
        """
        Run F5-TTS sample inference.

        Args:
            cond_waveform: Conditioning waveform as numpy array [nw] or [1, nw]
            text: List of text strings
            duration: Target duration in service-internal frames
            duration_sec: Target duration in seconds, converted by the service
            edit_spans_sec: Time spans to regenerate [[start, end], ...]
            preserve_unmasked_waveform: Preserve unedited waveform regions.
            preserve_spans_sec: Core spans to splice from generated waveform.
            input_sample_rate: Sample rate of raw conditioning waveform.
            source_crossfade_sec: Crossfade outside edited spans.
            regen_crossfade_sec: Crossfade inside regenerated spans.
            steps: Number of denoising steps
            cfg_strength: CFG strength
            sway_sampling_coef: Sway sampling coefficient
            seed: Random seed
            output_format: 'wav', 'numpy', 'mel', or 'both'
            trace_id: Caller-supplied trace ID for request tracing

        Returns:
            Based on output_format:
            - 'wav': WAV bytes
            - 'numpy': numpy waveform array
            - 'mel': dict with encoded mel tensor
            - 'both': dict with mel and waveform tensors
        """
        # Ensure 2D shape [1, nw]
        if cond_waveform.ndim == 1:
            cond_waveform = cond_waveform.reshape(1, -1)

        # Encode conditioning tensor
        cond_tensor = {
            "data": base64.b64encode(cond_waveform.astype(np.float32).tobytes()).decode(
                "utf-8"
            ),
            "shape": list(cond_waveform.shape),
            "dtype": "float32",
        }

        payload = {
            "cond": cond_tensor,
            "text": text,
            "steps": steps,
            "cfg_strength": cfg_strength,
            "sway_sampling_coef": sway_sampling_coef,
            "output_format": output_format,
        }

        if duration is not None:
            payload["duration"] = duration
        if duration_sec is not None:
            payload["duration_sec"] = duration_sec

        if edit_spans_sec is not None:
            payload["edit_spans_sec"] = edit_spans_sec
        if preserve_unmasked_waveform is not None:
            payload["preserve_unmasked_waveform"] = preserve_unmasked_waveform
        if preserve_spans_sec is not None:
            payload["preserve_spans_sec"] = preserve_spans_sec
        if input_sample_rate is not None:
            payload["input_sample_rate"] = input_sample_rate
        if source_crossfade_sec is not None:
            payload["source_crossfade_sec"] = source_crossfade_sec
        if regen_crossfade_sec is not None:
            payload["regen_crossfade_sec"] = regen_crossfade_sec

        if seed is not None:
            payload["seed"] = seed

        if trace_id is not None:
            payload["trace_id"] = trace_id

        resp = requests.post(f"{self.base_url}/sample", json=payload, timeout=timeout)

        if resp.status_code != 200:
            raise ServiceError(
                f"F5-TTS sample failed: {resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            return resp.content
        else:
            return resp.json()

    def smooth_boundaries(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        text: str,
        boundary_times: List[float],
        margin_sec: float = 0.3,
        source_crossfade_sec: float = 0.0,
        regen_crossfade_sec: float = 0.01,
        steps: int = 32,
        seed: Optional[int] = None,
    ) -> bytes:
        """
        Smooth audio at specified boundary times using F5-TTS editing.

        Args:
            waveform: Audio waveform [nw]
            sample_rate: Sample rate of waveform
            text: Full text of the audio
            boundary_times: List of boundary times (seconds) to smooth
            margin_sec: Margin around each boundary to regenerate
            source_crossfade_sec: F5 preserve splice crossfade outside edited spans
            regen_crossfade_sec: F5 preserve splice crossfade inside regenerated spans
            steps: Denoising steps
            seed: Random seed

        Returns:
            Smoothed audio as WAV bytes
        """
        target_sr = 24000
        hop_length = 256

        if sample_rate != target_sr:
            raise ValueError(
                f"F5-TTS boundary smoothing expects {target_sr} Hz input, got {sample_rate} Hz"
            )

        # Create edit spans around boundaries
        edit_spans = []
        for bt in boundary_times:
            start = max(0, bt - margin_sec)
            end = bt + margin_sec
            edit_spans.append([start, end])

        # Calculate duration in frames
        num_frames = (len(waveform) // hop_length) + 1

        # Call F5-TTS
        return self.sample(
            cond_waveform=waveform,
            text=[text],
            duration=num_frames,
            edit_spans_sec=edit_spans,
            preserve_unmasked_waveform=True,
            preserve_spans_sec=edit_spans,
            input_sample_rate=target_sr,
            source_crossfade_sec=source_crossfade_sec,
            regen_crossfade_sec=regen_crossfade_sec,
            steps=steps,
            seed=seed,
            output_format="wav",
        )


# -----------------------------------------
# Alignment Client
# -----------------------------------------


class AlignmentClient:
    """Client for alignment services.

    Defaults to the Qwen3 aligner gateway.
    """

    def __init__(self, base_url: str = "http://localhost:8029"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def align(
        self,
        audio_bytes: bytes,
        text: str,
        *,
        language: Optional[str] = None,
        backend: str = "auto",
        word_level: bool = False,
        separate_punctuation: bool = False,
        queries: Optional[List[str]] = None,
        return_response: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform forced alignment between audio and text.

        Args:
            audio_bytes: Audio WAV bytes.
            text: Text to align.
            language: Language code (auto-detected if None).
            queries: Optional tagged region queries to resolve via the backend.
            return_response: If True, return the full service response instead of
                just ``alignments``.

        Returns:
            By default, a list of dicts with keys like ``char``, ``start``, and
            ``end``. If ``queries`` is provided or ``return_response`` is True,
            returns the full response dict so callers can access fields such as
            ``answers`` and ``query_tokenization_matches_default``.
        """
        payload = {
            "backend": backend,
            "audio": encode_audio_bytes(audio_bytes),
            "text": text,
            "word_level": word_level,
            "separate_punctuation": separate_punctuation,
        }
        if language is not None:
            payload["language"] = language
        if queries is not None:
            payload["queries"] = queries

        resp = requests.post(f"{self.base_url}/align", json=payload)
        if resp.status_code != 200:
            raise ServiceError(f"Alignment failed: {resp.status_code} - {resp.text}")
        data = resp.json()
        if queries is not None or return_response:
            return data
        return data["alignments"]

    def align_batch(
        self,
        items: List[Dict[str, Any]],
        *,
        backend: str = "auto",
    ) -> List[Dict[str, Any]]:
        """Batch forced alignment through the aligner gateway.

        Each item accepts the same keys as ``align`` except ``audio`` may be
        raw WAV bytes. The returned list contains the service's per-item
        response objects.
        """
        payload_items: List[Dict[str, Any]] = []
        for item in items:
            payload_item = dict(item)
            audio = payload_item.get("audio")
            if isinstance(audio, (bytes, bytearray)):
                payload_item["audio"] = encode_audio_bytes(bytes(audio))
            payload_items.append(payload_item)
        payload = {
            "backend": backend,
            "items": payload_items,
        }
        resp = requests.post(f"{self.base_url}/align_batch", json=payload)
        if resp.status_code != 200:
            raise ServiceError(
                f"Alignment batch failed: {resp.status_code} - {resp.text}"
            )
        data = resp.json()
        results = data.get("results")
        if not isinstance(results, list):
            raise ServiceError(f"Alignment batch returned invalid payload: {data}")
        return results


# -----------------------------------------
# ASR Client
# -----------------------------------------


class ASRClient:
    """Client for the general ASR service."""

    def __init__(self, base_url: str = "http://localhost:8010", backend: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.backend = backend

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def transcribe(
        self,
        audio_bytes: bytes,
        *,
        language: Optional[str] = None,
        audio_format: str = "wav",
        timeout: float = 120,
    ) -> Dict[str, Any]:
        """Transcribe a single audio.

        Args:
            audio_bytes: Audio file bytes.
            language: Language hint (e.g., 'Chinese', 'English'). None = auto.
            audio_format: Audio format string.
            timeout: Request timeout in seconds.

        Returns:
            Dict with 'text' and 'language' keys.
        """
        payload = {
            "audio": encode_audio_bytes(audio_bytes, audio_format),
            "language": language,
        }
        if self.backend:
            payload["backend"] = self.backend
        resp = _request_with_retry(
            "post", f"{self.base_url}/transcribe", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"ASR transcribe failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()

    def transcribe_batch(
        self,
        audio_bytes_list: List[bytes],
        *,
        languages: Optional[List[Optional[str]]] = None,
        audio_format: str = "wav",
        timeout: float = 300,
    ) -> List[Dict[str, Any]]:
        """Transcribe a batch of audios.

        Returns:
            List of dicts with 'text' and 'language' keys.
        """
        if languages is None:
            languages = [None] * len(audio_bytes_list)

        items = []
        for ab, lang in zip(audio_bytes_list, languages):
            item = {
                "audio": encode_audio_bytes(ab, audio_format),
                "language": lang,
            }
            if self.backend:
                item["backend"] = self.backend
            items.append(item)
        resp = requests.post(
            f"{self.base_url}/transcribe_batch",
            json={"items": items},
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(f"ASR batch failed: {resp.status_code} - {resp.text}")
        return resp.json()["results"]


# -----------------------------------------
# LLM Client
# -----------------------------------------


class LLMClient:
    """Client for the unified LLM Generation Service."""

    def __init__(self, base_url: str = "http://localhost:8012"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def info(self) -> dict:
        resp = requests.get(f"{self.base_url}/info")
        resp.raise_for_status()
        return resp.json()

    def generate(
        self,
        prompt: str,
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        audio_path: Optional[str] = None,
        audio_base64: Optional[str] = None,
        audio_mime_type: Optional[str] = None,
        timeout: float = 300,
    ) -> str:
        """Generate text from a single prompt, optionally with audio input.

        Args:
            prompt: The input prompt string.
            backend: Optional explicit backend. When omitted, the LLM service
                uses its configured default backend.
            model: Optional model name override.
            temperature: Sampling temperature.
            max_tokens: Maximum output tokens.
            json_mode: Constrain output to valid JSON.
            stop_sequences: Optional stop strings.
            audio_path: Path to an audio file. Auto-encoded to base64 and MIME
                type inferred from extension. Mutually exclusive with audio_base64.
            audio_base64: Pre-encoded base64 audio data.
            audio_mime_type: MIME type of the audio (e.g. 'audio/wav').
                Auto-inferred from audio_path extension if not provided.
            timeout: Request timeout in seconds.

        Returns:
            Generated text string.
        """
        # Auto-encode audio from file path
        if audio_path and not audio_base64:
            import base64
            from pathlib import Path

            p = Path(audio_path)
            audio_base64 = base64.b64encode(p.read_bytes()).decode()
            if not audio_mime_type:
                suffix = p.suffix.lower()
                mime_map = {
                    ".wav": "audio/wav",
                    ".mp3": "audio/mpeg",
                    ".ogg": "audio/ogg",
                    ".opus": "audio/ogg",
                    ".flac": "audio/flac",
                }
                audio_mime_type = mime_map.get(suffix, "audio/wav")

        payload: Dict[str, Any] = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        if backend is not None:
            payload["backend"] = backend
        if model is not None:
            payload["model"] = model
        if stop_sequences is not None:
            payload["stop_sequences"] = stop_sequences
        if audio_base64 is not None:
            payload["audio_base64"] = audio_base64
        if audio_mime_type is not None:
            payload["audio_mime_type"] = audio_mime_type

        resp = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(f"LLM generate failed: {resp.status_code} - {resp.text}")
        return resp.json()["text"]

    def generate_with_template(
        self,
        template: str,
        template_vars: Dict[str, Any],
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        timeout: float = 300,
    ) -> str:
        """Generate text using a template + variables (server-side rendering).

        Args:
            template: Template string with {placeholders}.
            template_vars: Dict of placeholder values.

        Returns:
            Generated text string.
        """
        payload: Dict[str, Any] = {
            "template": template,
            "template_vars": template_vars,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        if backend is not None:
            payload["backend"] = backend
        if model is not None:
            payload["model"] = model

        resp = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(f"LLM generate failed: {resp.status_code} - {resp.text}")
        return resp.json()["text"]

    def generate_batch(
        self,
        prompts: List[str],
        *,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        json_mode: bool = False,
        stop_sequences: Optional[List[str]] = None,
        timeout: float = 600,
    ) -> List[str]:
        """Generate text for multiple prompts.

        Returns:
            List of generated text strings.
        """
        payload: Dict[str, Any] = {
            "prompts": prompts,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
        }
        if backend is not None:
            payload["backend"] = backend
        if model is not None:
            payload["model"] = model
        if stop_sequences is not None:
            payload["stop_sequences"] = stop_sequences

        resp = requests.post(
            f"{self.base_url}/generate_batch",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"LLM batch generate failed: {resp.status_code} - {resp.text}"
            )
        return [r["text"] for r in resp.json()["results"]]


# -----------------------------------------
# Boundary Pause Smoother Client
# -----------------------------------------


class BoundaryPauseSmootherClient:
    """Client for Boundary Pause Smoother service (port 8018).

    Each point declares a ``mode``.

    ``smooth_only`` skips aligner, VAD, and pause adjustment, and directly
    regenerates a seam-centered region around ``position_sec``.

    ``adjust`` resolves or consumes a native pause interval, adjusts its
    duration, then regenerates the adjusted pause region.
    """

    def __init__(self, base_url: str = "http://localhost:8018"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def smooth(
        self,
        audio_bytes: bytes,
        text: str,
        points: List[Dict],
        *,
        aligner_backend: Optional[str] = None,
        language: Optional[str] = None,
        f5_steps: int = 32,
        f5_cfg_strength: float = 2.0,
        f5_sway_sampling_coef: float = -1.0,
        f5_seed: Optional[int] = None,
        output_format: str = "wav",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        audio_format: str = "wav",
        timeout: float = 120,
        trace_id: Optional[str] = None,
    ) -> bytes:
        """Smooth audio at specified boundary points.

        Args:
            audio_bytes: Input audio bytes (WAV/MP3/FLAC).
            text: Full text corresponding to the audio.
            points: List of smoothing point specs, each a dict with:
                - mode (str, optional): "smooth_only" (default) or "adjust".
                - position_sec (float, optional): Boundary position in seconds.
                - pause_interval_sec ([float, float], optional): Native pause interval.
                - max_pause_sec (float, optional): Maximum adjusted pause duration.
                - min_pause_sec (float, optional): Minimum adjusted pause duration.
                - regen_margin_sec (float, optional): Extra masked context on each
                  side of the adjusted pause interval. In audio_insert mode this
                  instead means the left/right speech transition duration to
                  regenerate around the insert interval.
                - audio_insert (dict, optional): Encoded audio object
                  {"data": base64, "format": "wav"} that replaces the protected
                  pause interval after F5-TTS.
            f5_steps: F5-TTS denoising steps.
            f5_cfg_strength: F5-TTS CFG strength.
            output_format: "wav" or "numpy".
            debug: Whether to save intermediate artifacts.
            debug_dir: Directory for debug output.
            aligner_backend: Optional explicit aligner backend for pause interval
                resolution ("qwen3" or "auto").
            language: Optional language code passed to the aligner gateway.
            trace_id: Caller-supplied trace ID for request tracing.

        Returns:
            Smoothed WAV audio bytes.
        """
        payload = {
            "audio": encode_audio_bytes(audio_bytes, audio_format),
            "text": text,
            "points": points,
            "f5_steps": f5_steps,
            "f5_cfg_strength": f5_cfg_strength,
            "f5_sway_sampling_coef": f5_sway_sampling_coef,
            "output_format": output_format,
            "debug": debug,
        }
        if aligner_backend is not None:
            payload["aligner_backend"] = aligner_backend
        if language is not None:
            payload["language"] = language
        if f5_seed is not None:
            payload["f5_seed"] = f5_seed
        if debug_dir is not None:
            payload["debug_dir"] = debug_dir
        if trace_id is not None:
            payload["trace_id"] = trace_id

        resp = requests.post(f"{self.base_url}/smooth", json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise ServiceError(
                f"Boundary Pause Smoother failed: {resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            return resp.content
        else:
            return resp.json()


# -----------------------------------------
# Audio Assembly Client
# -----------------------------------------


class AudioAssemblyClient:
    """Client for Audio Assembly service (port 8021).

    Splices source and target audio according to `<target>...</target>` markup,
    using the CTC Aligner for timestamps and Audio Smoothing for transitions.
    """

    def __init__(self, base_url: str = "http://localhost:8021"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def assemble(
        self,
        source_audio_bytes: bytes,
        target_audio_bytes: bytes,
        edit_tagged_text: str,
        *,
        language: Optional[str] = None,
        aligner_backend: Optional[str] = None,
        regen_margin_sec: Optional[float] = None,
        smooth_continuation_boundaries: bool = True,
        f5_steps: int = 32,
        f5_cfg_strength: float = 2.0,
        f5_sway_sampling_coef: float = -1.0,
        f5_seed: Optional[int] = None,
        output_format: str = "wav",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        audio_format: str = "wav",
        timeout: float = 300,
        trace_id: Optional[str] = None,
        time_spans_from_source: Optional[List[Tuple[float, float]]] = None,
        time_spans_from_target: Optional[List[Tuple[float, float]]] = None,
        preserve_outer_boundaries: bool = True,
    ) -> bytes:
        """Assemble audio according to `<target>...</target>` edit markup.

        Args:
            source_audio_bytes: Source audio WAV bytes.
            target_audio_bytes: Target audio WAV bytes.
            edit_tagged_text: Full transcript with `<target>...</target>` spans
                marking content to take from target audio.
            language: Language code (auto-detected if None).
            aligner_backend: Optional explicit aligner backend for timestamp
                queries ("qwen3" or "auto").
            regen_margin_sec: Optional smoothing regen margin. Omit to use the
                Boundary Pause Smoother default ``regen_margin_sec=0.15``.
            smooth_continuation_boundaries: Whether to smooth internal splice
                points.
            f5_steps: F5-TTS denoising steps.
            f5_cfg_strength: F5-TTS CFG strength.
            f5_sway_sampling_coef: F5-TTS sway coefficient.
            f5_seed: Random seed for F5-TTS.
            output_format: "wav" (audio bytes) or "json" (with metadata).
            debug: Save intermediate artifacts.
            debug_dir: Directory for debug output.
            trace_id: Caller-supplied trace ID.
            time_spans_from_source: Optional source-audio spans for plain-text pieces.
            time_spans_from_target: Optional target-audio spans for raw target pieces.
            preserve_outer_boundaries: When True, keep the full first and last
                assembled segments instead of trimming them to alignment/span bounds.

        Returns:
            WAV audio bytes (if output_format="wav") or dict (if "json").
        """
        payload = {
            "source_audio": encode_audio_bytes(source_audio_bytes, audio_format),
            "target_audio": encode_audio_bytes(target_audio_bytes, audio_format),
            "edit_tagged_text": edit_tagged_text,
            "preserve_outer_boundaries": preserve_outer_boundaries,
            "smoothing": {
                "smooth_continuation_boundaries": smooth_continuation_boundaries,
                "f5_steps": f5_steps,
                "f5_cfg_strength": f5_cfg_strength,
                "f5_sway_sampling_coef": f5_sway_sampling_coef,
            },
            "output_format": output_format,
            "debug": debug,
        }

        if language is not None:
            payload["language"] = language
        if aligner_backend is not None:
            payload["aligner_backend"] = aligner_backend
        if regen_margin_sec is not None:
            payload["smoothing"]["regen_margin_sec"] = regen_margin_sec
        else:
            payload["smoothing"].pop("regen_margin_sec", None)
        if f5_seed is not None:
            payload["smoothing"]["f5_seed"] = f5_seed
        if debug_dir is not None:
            payload["debug_dir"] = debug_dir
        if trace_id is not None:
            payload["trace_id"] = trace_id
        if time_spans_from_source is not None:
            payload["time_spans_from_source"] = time_spans_from_source
        if time_spans_from_target is not None:
            payload["time_spans_from_target"] = time_spans_from_target

        resp = requests.post(f"{self.base_url}/assemble", json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise ServiceError(
                f"Audio Assembly failed: {resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            return resp.content
        else:
            return resp.json()


# -----------------------------------------
# UTMOS Client
# -----------------------------------------


class UTMOSClient:
    """Client for UTMOS speech quality MOS evaluation service (port 8017)."""

    def __init__(self, base_url: str = "http://localhost:8017"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def predict(
        self,
        audio_bytes: bytes,
        *,
        timeout: float = 30,
    ) -> dict:
        """Predict MOS score for a single audio.

        Returns:
            Dict with keys: score (1-5), sample_rate, duration_seconds,
            inference_time_seconds.
        """
        payload = {
            "audio_data": base64.b64encode(audio_bytes).decode("utf-8"),
        }
        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise ServiceError(
                f"UTMOS predict failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()

    def predict_batch(
        self,
        audio_list: List[bytes],
        *,
        timeout: float = 120,
    ) -> dict:
        """Predict MOS scores for multiple audio files.

        Returns:
            Dict with keys: scores (list of floats), count, avg_score,
            inference_time_seconds.
        """
        payload = {
            "audio_list": [base64.b64encode(ab).decode("utf-8") for ab in audio_list],
        }
        resp = requests.post(
            f"{self.base_url}/predict_batch", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"UTMOS predict_batch failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()


class SpkSimClient:
    """Client for speaker similarity evaluation service (port 8019)."""

    def __init__(self, base_url: str = "http://localhost:8019"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def similarity(
        self,
        audio_a: bytes,
        audio_b: bytes,
        *,
        timeout: float = 30,
    ) -> dict:
        """Compute cosine similarity between two audio samples.

        Returns:
            Dict with keys: similarity (-1.0 to 1.0), inference_time_seconds.
        """
        payload = {
            "audio_a": base64.b64encode(audio_a).decode("utf-8"),
            "audio_b": base64.b64encode(audio_b).decode("utf-8"),
        }
        resp = requests.post(
            f"{self.base_url}/similarity", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"SpkSim similarity failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()

    def similarity_batch(
        self,
        ref_audio: bytes,
        audio_list: List[bytes],
        *,
        timeout: float = 120,
    ) -> dict:
        """Compare a reference audio against a batch of audios.

        Returns:
            Dict with keys: similarities, count, avg_similarity,
            inference_time_seconds.
        """
        payload = {
            "audio_ref": base64.b64encode(ref_audio).decode("utf-8"),
            "audio_list": [base64.b64encode(ab).decode("utf-8") for ab in audio_list],
        }
        resp = requests.post(
            f"{self.base_url}/similarity_batch", json=payload, timeout=timeout
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"SpkSim similarity_batch failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()


# -----------------------------------------
# WER/CER Client
# -----------------------------------------


class WERClient:
    """Client for the WER/CER evaluation service (port 8011)."""

    def __init__(self, base_url: str = "http://localhost:8011"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def compute(
        self,
        reference: str,
        hypothesis: str,
        *,
        language: str = "auto",
        timeout: float = 30,
    ) -> dict:
        payload = {
            "reference": reference,
            "hypothesis": hypothesis,
            "language": language,
        }
        resp = requests.post(
            f"{self.base_url}/compute_wer",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"WER/CER compute failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()


class WDTWClient:
    """Client for the WDTW metric service (port 8032)."""

    def __init__(self, base_url: str = "http://localhost:8032"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def compute(
        self,
        *,
        source_audio: Optional[Union[bytes, str, Dict[str, Any]]] = None,
        target_audio: Optional[Union[bytes, str, Dict[str, Any]]] = None,
        source_transcript: Optional[str] = None,
        target_transcript: Optional[str] = None,
        edit_st_inst: Optional[str] = None,
        preservation_st_inst: Optional[str] = None,
        source_alignments: Optional[List[Dict[str, Any]]] = None,
        target_alignments: Optional[List[Dict[str, Any]]] = None,
        language: str = "eng",
        mode: str = "preserved",
        include_segments: bool = False,
        timeout: float = 180,
    ) -> dict:
        def audio_payload(audio: Union[bytes, str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
            if isinstance(audio, bytes):
                return encode_audio_bytes(audio, "wav")
            return audio

        if (
            source_audio is None
            or target_audio is None
        ) and (
            source_alignments is None
            or target_alignments is None
        ):
            raise ValueError(
                "Provide either source_audio/target_audio or source_alignments/target_alignments"
            )

        payload: Dict[str, Any] = {
            "language": language,
            "mode": mode,
            "include_segments": include_segments,
        }
        if source_audio is not None:
            payload["source_audio"] = audio_payload(source_audio)
        if target_audio is not None:
            payload["target_audio"] = audio_payload(target_audio)
        if source_transcript is not None:
            payload["source_transcript"] = source_transcript
        if target_transcript is not None:
            payload["target_transcript"] = target_transcript
        if edit_st_inst is not None:
            payload["edit_st_inst"] = edit_st_inst
        if preservation_st_inst is not None:
            payload["preservation_st_inst"] = preservation_st_inst
        if source_alignments is not None:
            payload["source_alignments"] = source_alignments
        if target_alignments is not None:
            payload["target_alignments"] = target_alignments
        resp = requests.post(
            f"{self.base_url}/compute_wdtw",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"WDTW compute failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()


# -----------------------------------------
# F0 Eval Client
# -----------------------------------------


class F0EvalClient:
    """Client for the F0 evaluation service (port 8034)."""

    def __init__(self, base_url: str = "http://localhost:8034"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def analyze(
        self,
        audio_bytes: bytes,
        *,
        spans: Optional[List[Dict[str, Any]]] = None,
        f0_min_hz: float = 50.0,
        f0_max_hz: float = 600.0,
        timeout: float = 60,
    ) -> dict:
        payload: Dict[str, Any] = {
            "audio": encode_audio_bytes(audio_bytes),
            "f0_min_hz": f0_min_hz,
            "f0_max_hz": f0_max_hz,
        }
        if spans is not None:
            payload["spans"] = spans
        resp = requests.post(
            f"{self.base_url}/analyze",
            json=payload,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise ServiceError(
                f"F0 analyze failed: {resp.status_code} - {resp.text}"
            )
        return resp.json()


# -----------------------------------------
# Pitch Conversion Client
# -----------------------------------------


class PitchConversionClient:
    """Client for the pitch conversion service (port 8027).

    Supports three methods via a single ``shift()`` call:
      - psola:              Pure Praat PSOLA overlap-add (CPU-only)
      - psola+qwen3_codec:  PSOLA + Qwen3-TTS 12Hz codec roundtrip (default)
      - psola+indextts2_cco: PSOLA + IndexTTS2 CCO resynthesis
    """

    METHODS = (
        "psola",
        "psola+qwen3_codec",
        "psola+indextts2_cco",
    )

    def __init__(self, base_url: str = "http://localhost:8027"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> dict:
        resp = requests.get(f"{self.base_url}/health", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def shift(
        self,
        semitones: float,
        audio_bytes: bytes,
        *,
        method: str = "psola+qwen3_codec",
        text: Optional[str] = None,
        output_format: str = "wav",
        timeout: float = 300,
    ) -> Union[bytes, np.ndarray]:
        """Pitch-shift audio by the given number of semitones.

        Args:
            semitones: Pitch shift in semitones (-24 to +24).
                Positive = higher pitch, negative = lower pitch.
            audio_bytes: Input WAV audio bytes.
            method: Pitch shifting method.
                'psola' — pure PSOLA (CPU, no external deps).
                'psola+qwen3_codec' (default) — PSOLA + Qwen3-TTS codec
                    refinement (best UTMOS).
                'psola+indextts2_cco' — PSOLA + IndexTTS2 CCO resynthesis
                    (requires text).
            text: Segment transcript, required by 'psola+indextts2_cco'.
            output_format: 'wav' returns WAV bytes, 'numpy' returns ndarray.
            timeout: Request timeout in seconds.

        Returns:
            WAV bytes (default) or numpy array.
        """
        payload: dict = {
            "semitones": semitones,
            "audio": encode_audio_bytes(audio_bytes),
            "method": method,
            "output_format": output_format,
        }
        if text is not None:
            payload["text"] = text

        resp = requests.post(f"{self.base_url}/shift", json=payload, timeout=timeout)
        if resp.status_code != 200:
            raise ServiceError(
                f"Pitch conversion shift failed ({method}): "
                f"{resp.status_code} - {resp.text}"
            )

        if output_format == "wav":
            return resp.content
        data = resp.json()
        return decode_tensor_response(data["tensor"])
