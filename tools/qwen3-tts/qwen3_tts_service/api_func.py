"""
Qwen3-TTS Function API: Direct Python interface (no HTTP).

Mirrors the HTTP API surface but returns raw Python objects.
Useful for in-process calls from the data pipeline.
"""

import io
import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import soundfile as sf

try:
    from .core import (
        SynthesisResult,
        TokenDecodeResult,
        TokenEncodeResult,
        decode_tokens,
        encode_tokens,
        get_model_info,
        get_models,
        init_models,
        synthesize_custom_voice,
        synthesize_voice_clone,
    )
except ImportError:
    from core import (
        SynthesisResult,
        TokenDecodeResult,
        TokenEncodeResult,
        decode_tokens,
        encode_tokens,
        get_model_info,
        get_models,
        init_models,
        synthesize_custom_voice,
        synthesize_voice_clone,
    )

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom-voice synthesis
# ---------------------------------------------------------------------------

def custom_voice(
    text: str,
    speaker: str,
    language: str = "Auto",
    instruct: Optional[str] = None,
    output_format: str = "numpy",
    **gen_kwargs,
) -> Union[Tuple[np.ndarray, int], bytes]:
    """
    Synthesize with a preset speaker.

    Args:
        text: text to speak.
        speaker: preset speaker name.
        language: language code.
        instruct: optional emotion / style instruction.
        output_format: "numpy" → (ndarray, sr);  "wav" → wav bytes.

    Returns:
        Depending on output_format.
    """
    result: SynthesisResult = synthesize_custom_voice(
        text=text, speaker=speaker, language=language, instruct=instruct, **gen_kwargs,
    )
    if output_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, result.audio, result.sample_rate, format="wav")
        return buf.getvalue()
    return result.audio, result.sample_rate


# ---------------------------------------------------------------------------
# Voice-clone synthesis
# ---------------------------------------------------------------------------

def voice_clone(
    text: str,
    language: str = "Auto",
    ref_audio: Any = None,
    ref_text: Optional[str] = None,
    x_vector_only_mode: bool = False,
    output_format: str = "numpy",
    **gen_kwargs,
) -> Union[Tuple[np.ndarray, int], bytes]:
    """
    Clone speaker from reference audio and synthesize.

    Args:
        text: text to speak.
        language: language code.
        ref_audio: reference audio (path, base64, or (ndarray, sr)).
        ref_text: reference transcript (required unless x_vector_only_mode).
        x_vector_only_mode: use speaker embedding only.
        output_format: "numpy" → (ndarray, sr);  "wav" → wav bytes.

    Returns:
        Depending on output_format.
    """
    result: SynthesisResult = synthesize_voice_clone(
        text=text, language=language, ref_audio=ref_audio,
        ref_text=ref_text, x_vector_only_mode=x_vector_only_mode,
        **gen_kwargs,
    )
    if output_format == "wav":
        buf = io.BytesIO()
        sf.write(buf, result.audio, result.sample_rate, format="wav")
        return buf.getvalue()
    return result.audio, result.sample_rate


# ---------------------------------------------------------------------------
# Token encode / decode
# ---------------------------------------------------------------------------

def token_encode(
    audio: Any,
    sr: Optional[int] = None,
) -> TokenEncodeResult:
    """Encode audio to 12Hz tokens. Returns TokenEncodeResult."""
    return encode_tokens(audio, sr=sr)


def token_decode(
    codes: Any,
    output_format: str = "numpy",
) -> Union[TokenDecodeResult, bytes]:
    """
    Decode tokens to audio.

    Args:
        codes: int64 ndarray(s) of shape (T, Q).
        output_format: "numpy" → TokenDecodeResult;  "wav" → wav bytes of first item.

    Returns:
        Depending on output_format.
    """
    result = decode_tokens(codes)
    if output_format == "wav" and result.audios:
        buf = io.BytesIO()
        sf.write(buf, result.audios[0], result.sample_rate, format="wav")
        return buf.getvalue()
    return result
