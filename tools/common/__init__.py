"""
Shared models, utilities, and service runner for tool services.

Provides common Pydantic models (EncodedAudio, EncodedTensor, etc.), utility
functions (decode_audio_input, encode_numpy, cleanup_temp_files), and a
``ServiceRunner`` class used across multiple services to eliminate duplication.
"""

from .models import EncodedAudio, EncodedTensor, HealthResponse
from .runner import ServiceRunner
from .utils import (
    decode_audio_input,
    encode_numpy,
    decode_numpy,
    encode_audio_to_base64,
    cleanup_temp_files,
    setup_logging,
)

__all__ = [
    "EncodedAudio",
    "EncodedTensor",
    "HealthResponse",
    "ServiceRunner",
    "decode_audio_input",
    "encode_numpy",
    "decode_numpy",
    "encode_audio_to_base64",
    "cleanup_temp_files",
    "setup_logging",
]
