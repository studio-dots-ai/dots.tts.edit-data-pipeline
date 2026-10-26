"""
Speech Rate Conversion Package.

Provides WSOLA-based time stretching using SoX.

Usage:
    # Direct function call (recommended for local use)
    from tools.speech_rate_conversion import stretch
    wav_bytes = stretch(audio_bytes, rate=1.2)
    
    # Or with numpy output
    audio, sr = stretch(audio_array, rate=0.8, sample_rate=22050, output_format="numpy")
    
    # HTTP service (for network access)
    # Run: python -m tools.speech_rate_conversion.speech_rate_service --port 8004
"""

from .core import stretch_audio, stretch_audio_to_bytes, StretchResult
from .api_func import stretch, stretch_with_metadata

__all__ = [
    "stretch",
    "stretch_with_metadata",
    "stretch_audio",
    "stretch_audio_to_bytes",
    "StretchResult",
]
