"""
Pitch Conversion Package.

Provides PSOLA-based pitch shifting using Praat (parselmouth).

Usage:
    # Direct function call (recommended for local use)
    from tools.pitch_conversion import pitch_shift
    wav_bytes = pitch_shift(audio_bytes, semitones=3.0)

    # Or with numpy output
    audio, sr = pitch_shift(audio_array, semitones=-2.0, sample_rate=16000, output_format="numpy")

    # HTTP service (for network access, supports codec refinement)
    # Run: python -m tools.pitch_conversion.api_service --port 8027
"""

from .core import shift_audio, shift_audio_to_bytes, PitchShiftResult
from .api_func import pitch_shift, pitch_shift_with_metadata

__all__ = [
    "pitch_shift",
    "pitch_shift_with_metadata",
    "shift_audio",
    "shift_audio_to_bytes",
    "PitchShiftResult",
]
