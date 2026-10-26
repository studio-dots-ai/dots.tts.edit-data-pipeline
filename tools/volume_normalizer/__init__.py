"""
Volume Normalizer Package.

Provides audio loudness normalization using RMS or peak methods.

Usage:
    # Direct function call (recommended for local use)
    from tools.volume_normalizer import normalize
    wav_bytes = normalize(audio_bytes, target_db=-20.0)
    
    # Or with numpy output
    audio, sr = normalize(audio_array, target_db=-18.0, sample_rate=22050, output_format="numpy")
    
    # Batch normalization
    from tools.volume_normalizer import normalize_batch
    normalized_audios = normalize_batch([audio1, audio2], target_db=-20.0)
    
    # HTTP service (for network access)
    # Run: python -m tools.volume_normalizer.volume_normalizer_service --port 8006
"""

from .core import (
    normalize_audio,
    normalize_audio_to_bytes,
    normalize_multiple,
    normalize_to_match,
    NormalizeResult,
    compute_rms_db,
    compute_peak_db,
)
from .api_func import (
    normalize,
    normalize_with_metadata,
    normalize_batch,
    normalize_to_reference,
)

__all__ = [
    # Main functions
    "normalize",
    "normalize_with_metadata",
    "normalize_batch",
    "normalize_to_reference",
    # Core functions
    "normalize_audio",
    "normalize_audio_to_bytes",
    "normalize_multiple",
    "normalize_to_match",
    # Types
    "NormalizeResult",
    # Utilities
    "compute_rms_db",
    "compute_peak_db",
]
