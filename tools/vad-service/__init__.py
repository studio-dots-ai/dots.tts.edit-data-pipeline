"""
VAD (Voice Activity Detection) Package.

Provides voice activity detection and silence trimming functionality.

Usage:
    from vad_service import detect, trim, get_boundaries

    # Detect speech segments
    result = detect("audio.wav")
    for seg in result.segments:
        print(f"Speech: {seg.start:.2f}s - {seg.end:.2f}s")

    # Trim silence from boundaries
    waveform = trim("audio.wav", target_start_silence=0.1)

    # Get overall speech boundaries
    start, end = get_boundaries("audio.wav")
"""

try:
    from .core import (
        VADResult,
        TrimResult,
        SpeechSegment,
        detect_speech_segments,
        trim_silence,
        get_speech_boundaries,
        waveform_to_wav_bytes,
        load_audio_sf,
    )

    from .api_func import (
        detect,
        detect_to_dict,
        trim,
        trim_with_metadata,
        get_boundaries,
        get_segment_times,
    )
except ImportError:
    from core import (
        VADResult,
        TrimResult,
        SpeechSegment,
        detect_speech_segments,
        trim_silence,
        get_speech_boundaries,
        waveform_to_wav_bytes,
        load_audio_sf,
    )

    from api_func import (
        detect,
        detect_to_dict,
        trim,
        trim_with_metadata,
        get_boundaries,
        get_segment_times,
    )

audio_segment_to_bytes = waveform_to_wav_bytes
load_audio = load_audio_sf

__all__ = [
    # Core
    "VADResult",
    "TrimResult",
    "SpeechSegment",
    "detect_speech_segments",
    "trim_silence",
    "get_speech_boundaries",
    "audio_segment_to_bytes",
    "load_audio",
    # API Functions
    "detect",
    "detect_to_dict",
    "trim",
    "trim_with_metadata",
    "get_boundaries",
    "get_segment_times",
]
