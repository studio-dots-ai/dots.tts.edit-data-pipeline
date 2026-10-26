# Pitch Conversion Service

## Overview

Pitch shifting service with four methods, all accessible through the unified `/shift` endpoint:

- **`psola`**: Pure Praat PSOLA overlap-add resynthesis. CPU-only, no external dependencies.
- **`psola+qwen3_codec`** (default): PSOLA pitch shift followed by Qwen3-TTS 12Hz codec encode→decode to remove artifacts.
- **`psola+indextts2_cco`**: PSOLA pitch shift followed by IndexTTS2 CCO resynthesis.
- **`psola+mimo_codec`**: PSOLA pitch shift followed by MiMo-Audio RVQ tokenize→resyn to remove artifacts.

**Port**: 8027 (default)

**Environment**: `envs/pipe-cpu`
**Key dependencies**: parselmouth (Praat), numpy, soundfile

## Launch

```bash
source envs/pipe-cpu/.venv/bin/activate
python -m tools.pitch_conversion.api_service --port 8027

# With custom external service URLs
python -m tools.pitch_conversion.api_service --port 8027 \
    --qwen3tts-url http://localhost:8007 \
    --indextts-url http://localhost:8002 \
    --mimo-url http://localhost:8024
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/shift` | Unified pitch shifting (psola / psola+qwen3_codec / psola+indextts2_cco / psola+mimo_codec) |

---

### POST `/shift`

Unified pitch-shift endpoint. The `method` field selects the processing pipeline.

#### Input (`ShiftRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `semitones` | float | Yes | — | Pitch shift in semitones (-24 to +24). Positive = higher, negative = lower. |
| `method` | str | No | `psola+qwen3_codec` | `"psola"`, `"psola+qwen3_codec"`, `"psola+indextts2_cco"`, or `"psola+mimo_codec"` |
| `audio` | EncodedAudio or str | Yes | — | Input audio: base64-encoded `{data, format}` or file path |
| `text` | str | Conditional | — | Segment transcript. Required for `"psola+indextts2_cco"` |
| `output_format` | str | No | `wav` | `"wav"` or `"numpy"` |

#### Output

- `wav`: Binary WAV with headers `X-Sample-Rate`, `X-Semitones`, `X-Method`, `X-Original-Duration`, `X-Shifted-Duration`
- `numpy`: JSON `ShiftResponse` with `EncodedTensor` + metadata

#### EncodedAudio format

```json
{
    "data": "<base64-encoded-bytes>",
    "format": "wav"
}
```

#### Examples

```bash
# Pure PSOLA pitch shift (+3 semitones)
curl -X POST http://localhost:8027/shift \
  -H "Content-Type: application/json" \
  -d '{
    "audio": {"data": "'$(base64 -w0 input.wav)'", "format": "wav"},
    "semitones": 3,
    "method": "psola"
  }' --output shifted.wav

# Default method (psola+qwen3_codec) — requires Qwen3-TTS on port 8007
curl -X POST http://localhost:8027/shift \
  -H "Content-Type: application/json" \
  -d '{
    "audio": {"data": "'$(base64 -w0 input.wav)'", "format": "wav"},
    "semitones": -5
  }' --output shifted_refined.wav

# IndexTTS2 CCO method — requires text and IndexTTS2 on port 8002
curl -X POST http://localhost:8027/shift \
  -H "Content-Type: application/json" \
  -d '{
    "audio": {"data": "'$(base64 -w0 input.wav)'", "format": "wav"},
    "text": "the words spoken in this segment",
    "semitones": 3,
    "method": "psola+indextts2_cco"
  }' --output shifted_indextts2_cco.wav
```

## Function API

For direct Python usage without HTTP overhead (PSOLA only):

```python
from tools.pitch_conversion import pitch_shift

# Returns WAV bytes
wav_bytes = pitch_shift(audio_bytes, semitones=3.0)

# Returns numpy array + sample rate
audio, sr = pitch_shift(audio_array, semitones=-2.0, sample_rate=16000, output_format="numpy")
```

## Dependencies

- **PSOLA**: No external services. Runs entirely on CPU with parselmouth.
- **psola+qwen3_codec**: Requires Qwen3-TTS service on port 8007 (`/encode_tokens` + `/decode_tokens`).
- **psola+indextts2_cco**: Requires IndexTTS2 service on port 8002 (`/extract_tokens` + `/synthesize_from_tokens`).
- **psola+mimo_codec**: Requires MiMo-Audio service on port 8024 (`/tokenize` + `/resyn`).
