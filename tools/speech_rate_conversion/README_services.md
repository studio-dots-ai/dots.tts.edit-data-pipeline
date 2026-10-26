# Speech Rate Conversion Service

## Overview

Time stretching service with three methods, all accessible through the unified `/stretch` endpoint:

- **`wsola`**: Signal-level time stretching via SoX tempo effect. Preserves pitch while changing speed.
- **`indextts2`**: Extract IndexTTS2 50Hz discrete tokens → nearest-neighbor resample → re-synthesize.
- **`qwen3_tts`**: Encode with Qwen3-TTS 12Hz codec → nearest-neighbor resample → decode.

**Port**: 8004 (default)


**Environment**: `envs/pipe-cpu`
**Python**: 3.10
**Key dependencies**: numpy, sox, torch (for token ops)

## Launch

```bash
source envs/pipe-cpu/.venv/bin/activate
python -m tools.speech_rate_conversion.api_service --port 8004
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/stretch` | Unified time stretching (wsola / indextts2 / qwen3_tts) |

---

### POST `/stretch`

Unified time-stretch endpoint. The `method` field selects the stretching backend.

#### Input (`StretchRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `rate` | float | Yes | — | Speed factor: >1 = faster/shorter, <1 = slower/longer |
| `method` | str | No | `wsola` | `"wsola"`, `"indextts2"`, or `"qwen3_tts"` |
| `audio` | EncodedAudio or str | Cond. | — | Input audio. Required for `wsola`; required for token methods unless `codes` is supplied |
| `codes` | EncodedTensor | No | — | Pre-extracted discrete tokens (skips extraction step for token methods) |
| `token_rate` | float | Cond. | — | Token rate in Hz. Required when `codes` is supplied |
| `text` | str | Cond. | — | Text content. Required for `indextts2` re-synthesis |
| `spk_audio` | EncodedAudio or str | Cond. | — | Speaker reference audio. Required for `indextts2` |
| `output_format` | str | No | `wav` | `"wav"` or `"numpy"` |

#### Output

- `wav`: Binary WAV with headers `X-Sample-Rate`, `X-Original-Duration`, `X-Stretched-Duration`, `X-Rate`
- `numpy`: JSON with `EncodedTensor` + duration metadata

#### Token-based Speed Change Workflow (indextts2 / qwen3_tts)

1. Extract tokens from audio via IndexTTS `/extract_tokens` (50Hz) or Qwen3-TTS `/encode_tokens` (12Hz)
2. Pass `codes`, `token_rate`, and other required fields to `/stretch` with the appropriate `method`
3. Alternatively, pass `audio` directly and let the service handle extraction internally

## Function API (no HTTP)

```python
from tools.speech_rate_conversion.api_func import stretch_audio, stretch_tokens

# WSOLA (signal-level)
result = stretch_audio("input.wav", rate=1.25)

# Token stretch (no audio I/O — tokens only)
result = stretch_tokens(codes_np, rate=1.25, token_rate=50.0)
stretched_codes = result.stretched_codes
```
