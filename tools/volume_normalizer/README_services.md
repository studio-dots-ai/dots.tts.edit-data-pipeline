# Volume Normalizer Service

## Overview

Audio loudness normalization service. Normalizes audio volume to a target dB level using RMS or peak methods. Supports single and batch normalization.

**Port**: 8006 (default)


**Environment**: `envs/pipe-cpu`
**Python**: 3.10
**Key dependencies**: numpy, pydub

## Launch

```bash
source envs/pipe-cpu/.venv/bin/activate
python -m tools.volume_normalizer.api_service --port 8006
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/normalize` | Single audio normalization |
| POST | `/normalize_batch` | Batch normalization to same target level |

---

### POST `/normalize`

Normalize a single audio to target loudness.

#### Input (`NormalizeRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | EncodedAudio or str | Yes | — | Input audio |
| `target_db` | float | No | -20.0 | Target RMS level in dB |
| `method` | str | No | `rms` | Normalization method: `rms` or `peak` |
| `headroom_db` | float | No | -1.0 | Headroom to prevent clipping |
| `output_format` | str | No | `wav` | `wav` or `numpy` |

#### Output

- `wav`: Binary WAV response
- `numpy`: JSON with `EncodedTensor` + `original_rms_db`, `normalized_rms_db`, `gain_db`

---

### POST `/normalize_batch`

Normalize multiple audio files to the same loudness level.

#### Input (`NormalizeBatchRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audios` | List[EncodedAudio or str] | Yes | — | List of input audios |
| `target_db` | float | No | -20.0 | Target RMS level in dB |
| `method` | str | No | `rms` | `rms` or `peak` |
| `headroom_db` | float | No | -1.0 | Headroom |

#### Output

JSON list of `NormalizeResponse` objects (always numpy format in batch mode), each with `tensor`, `sample_rate`, `original_rms_db`, `normalized_rms_db`, `gain_db`.
