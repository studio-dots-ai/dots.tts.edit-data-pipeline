# VAD (Voice Activity Detection) Service

## Overview

Voice activity detection and silence trimming service using pydub's silence detection. Identifies speech/silence boundaries in audio and optionally trims silence.

**Port**: 8005 (default)


**Environment**: `envs/pipe-gpu`
**Python**: 3.11
**Key dependencies**: torch 2.8, pydub

## Launch

```bash
envs/pipe-gpu/.venv/bin/python tools/vad-service/api_service.py --port 8005
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/detect` | Detect speech segments |
| POST | `/trim` | Trim silence from audio boundaries |

---

### POST `/detect`

Detect speech and silence segments in audio.

#### Input (`DetectRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | EncodedAudio or str | Yes | — | Input audio |
| `min_silence_len` | int | No | 300 | Minimum silence length in ms |
| `silence_thresh` | int | No | -40 | Silence threshold in dBFS |
| `seek_step` | int | No | 10 | Seek step in ms |

#### Output (`DetectResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `status` | str | Request status ("success") |
| `segments` | List[dict] | List of detected speech segments (`{start, end, duration}` in seconds) |
| `total_duration` | float | Total audio duration in seconds |
| `speech_duration` | float | Total speech duration in seconds |
| `silence_duration` | float | Total silence duration in seconds |
| `num_segments` | int | Number of detected speech segments |

---

### POST `/trim`

Trim silence from audio start and end.

#### Input (`TrimRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `audio` | EncodedAudio or str | Yes | e | Input audio |
| `target_start_silence` | float | No | 0.05 | Desired silence at start (seconds) |
| `target_end_silence` | float | No | 0.05 | Desired silence at end (seconds) |
| `min_silence_len` | int | No | 100 | Minimum silence length in ms |
| `silence_thresh` | int | No | -40 | Silence threshold in dBFS (audio below is silence) |
| `output_format` | str | No | `wav` | `wav` or `numpy` |

#### Output

- `wav`: Binary WAV response
- `numpy`: JSON with `EncodedTensor` + timing metadata
