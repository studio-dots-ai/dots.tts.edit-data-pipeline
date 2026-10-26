# UTMOS Speech Quality Evaluation Service

Lightweight MOS (Mean Opinion Score) predictor using UTMOS22 model via `torch.hub`.

## Overview

Predicts speech quality on a 1–5 MOS scale. Uses the `utmos22_strong` model from [SpeechMOS](https://github.com/tarepan/SpeechMOS).

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET/POST | Health check |
| `/predict` | POST | Single audio → MOS score |
| `/predict_batch` | POST | Multiple audios → MOS scores |

### POST `/predict`

**Request:**
```json
{"audio_data": "<base64-encoded-audio>"}
```

**Response:**
```json
{"score": 4.12, "sample_rate": 24000, "duration_seconds": 3.5, "inference_time_seconds": 0.05}
```

### POST `/predict_batch`

**Request:**
```json
{"audio_list": ["<base64-audio-1>", "<base64-audio-2>"]}
```

**Response:**
```json
{"scores": [4.12, 3.85], "count": 2, "avg_score": 3.99, "inference_time_seconds": 0.10}
```

## Launch

```bash
source envs/pipe-gpu/.venv/bin/activate
python tools/utmos/api_service.py --port 8017 [--device cuda]
```

Port: **8017**. Environment: **pipe-gpu**.

The service serializes `torch.hub` model loading with a cache-local file lock
and calls `torch.hub.load(..., skip_validation=True)` to avoid GitHub API
validation rate limits. For repeated production use, set `TORCH_HOME` to a
stable cache directory. The pitch/rate production launcher normally sets
the generation-only `defaults` service profile, so UTMOS is not started during data generation.

## Failure behavior

Batch evaluation fails the request if any audio cannot be decoded or scored. Failed items are never represented by numeric sentinel scores.
