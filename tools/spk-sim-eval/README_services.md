# Speaker Similarity Evaluation Service

Speaker verification service using WavLM-large fine-tuned ECAPA-TDNN model.
Computes cosine similarity between speaker embeddings.

## Model

- Architecture: ECAPA_TDNN_SMALL (feat_dim=1024, feat_type=wavlm_large)
- Checkpoint: `$CHECKPOINTS_DIR/wavlm_large_finetune.pth` (set `CHECKPOINTS_DIR` to an external model directory); override with `--checkpoint`
- Output: 256-dim speaker embeddings
- Similarity range: -1.0 to 1.0 (cosine similarity)

## Quick Start

```bash
# Start (uses pipe-gpu venv)
envs/pipe-gpu/.venv/bin/python tools/spk-sim-eval/api_service.py --port 8019 --device cuda
```

| Property | Value |
|---|---|
| **Port** | 8019 |
| **Environment** | `pipe-gpu` |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET/POST | `/health` | Health check |
| POST | `/similarity` | Cosine similarity between two audios |
| POST | `/similarity_batch` | One reference vs batch of audios |
| POST | `/embedding` | Extract speaker embedding |

---

### GET/POST `/health`

```json
{"status": "healthy", "model_loaded": true}
```

---

### POST `/similarity`

Compute cosine similarity between two audio samples.

**Request (`SimilarityRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_a` | str | Yes | Base64-encoded WAV audio (reference/prompt) |
| `audio_b` | str | Yes | Base64-encoded WAV audio (synthesized) |

**Response (`SimilarityResponse`):**

| Field | Type | Description |
|-------|------|-------------|
| `similarity` | float | Cosine similarity (-1.0 to 1.0) |
| `inference_time_seconds` | float | Processing time |

---

### POST `/similarity_batch`

Compare one reference audio against a batch of audio samples.

**Request (`BatchSimilarityRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_ref` | str | Yes | Base64-encoded reference audio (shared across batch) |
| `audio_list` | list[str] | Yes | List of base64-encoded audio files to compare |

**Response (`BatchSimilarityResponse`):**

| Field | Type | Description |
|-------|------|-------------|
| `similarities` | list[float] | Per-audio cosine similarities. Failed audios are marked with `-2.0` and excluded from average. |
| `count` | int | Number of audio files processed |
| `avg_similarity` | float | Average similarity (excluding failures) |
| `inference_time_seconds` | float | Processing time |

---

### POST `/embedding`

Extract a 256-dim speaker embedding from audio.

**Request (`EmbeddingRequest`):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_data` | str | Yes | Base64-encoded WAV audio |

**Response (`EmbeddingResponse`):**

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | list[float] | 256-dim speaker embedding |
| `embedding_dim` | int | Embedding dimensionality (256) |
| `inference_time_seconds` | float | Processing time |

---

## Notes

- All audio inputs are base64-encoded WAV files.
- Audio is automatically resampled to 16 kHz mono before inference.
- Inputs longer than 60 seconds are deterministically truncated to the first
  59.9 seconds, leaving a 100 ms safety margin. The service emits a warning
  containing the original duration.

## Failure behavior

Batch evaluation fails the request if any audio cannot be decoded or embedded. Failed items are never represented by numeric sentinel similarities.
