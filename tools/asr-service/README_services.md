# ASR Gateway

Repo-native ASR HTTP gateway with async request buffering. It delegates to
standalone backend services and normalizes them to `/transcribe` and
`/transcribe_batch`. Supports **Qwen3-ASR** through HTTP or an in-process Transformers backend.


## Environment

- **uv-managed environment**: `envs/pipe-cpu` for this gateway. Backend services
  use their own environments.
- **Python**: 3.10
- **Port**: 8010

## Start

```bash
# Start backend services first, then start the gateway.
envs/pipe-cpu/.venv/bin/python tools/asr-service/asr_service.py \
  --port 8010 \
  --default-backend qwen3-asr \
  --backend-url qwen3-asr=http://127.0.0.1:8033
```

## Endpoints

### `POST /transcribe`

Transcribe a single audio.

**Request body:**
```json
{
  "audio": {
    "data": "<base64-encoded audio bytes>",
    "format": "wav"
  },
  "language": null,
  "backend": "qwen3-asr"
}
```

- `audio`: Base64-encoded audio dict (`data` + `format`).
- `language`: Optional. `null`, `"Chinese"`, `"English"`, etc.
- `backend`: Optional. Defaults to the gateway `--default-backend`.

**Response:**
```json
{
  "text": "transcribed text",
  "language": "English"
}
```

### `POST /transcribe_batch`

Transcribe multiple audios in one request.

**Request body:**
```json
{
  "items": [
    {"audio": {"data": "<base64>", "format": "wav"}, "language": null},
    {"audio": {"data": "<base64>", "format": "wav"}, "language": null, "backend": "qwen3-asr"}
  ],
  "backend": "qwen3-asr"
}
```

**Response:**
```json
{
  "results": [
    {"text": "...", "language": "auto"},
    {"text": "...", "language": "auto"}
  ]
}
```

### `GET /health`

Returns service status and model info.

### `GET /info`

Returns backend info (model name, buffer config).

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | 8010 | Service port |
| `--host` | 0.0.0.0 | Bind host |
| `--backend-url` | none | Backend URL in `backend=url` form; may be repeated |
| `--default-backend` | qwen3-asr | Backend used when a request omits `backend` |
| `--backend-timeout` | 300 | Backend request timeout in seconds |
| `--backend-startup-timeout` | 1200 | Backend health wait timeout during gateway startup |
| `--buffer-batch-size` | 16 | Request buffer batch size |
| `--buffer-wait-ms` | 500 | Buffer wait timeout (ms) |

## Request Buffering

The gateway keeps one async queue per backend. A background consumer processes
batches when either `buffer-batch-size` requests accumulate or
`buffer-wait-ms` timeout is reached. Requests for different backends are never
mixed into the same backend batch.

### Stability-first direct Qwen3-ASR mode

Evaluation jobs that prioritize startup stability over throughput can load
Qwen3-ASR directly in the gateway process with Transformers:

```bash
envs/pipe-asr/.venv/bin/python tools/asr-service/asr_service.py \
  --port 8010 \
  --default-backend qwen3-asr \
  --qwen-transformers-model Qwen/Qwen3-ASR-1.7B \
  --qwen-transformers-device cuda:0 \
  --qwen-transformers-dtype bfloat16 \
  --buffer-batch-size 1 \
  --buffer-wait-ms 0
```

This mode does not start vLLM, TorchInductor compilation, or a separate Qwen
HTTP backend. It keeps one in-flight request and runs inputs sequentially. The
HTTP `/transcribe` and `/transcribe_batch` contracts are unchanged.
