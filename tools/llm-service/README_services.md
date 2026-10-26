# Unified LLM Generation Service

## Overview

A unified FastAPI gateway that routes text generation requests to a **Gemini API backend** (remote), a **local vLLM backend** (a private asynchronous vLLM server for Qwen3.5, Qwen3-Omni, Gemma 4, and compatible instruct models), or an **OpenAI-compatible backend** (external vLLM server such as Gemma 4 MTP).


- **Port**: 8012

**Environment**: `envs/pipe-llm`
**Python**: 3.10
**Key dependencies**: torch 2.10, vLLM 0.19.0, transformers >=5.5.0

Configure model paths and API credentials as described in [environment setup](../../docs/environment-setup.md).

## Launch

```bash
source envs/pipe-llm/.venv/bin/activate

# With local vLLM backend only
# Qwen3.5-27B: use --max-cudagraph-capture-size 256 (hybrid Mamba arch can't handle default 512)
python tools/llm-service/api_service.py \
    --local-model Qwen/Qwen3.5-27B --max-cudagraph-capture-size 256 --port 8012

# Gemma 4 31B
python tools/llm-service/api_service.py \
    --local-model google/gemma-4-31B \
    --gpu-memory-utilization 0.95 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --enforce-eager \
    --port 8012

# Gemma 4 26B A4B IT
python tools/llm-service/api_service.py \
    --local-model google/gemma-4-26B-A4B-it --port 8012

# Qwen3-Omni audio understanding (Thinker only, two GPUs)
python tools/llm-service/api_service.py \
    --local-model qwen3-omni-30b-a3b-instruct \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --enforce-eager \
    --port 8012

# With Gemini 3 Flash (default, no GPU needed)
python tools/llm-service/api_service.py --port 8012

# With Gemini 2.5 Pro
python tools/llm-service/api_service.py --gemini-model 2_5pro --port 8012

# With both Gemini and local backends
python tools/llm-service/api_service.py \
    --gemini-model 2_5pro --local-model Qwen/Qwen3.5-27B --max-cudagraph-capture-size 256 --port 8012

# One-command Gemma 4 MTP: start/reuse external vLLM on GPUs 0,1 and register gemma4_mtp
python tools/llm-service/api_service.py \
    --gemma4-mtp-gpus 0,1 \
    --port 8012

# Auto-select all currently visible GPUs for Gemma 4 MTP
python tools/llm-service/api_service.py \
    --gemma4-mtp-gpus \
    --port 8012

# Or launch Gemma 4 MTP separately, then proxy it from llm-service
tools/llm-service/scripts/launch_gemma4_mtp_vllm.sh \
  --model-path /path/to/models/gemma4 --draft-model-path /path/to/models/gemma4-mtp \
  --log-dir "$ARTIFACT_DIR/gemma4-logs" --gpus 0,1
python tools/llm-service/api_service.py \
    --openai-base-url http://localhost:18112/v1 \
    --openai-backend-name gemma4_mtp \
    --openai-model gemma-4-31b-it-mtp-tp2-32k \
    --openai-batch-workers 8 \
    --port 8012

# Use 9B model instead (no special flags needed)
python tools/llm-service/api_service.py --local-model Qwen/Qwen3.5-9B --port 8012

# Disable CUDA graphs explicitly when needed
python tools/llm-service/api_service.py \
    --local-model Qwen/Qwen3.5-27B --enforce-eager --port 8012
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host to bind |
| `--port` | `8012` | Port to bind |
| `--local-model` | None | HuggingFace model ID, local path, or supported alias for vLLM (e.g. `Qwen/Qwen3.5-27B`, `qwen3-omni-30b-a3b-instruct`, `gemma-4-31b`) |
| `--gpu-memory-utilization` | `0.90` | GPU memory fraction for vLLM |
| `--max-model-len` | `16384` | Maximum context length |
| `--max-num-seqs` | None | Maximum concurrent sequences for vLLM scheduler and warmup; lower this for large models that OOM on startup |
| `--tensor-parallel-size` | `1` | Tensor parallelism |
| `--max-cudagraph-capture-size` | None | Max batch size for CUDA graph capture (use `256` for 27B) |
| `--enforce-eager` | `false` | Disable CUDA graphs entirely |
| `--enable-thinking` | `false` | Enable tokenizer thinking mode when supported (currently Qwen-only) |
| `--openai-base-url` | None | Base URL for an external OpenAI-compatible server, e.g. `http://localhost:18112/v1` |
| `--openai-backend-name` | `gemma4_mtp` | Backend name exposed through `/generate` and `/generate_batch` |
| `--openai-model` | `gemma-4-31b-it-mtp-tp2-32k` | Default served model name for the OpenAI-compatible backend |
| `--openai-timeout` | `300.0` | HTTP timeout in seconds for external backend requests |
| `--openai-batch-workers` | `8` | Concurrent request workers used by `/generate_batch` for the external backend |
| `--gemma4-mtp-gpus` | None | Optional comma-separated GPU ids; when the flag is present without a value, or set to `auto`, starts/reuses Gemma 4 MTP on all visible GPUs |
| `--gemma4-mtp-launch-script` | `tools/llm-service/scripts/launch_gemma4_mtp_vllm.sh` | Launcher script path |
| `--gemma4-mtp-port` | `18112` | External Gemma 4 MTP vLLM port |
| `--gemma4-mtp-launch-timeout` | `900` | Seconds to wait for the external vLLM server readiness check |
| `--gemma4-mtp-log-dir` | None | Optional launcher log directory; defaults to `tools/llm-service/logs` inside the script |
| `--gemma4-mtp-max-model-len` | Auto | Override external Gemma 4 MTP max context length; launcher defaults to `8192` for 1 GPU and `32768` for 2+ GPUs |
| `--gemma4-mtp-max-num-batched-tokens` | Auto | Override external Gemma 4 MTP scheduler token budget; launcher defaults to `8192` for 1 GPU and `32768` for 2+ GPUs |
| `--gemma4-mtp-max-num-seqs` | Auto | Override external Gemma 4 MTP max concurrent sequences; launcher defaults to `4` for 1 GPU and `8` for 2+ GPUs |
| `--gemma4-mtp-gpu-memory-utilization` | `0.92` | Override external Gemma 4 MTP GPU memory utilization |
| `--gemini-model` | `3flash` | Gemini model: `3flash` (Gemini 3 Flash) or `2_5pro` (Gemini 2.5 Pro) |

## Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check with backend status |
| GET | `/info` | List available backends and models |
| POST | `/generate` | Single prompt text generation |
| POST | `/generate_batch` | Batch text generation |

### POST `/generate`

Generate text from a single prompt.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `"local"` | `"gemini"` or `"local"` |
| `model` | str? | null | Model name override |
| `prompt` | str? | null | Pre-rendered prompt text |
| `template` | str? | null | Template with `{placeholders}` |
| `template_vars` | dict? | null | Variables to fill template |
| `temperature` | float | 0.7 | Sampling temperature [0, 2] |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `json_mode` | bool | false | Constrain output to valid JSON |
| `stop_sequences` | list[str]? | null | Stop strings |
| `audio_base64` | str? | null | Base64-encoded audio data for Gemini or an audio-capable local model |
| `audio_mime_type` | str? | null | MIME type of audio (e.g. `audio/wav`, `audio/mpeg`). Defaults to `audio/wav` |

Either `prompt` or `template` + `template_vars` must be provided.

When `audio_base64` is provided, the audio is prepended to the prompt as a multimodal part. Gemini uses `inline_data`; Qwen3-Omni uses an OpenAI-compatible `audio_url` data URI. Text-only local models reject audio explicitly.

**Response:**

```json
{
  "text": "generated text...",
  "model": "qwen3.5-27b",
  "backend": "local",
  "usage": {"prompt_tokens": 123, "completion_tokens": 456}
}
```

### POST `/generate_batch`

Generate text for multiple prompts.

**Request body:**

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | str | `"local"` | `"gemini"` or `"local"` |
| `model` | str? | null | Model name override |
| `prompts` | list[str]? | null | Pre-rendered prompts |
| `template` | str? | null | Shared template |
| `template_vars_list` | list[dict]? | null | Per-prompt template variables |
| `temperature` | float | 0.7 | Sampling temperature |
| `max_tokens` | int | 4096 | Maximum output tokens |
| `json_mode` | bool | false | Constrain output to valid JSON |
| `stop_sequences` | list[str]? | null | Stop strings |
| `audio_base64_list` | list[str?]? | null | Per-prompt base64 audio data; null entries are text-only |
| `audio_mime_type` | str? | null | Shared MIME type for batch audio inputs |

**Response:**

```json
{
  "results": [
    {"text": "...", "model": "qwen3.5-27b", "backend": "local", "usage": {}},
    ...
  ],
  "count": 3
}
```

## Backends

### Gemini Backend

- Set `GEMINI_API_URL` to a Gemini-compatible generate-content endpoint.
- Supply `GEMINI_API_KEY`, or model-specific `GEMINI_3FLASH_API_KEY` /
  `GEMINI_2_5PRO_API_KEY` overrides. Keys are read at startup.
- Model selection is controlled by `--gemini-model`:
  - `3flash` → Gemini 3 Flash (default, faster)
  - `2_5pro` → Gemini 2.5 Pro (more capable)
- Automatic long retry for transient gateway failures; token-budget exhaustion
  fails immediately because retrying the same request will not help
- Thread-safe connection pooling (32 connections)
- Reports not ready until both endpoint and credentials are configured.

### Local vLLM Backend

- Launches a private loopback-only vLLM OpenAI server and forwards every input as an independent request
- Supports Qwen3.5-27B, Qwen3.5-9B, Qwen3-Omni-30B-A3B-Instruct, `google/gemma-4-26B-A4B-it`, and `google/gemma-4-31B`
- The `qwen3-omni-30b-a3b-instruct` alias resolves to the repository checkpoint and accepts audio through vLLM chat completions; text-only local models reject audio explicitly
- Also accepts the aliases `qwen3.5-27b`, `qwen3.5-9b`, `gemma-4-26b-a4b-it`, and `gemma-4-31b` for `--local-model`
- The private vLLM server owns chat-template rendering and JSON constrained decoding
- Independent `/generate` requests remain concurrent across the gateway and are continuously batched by vLLM; `/generate_batch` submits independent requests concurrently while preserving response order
- The private server binds to a dynamically allocated loopback port and is terminated with the gateway, including unexpected parent exit
- `google/gemma-4-31B` is a much tighter fit on a single GPU than the instruct checkpoints; use reduced `--max-model-len`, lowered `--max-num-seqs`, and `--enforce-eager` when validating it locally

### OpenAI-Compatible Backend

- Proxies to an external `/v1/chat/completions` server, typically vLLM's OpenAI API server
- Designed for custom runtimes such as Gemma 4 MTP that should not be installed into `envs/pipe-llm`
- Default backend name: `gemma4_mtp`
- Default model: `gemma-4-31b-it-mtp-tp2-32k`
- `json_mode=True` maps to `response_format={"type":"json_object"}`
- `temperature`, `max_tokens`, `stop_sequences`, and per-request `model` overrides are passed through
- `/generate_batch` sends concurrent chat-completions requests while preserving response order; vLLM handles continuous batching behind the HTTP boundary

#### Gemma 4 MTP External vLLM Server

The launcher adds the configured interpreter's directory to `PATH`, so build tools such as `ninja` come from the same environment even when it is not activated in the shell.
On shutdown, the gateway stops the process group it launched, including workers whose parent has exited. Reused servers remain running.

Use the bundled launcher instead of pasting the full vLLM command. It uses the configured vLLM interpreter and the local Gemma 4 / MTP checkpoints, writes logs under the explicit `--log-dir` inside `ARTIFACT_DIR`, waits for `/v1/models`, and prints `BASE_URL=...` when ready. Reuse is intentionally conservative: an existing server on the same port is reused only when the launcher's metadata matches the requested GPUs and runtime shape.

```bash
# Run with two GPUs.
tools/llm-service/scripts/launch_gemma4_mtp_vllm.sh \
  --model-path /path/to/models/gemma4 --draft-model-path /path/to/models/gemma4-mtp \
  --log-dir "$ARTIFACT_DIR/gemma4-logs" --gpus 0,1

# Use a single visible device.
tools/llm-service/scripts/launch_gemma4_mtp_vllm.sh \
  --model-path /path/to/models/gemma4 --draft-model-path /path/to/models/gemma4-mtp \
  --log-dir "$ARTIFACT_DIR/gemma4-logs" --gpus 0

# Explicitly select visible devices.
tools/llm-service/scripts/launch_gemma4_mtp_vllm.sh \
  --model-path /path/to/models/gemma4 --draft-model-path /path/to/models/gemma4-mtp \
  --log-dir "$ARTIFACT_DIR/gemma4-logs" --gpus 0,1,2,3,4,5,6,7

# Full one-command llm-service startup.
python tools/llm-service/api_service.py \
  --gemma4-mtp-model-path /path/to/models/gemma4 \
  --gemma4-mtp-draft-model-path /path/to/models/gemma4-mtp \
  --gemma4-mtp-log-dir "$ARTIFACT_DIR/gemma4-logs" --gemma4-mtp-gpus 0,1 --port 8012
```

For 4 visible GPUs the launcher uses `--tensor-parallel-size 4` and no data parallelism. For 8 visible GPUs it uses tensor and data parallelism: `--tensor-parallel-size 4 --data-parallel-size 2 --data-parallel-size-local 2`.

Do not bind this external vLLM server to port `8001`; this project reserves `8001` for F5-TTS synthesis. Use `18112` unless there is a concrete reason to choose another free port.

## Prompt Template System

The service supports server-side prompt rendering via templates:

```python
# Client-side rendering (send pre-built prompt)
client.generate(prompt="Translate 'Hello' to French.")

# Server-side rendering (send template + vars)
client.generate_with_template(
    template="Translate '{text}' to {language}.",
    template_vars={"text": "Hello", "language": "French"},
)
```

## Audio Input

The Gemini backend and Qwen3-Omni local backend support multimodal audio+text generation. Pass `audio_base64` (and optionally `audio_mime_type`) alongside the text prompt to enable audio understanding tasks like emotion classification, stress detection, transcription, or captioning. Other local models return an error instead of ignoring audio.

```python
import base64
from speech_edit.clients.services import LLMClient

client = LLMClient("http://localhost:8012")

# Using audio_path (auto-encodes to base64)
result = client.generate(
    prompt="Identify the most stressed word in this utterance.",
    backend="gemini",
    audio_path="/path/to/audio.wav",
    temperature=0.0,
)

# Using pre-encoded base64
with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = client.generate(
    prompt="Transcribe this audio.",
    backend="gemini",
    audio_base64=audio_b64,
    audio_mime_type="audio/wav",
)
```

### curl example with audio

```bash
# Encode audio to base64
AUDIO_B64=$(base64 -w0 /path/to/audio.wav)

curl -X POST http://localhost:8012/generate \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "gemini",
    "prompt": "What is the speaker saying?",
    "audio_base64": "'$AUDIO_B64'",
    "audio_mime_type": "audio/wav",
    "temperature": 0.0
  }'
```

## Local deployment paths

Set `params.model_path` and `params.draft_model_path` in
`configs/service/llm_service/gemma4_mtp_tp2.yaml`. The profile passes these paths,
its `python_bin`, and its artifact log directory to the Gemma MTP launcher.
Service parameters go from YAML through CLI arguments to FastAPI application
state; credentials remain external environment inputs.

For standalone use, pass `--model-path`, `--draft-model-path`, `--vllm-python`,
and `--log-dir` to the shell launcher. The gateway accepts the corresponding
`--gemma4-mtp-*` options. `PYTHONPATH` may select a source overlay when required
by the chosen vLLM build.
