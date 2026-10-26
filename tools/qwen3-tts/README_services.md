# Qwen3-TTS Service

## Overview

TTS service based on Qwen3-TTS models. Supports two synthesis modes:

- **Custom Voice** (`/synthesize_custom`): Preset speakers with natural language emotion/style instructions
- **Voice Clone** (`/synthesize_clone`): Zero-shot voice cloning from a reference audio prompt

Also supports 12Hz discrete token encoding/decoding for token-level manipulation.



**Environment**: `envs/pipe-gpu`
**Python**: 3.10
**Key dependencies**: torch 2.8, transformers >=4.57
**Models**:
- `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` — preset speakers + instruct
- `Qwen/Qwen3-TTS-12Hz-1.7B-Base` — voice cloning
- `Qwen/Qwen3-TTS-Tokenizer-12Hz` — 12Hz audio codec

## Launch

```bash
source envs/pipe-gpu/.venv/bin/activate
cd tools/qwen3-tts
python -m qwen3_tts_service.api_service --port 8007
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8007 | Port to bind |
| `--device` | cuda:0 | Device (cuda:0, cpu) |
| `--dtype` | bfloat16 | Model dtype (float16, bfloat16, float32) |
| `--no-custom-voice` | false | Skip loading CustomVoice model |
| `--no-base` | false | Skip loading Base (clone) model |
| `--no-tokenizer` | false | Skip loading audio tokenizer |

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/info` | Model info + available speakers |
| POST | `/synthesize_custom` | Preset speaker + instruct synthesis |
| POST | `/synthesize_clone` | Voice cloning synthesis |
| POST | `/encode_tokens` | Audio → 12Hz discrete tokens |
| POST | `/decode_tokens` | 12Hz discrete tokens → audio |

---

### POST `/synthesize_custom`

Synthesize with preset speaker and optional emotion/style instruction.

#### Input (`CustomVoiceRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | str | Yes | — | Text to synthesize |
| `speaker` | str | Yes | — | Preset speaker name (e.g. `Ryan`, `Vivian`). See `/info` for available speakers. |
| `instruct` | str | No | None | Style instruction (e.g. "A very happy voice.") |
| `language` | str | No | `Auto` | Language code hint |
| `seed` | int | No | None | Random seed for reproducible inference |
| `do_sample` | bool | No | None | Override sampling behavior (use model default if not set) |
| `temperature` | float | No | None | Sampling temperature |
| `top_k` | int | No | None | Top-k sampling |
| `top_p` | float | No | None | Top-p (nucleus) sampling |
| `subtalker_dosample` | bool | No | None | Sub-talker (code predictor) sampling flag |
| `subtalker_top_k` | int | No | None | Sub-talker top-k |
| `subtalker_top_p` | float | No | None | Sub-talker top-p |
| `subtalker_temperature` | float | No | None | Sub-talker temperature |
| `output_format` | str | No | `wav` | `wav` or `numpy` |

#### Available Speakers

The `/info` endpoint lists all available preset speakers. Common ones include: `Ryan`, `Vivian`, etc.

#### Output

- `wav`: Binary WAV response
- `numpy`: JSON with `EncodedTensor` + `sample_rate`, `duration_seconds`

---

### POST `/synthesize_clone`

Zero-shot voice cloning from reference audio.

#### Input (`VoiceCloneRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | str | Yes | Text to synthesize |
| `language` | str | No | Language code or `"Auto"` (default: `"Auto"`) |
| `ref_audio` | EncodedAudio or str | Yes | Reference audio for voice cloning |
| `ref_text` | str | Yes* | Transcript of the reference audio — must be an **accurate** transcript of the spoken content in `ref_audio`. Inaccurate text causes degraded output due to position-aligned conditioning. (*required unless `x_vector_only_mode=True`) |
| `seed` | int | No | Random seed for reproducible inference |
| `x_vector_only_mode` | bool | No | Use only speaker embedding, skip reference text/code (default: `false`) |
| `output_format` | str | No | `wav` or `numpy` (default: `wav`) |
| `do_sample` | bool | No | Override sampling behavior (use model default if not set) |
| `temperature` | float | No | Sampling temperature |
| `top_k` | int | No | Top-k sampling |
| `top_p` | float | No | Top-p (nucleus) sampling |
| `subtalker_dosample` | bool | No | Sub-talker (code predictor) sampling switch (False=greedy for determinism) |
| `subtalker_top_k` | int | No | Sub-talker top-k |
| `subtalker_top_p` | float | No | Sub-talker top-p |
| `subtalker_temperature` | float | No | Sub-talker temperature |

#### Output

Same as `/synthesize_custom`.

---

### POST `/encode_tokens`

Encode audio into 12Hz discrete tokens.

#### Input (`TokenEncodeRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | EncodedAudio or str | Yes | Input audio |
| `output_format` | str | No | Defaults to `numpy` |

#### Output (`TokenEncodeResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `codes` | EncodedTensor | int64 token tensor of shape (T, Q) |
| `token_rate_hz` | float | Tokenizer input sample rate divided by encode downsample rate |
| `duration_seconds` | float | Duration represented by tokens (`T / token_rate_hz`) |
| `status` | str | `success` |
| `message` | str or null | Encoded shape information |

---

### POST `/decode_tokens`

Decode 12Hz discrete tokens back to audio.

#### Input (`TokenDecodeRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `codes` | EncodedTensor | Yes | int64 token tensor of shape (T, Q) |
| `output_format` | str | No | `wav` (default) or `numpy` |

#### Output

- `wav`: Binary WAV response
- `numpy`: JSON with `EncodedTensor` + `sample_rate`

---

## Design Notes

The service is designed for future replacement of the HuggingFace model backend with vLLM for increased throughput. The core layer (`core.py`) isolates model loading and inference, making backend swaps straightforward.

## Official source layout

The service wrapper lives in this directory; `upstream/` pins the official source. Initialize that submodule before launching. Reuse an existing uv interpreter by absolute path in a new worktree. The service activates and verifies this checkout before importing model code, so a previous editable installation cannot silently supply the implementation. Pinned source revisions and patches are recorded in `upstream.json`; regression contracts live in `tests/`.
