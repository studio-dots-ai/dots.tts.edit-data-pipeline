# IndexTTS Synthesis Service

## Overview

Zero-shot TTS with speaker voice cloning and emotion control. The service provides atomic synthesis operations and comprehensive features built on top of them.

**Output sample rate**: 22050 Hz
**Port**: 8002 (default)

**Environment**: `envs/pipe-indextts`
**Python**: 3.10
**Key dependencies**: torch, transformers 4.52 (compatibility requirement)

The service imports the adjacent official `upstream/` checkout pinned by the repository submodule.

## Launch

```bash
# run under root repo directory for production
CHECKPOINTS_DIR=/path/to/model-cache envs/pipe-indextts/.venv/bin/python tools/index-tts/api_service.py --port 8002 --accel --torch-compile --cuda-kernel
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8002 | Port to bind |
| `--cfg-path` | `$CHECKPOINTS_DIR/config.yaml` | Config file path (supports `CHECKPOINTS_DIR` env var) |
| `--model-dir` | `$CHECKPOINTS_DIR` | Model directory (supports `CHECKPOINTS_DIR` env var) |
| `--device` | auto | Device (`cuda:0`, `cpu`, or `auto`) |
| `--fp16` | False | FP16 inference |
| `--deepspeed` | False | Enable DeepSpeed optimization |
| `--accel` | False | Enable AccelInferenceEngine (FlashAttention + CUDA graphs). Requires `flash_attn`. |
| `--cuda-kernel` | False | Use custom CUDA kernel for BigVGAN |
| `--torch-compile` | False | Enable torch.compile optimization |

## API reference

### Architecture: Atomic → Comprehensive

```
Atomic Operations (model layer in indextts/infer_v2.py)
├── infer()                    → text + speaker + emotion → audio
├── extract_semantic_tokens()  → audio → 50Hz tokens
├── synthesize_from_tokens()   → tokens + text → audio
└── infer(prefix_tokens=...)   → prefix tokens + continuation text → continued audio

Comprehensive Features (service layer in synthesis_service.py)
├── /synthesize                → wraps infer() with I/O handling
├── /extract_tokens            → wraps extract_semantic_tokens()
├── /synthesize_from_tokens    → wraps synthesize_from_tokens()
├── Emotion-tagged text (independent) → parses tags, calls infer() per segment, concatenates
└── Emotion-tagged text (continuation) → parses tags, chains infer(prefix_tokens=...) with configured alignment
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| GET | `/info` | Model/service info |
| POST | `/synthesize` | Basic TTS with speaker + emotion control |
| POST | `/extract_tokens` | Extract 50Hz semantic tokens from audio |
| POST | `/synthesize_from_tokens` | Synthesize from pre-computed tokens |

All model endpoints share one process-wide inference lock. Requests may wait in
parallel at the HTTP layer, but model preparation, CUDA graph capture, token
extraction, synthesis, and response construction execute one request at a time.
`/health` and `/info` remain available while inference is running.

---

### Basic Synthesis

#### POST `/synthesize`

Atomic text-to-speech synthesis with voice cloning and emotion control. Wraps the model's `infer()` method with audio encoding/decoding.

##### Input (`SynthesisRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | str | Yes | Text to synthesize |
| `spk_audio_prompt` | EncodedAudio or str | Yes | Speaker reference audio (file path or base64 object) |
| `emo_audio_prompt` | EncodedAudio or str | No | Emotion reference audio |
| `emo_text` | str | No | Text for QwenEmotion LLM emotion detection |
| `emo_vector` | List[float] (8-dim) | No | Direct emotion vector |
| `emo_alpha` | float | No | Public emotion intensity control |
| `use_emotion_tags` | bool | No | Parse inline emotion tags (default: false) |
| `generation_paradigm` | str | No | `"independent"` (default) or `"continuation"`. `"independent_smoothed"` is deprecated and no longer supported by this service. |
| `aligner_url` | str | No | Aligner gateway URL (default: `http://localhost:8029`). Continuation mode uses the gateway plus text queries for boundary finding. |
| `aligner_backend` | str | No | Explicit aligner gateway backend for continuation boundary queries: `auto` (default), or `qwen3`. |
| `aligner_language` | str | No | Optional aligner gateway language code for continuation boundary queries, for example `eng` or `zho`. |
| `smoother_url` | str | No | Boundary Pause Smoother URL (default: `http://localhost:8018`). Only used when continuation boundary smoothing is enabled. |
| `continuation_debug` | bool | No | If `true`, save per-step tagged continuation debug artifacts including each untruncated full-step waveform and the exact texts/params used (default: `false`). |
| `continuation_debug_dir` | str | No | Directory for tagged continuation debug artifacts. Only used when `continuation_debug=true`. |
| `smooth_continuation_boundaries` | bool | No | If `true`, smooth only internal splice points in emotion-tagged continuation synthesis using the smoother regen-only fast path (default: `true`). |
| `smooth_radius_sec` | float | No | Manual continuation smoothing half-width in seconds. When omitted, the service does not send `regen_margin_sec`, so Boundary Pause Smoother uses its own default `regen_margin_sec=0.15`. |
| `top_p` | float | No | Top-p sampling (default: 0.8) |
| `top_k` | int | No | Top-k sampling (default: 30) |
| `temperature` | float | No | Sampling temperature (default: 0.8) |
| `repetition_penalty` | float | No | Repetition penalty (default: 10.0) |
| `max_mel_tokens` | int | No | Max mel tokens (default: 1500) |
| `max_text_tokens_per_segment` | int | No | Max text tokens per segment (default: 120) |
| `interval_silence` | int | No | Silence between segments in ms (default: 200) |
| `seed` | int | No | Random seed for reproducible inference |
| `output_format` | str | No | `wav` (binary) or `numpy` (JSON) |
| `return_cache` | bool | No | If `true`, response includes precomputed conditioning tensors (default: `false`). For WAV output the response becomes `multipart/mixed`. Not supported for emotion-tagged multi-segment synthesis. |
| `cache` | InferenceCache | No | Precomputed cache from a previous `return_cache=true` response. When provided, speaker/emotion conditioning is loaded from cache instead of being recomputed from audio. |
| `return_tokens` | bool | No | If `true`, return generated semantic tokens alongside audio (default: `false`). When `return_cache=true`, tokens are stored in the cache's `continuation` field. For WAV output with return\_cache, tokens are embedded in the cache JSON. For numpy output, a `tokens` field is added. |
| `skip_vocode` | bool | No | If `true`, skip S2Mel + BigVGAN synthesis and return only semantic tokens (default: `false`). Implies `return_tokens=true`. Useful for getting tokens without audio synthesis cost. |
| `return_segment_timestamps` | bool | No | If `true`, include `segment_timestamps` in JSON-compatible responses for emotion-tagged synthesis |

**Emotion vector order**: `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

**Emotion modes are mutually exclusive** — provide at most one of: `emo_audio_prompt`, `emo_text`, `emo_vector`.

**Alpha vs. Legacy Alias**

- `emo_alpha` is the public emotion intensity control for `emo_vector`, `emo_text`, and inline `alpha=` tags.
- In vector/text/tag modes, the service applies IndexTTS2's upstream emotion-specific calibration before passing the final vector to the model.
- For `emo_audio_prompt`, `emo_alpha` remains the active control.

##### Example

```python
import requests

response = requests.post("http://localhost:8002/synthesize", json={
    "text": "欢迎使用 IndexTTS 语音合成服务。",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "emo_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],  # happy + calm
    "emo_alpha": 0.8,
    "output_format": "wav",
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

##### Output

- `wav`: Binary WAV with headers `X-Sample-Rate`, `X-Duration-Seconds`
- `numpy`: JSON with `EncodedTensor` + metadata

---

### Token-Level Operations

#### POST `/extract_tokens`

Extract 50Hz discrete semantic tokens from audio. Wraps the model's `extract_semantic_tokens()` method.

##### Input (`TokenExtractRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | EncodedAudio or str | Yes | Input audio (file path or base64 object) |

##### Output (`TokenExtractResponse`)

| Field | Type | Description |
|-------|------|-------------|
| `codes` | EncodedTensor | 1-D int64 semantic codes |
| `token_rate` | float | Token rate (50.0 Hz) |
| `num_tokens` | int | Number of tokens extracted |

##### Example

```python
import requests

response = requests.post("http://localhost:8002/extract_tokens", json={
    "audio": "/path/to/input.wav",
})

print(f"Extracted {response.json()['num_tokens']} tokens")
```

---

#### POST `/synthesize_from_tokens`

Synthesize audio from pre-computed 50Hz semantic tokens. Wraps the model's `synthesize_from_tokens()` method. The GPT forward pass produces the latent representation, then s2mel + BigVGAN decodes to audio.

##### Input (`SynthesizeFromTokensRequest`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `codes` | EncodedTensor | Yes | 1-D int64 semantic codes |
| `text` | str | Yes | Corresponding text |
| `spk_audio_prompt` | EncodedAudio or str | Yes | Speaker reference audio |
| `emo_audio_prompt` | EncodedAudio or str | No | Emotion reference audio |
| `emo_text` | str | No | Emotion text |
| `emo_vector` | List[float] | No | Emotion vector |
| `emo_alpha` | float | No | Public emotion intensity control |
| `interval_silence` | int | No | Silence between segments in ms |
| `max_text_tokens_per_segment` | int | No | Max text tokens per segment |
| `seed` | int | No | Random seed for reproducible inference |
| `output_format` | str | No | `wav` or `numpy` |
| `return_cache` | bool | No | Same as `/synthesize` — return precomputed conditioning cache (default: `false`) |
| `cache` | InferenceCache | No | Same as `/synthesize` — reuse precomputed cache |

##### Output

Same as `/synthesize`.

---

### Advanced Synthesis Features

#### Emotion-Tagged Text Generation

**Built on**: Multiple calls to `infer()` or chained `infer(prefix_tokens=..., prefix_text=...)` (service-layer composition)

When `use_emotion_tags=True`, the service parses inline emotion tags and synthesizes each segment with its corresponding emotion. Two paradigms are available in the service:

##### Independent Paradigm (default)

`generation_paradigm="independent"` — Synthesizes each segment separately via `infer()`, concatenates with silence. Fast but may have audible boundaries.

**How it works**:
1. Parse tags like `<happy>text</happy><calm, alpha=0.55>text</calm>` into segments
2. Call `infer()` for each segment with its emotion vector
3. Concatenate audio parts with configurable silence intervals

**Inline control syntax**

- `<happy>text</happy>`: bare tag, falls back to request-level control
- `<happy, alpha=0.72>text</happy>`: preferred public intensity control
- `level=` tags are not part of the service contract. If a caller uses a higher-level
  tag schema such as `emotion + level`, it must convert those tags to explicit
  `alpha=` values before calling this API.
- Tagged input must cover the full text. Untagged spans are rejected; callers must wrap neutral spans with `<none>...</none>`.


##### Continuation Paradigm

`generation_paradigm="continuation"` — Each step synthesizes the full remaining text, then truncates at the current segment boundary and reuses the truncated tokens/text as the next continuation prefix. Requires the aligner gateway.

**How it works**:
1. Parse tags into segments
2. **Each non-final step**: generate the entire remaining text with the current segment's emotion. Step 0 uses standard `infer()` with no prefix; later steps use `infer()` with accumulated prefix tokens/text from previous steps.
3. Use the aligner gateway with query-based boundary lookup to find where the current segment ends in the generated audio. The caller may override the gateway backend with `aligner_backend` and optionally pass `aligner_language`.
4. Truncate at that boundary in token space and use the truncated prefix for the next segment.
5. The final segment's output contains all segments synthesized through a single BigVGAN pass, so optional smoothing still targets only internal splice points.

**Optional continuation splice smoothing**:

- Continuation-tagged synthesis smooths internal splice points by default. Set `smooth_continuation_boundaries=false` to keep the raw continuation waveform.
- The service sends explicit splice timestamps to Boundary Pause Smoother in `smooth_only` mode, so the smoother skips alignment/VAD pause resolution and regenerates only seam-centered waveform regions.
- When `smooth_radius_sec` is omitted, the service does not send `regen_margin_sec`, so Boundary Pause Smoother uses its own default `regen_margin_sec=0.15`.
- `smooth_radius_sec` controls the per-splice regeneration half-width and maps directly to `regen_margin_sec` when provided.
- This does not smooth the start or end of the utterance and does not remove pauses.

**Additional required parameters**:
- `aligner_url`: aligner gateway service URL (default: `http://localhost:8029`)

**Additional optional aligner parameters**:
- `aligner_backend`: explicit aligner gateway backend for continuation boundary queries: `auto` (default), or `qwen3`
- `aligner_language`: optional aligner gateway language code such as `eng` or `zho`

**Additional optional parameters for splice smoothing**:
- `smooth_continuation_boundaries`: enable/disable internal-boundary smoothing (default: `true`)
- `smooth_radius_sec`: manual regen radius in seconds; omit to use Boundary Pause Smoother's default `regen_margin_sec=0.15`
- `smoother_url`: smoother service URL (default: `http://localhost:8018`)

**Requirements**:
- aligner gateway service must be running (HTTP 503 if unavailable)
- if `smooth_continuation_boundaries=true`, Boundary Pause Smoother must also be running

##### Removed Service Mode

`generation_paradigm="independent_smoothed"` is deprecated and no longer supported by `POST /synthesize`.

If you still need boundary pause smoothing, run it in the pipeline after tagged synthesis instead of requesting it from the IndexTTS service.

##### Example (Independent)

```python
import requests

response = requests.post("http://localhost:8002/synthesize", json={
    "text": "<happy, alpha=0.78>你好，欢迎使用我们的服务！</happy><calm, alpha=0.42>如有任何问题，请随时联系我们。</calm>",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "use_emotion_tags": True,
    "return_segment_timestamps": True,
    "interval_silence": 200,  # ms between segments
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

##### Example (Continuation)

```python
import requests

response = requests.post("http://localhost:8002/synthesize", json={
    "text": "<happy, alpha=0.78>你好，欢迎使用我们的服务！</happy><calm, alpha=0.42>如有任何问题，请随时联系我们。</calm>",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "use_emotion_tags": True,
    "generation_paradigm": "continuation",
    "return_segment_timestamps": True,
    "aligner_url": "http://localhost:8029",
})

with open("output.wav", "wb") as f:
    f.write(response.content)
```

**Supported emotion tags**: `happy`, `angry`, `sad`, `afraid`, `disgusted`, `melancholic`, `surprised`, `calm`

---

### Inference Cache API

The service is fully **stateless** — no speaker or emotion conditioning tensors are stored between requests. Instead, conditioning can be **externalized** via the `return_cache` / `cache` request fields, giving callers explicit control over reuse.

#### How It Works

1. **Get cache**: Send a request with `return_cache: true`. The response includes an `InferenceCache` object containing precomputed speaker and emotion conditioning tensors.
2. **Reuse cache**: Send a subsequent request with the `cache` field set to the previously returned `InferenceCache`. The service skips conditioning recomputation and uses the cached tensors directly.

This avoids redundant GPU work when synthesizing multiple utterances with the same speaker/emotion — the speaker audio only needs to be processed once.

#### Response Format by Output Mode

| `output_format` | `return_cache` | Response format |
|---|---|---|
| `wav` | `false` | Binary WAV (`audio/wav`) |
| `wav` | `true` | `multipart/mixed` — part 1: WAV bytes, part 2: cache JSON |
| `numpy` | `false` | JSON with `tensor` field |
| `numpy` | `true` | JSON with `tensor` + `cache` fields |

#### Cache Schema

```
InferenceCache
├── speaker (SpeakerCache, required)
│   ├── spk_cond    (EncodedTensor) — w2v-bert speaker conditioning embedding
│   ├── s2mel_style (EncodedTensor) — CAM++ global style vector
│   ├── s2mel_prompt(EncodedTensor) — length-regulated prompt condition
│   └── mel         (EncodedTensor) — reference mel spectrogram
└── emotion (EmotionCache, optional)
    └── emo_cond    (EncodedTensor) — w2v-bert emotion conditioning embedding
```

Each `EncodedTensor` is a base64-encoded numpy array with `data`, `shape`, and `dtype` fields.

#### Limitations

- **Not supported for emotion-tagged multi-segment synthesis** (`use_emotion_tags=true`). The `cache` field in the response will be `null` even if `return_cache=true`.
- Cache is specific to a particular speaker audio + emotion audio combination. Using a cache computed from one speaker with a different speaker's text will produce incorrect results.

#### Example: Cache Round-Trip (numpy)

```python
import requests

BASE = "http://localhost:8002"

# Step 1: Synthesize with return_cache=True
resp1 = requests.post(f"{BASE}/synthesize", json={
    "text": "你好世界。",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "emo_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "output_format": "numpy",
    "return_cache": True,
}).json()

cache = resp1["cache"]  # InferenceCache object

# Step 2: Reuse cache for a different text (same speaker/emotion)
resp2 = requests.post(f"{BASE}/synthesize", json={
    "text": "再见世界。",
    "spk_audio_prompt": "/path/to/speaker.wav",  # still required, but ignored when cache is present
    "output_format": "numpy",
    "cache": cache,
}).json()
```

#### Example: Cache Round-Trip (WAV, multipart)

```python
import requests

BASE = "http://localhost:8002"

# Step 1: Get WAV + cache via multipart response
resp = requests.post(f"{BASE}/synthesize", json={
    "text": "你好世界。",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "output_format": "wav",
    "return_cache": True,
})

# Parse multipart response
boundary = resp.headers["Content-Type"].split("boundary=")[1]
parts = resp.content.split(f"--{boundary}".encode())
# parts[1] = WAV bytes (after headers), parts[2] = cache JSON (after headers)

wav_bytes = parts[1].split(b"\r\n\r\n", 1)[1]
cache_json = parts[2].split(b"\r\n\r\n", 1)[1]

import json
cache = json.loads(cache_json)

# Step 2: Reuse cache
resp2 = requests.post(f"{BASE}/synthesize", json={
    "text": "再见世界。",
    "spk_audio_prompt": "/path/to/speaker.wav",
    "output_format": "wav",
    "cache": cache,
})

with open("output.wav", "wb") as f:
    f.write(resp2.content)
```

---

### Audio Input Format

Audio inputs can be provided as:

1. **File path string**: `"/path/to/audio.wav"`
2. **Base64-encoded object**:
   ```json
   {
     "data": "base64-encoded-audio-bytes...",
     "format": "wav"
   }
   ```

---

### Emotion Vector Reference

| Emotion | Vector |
|---------|--------|
| `happy` | `[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| `angry` | `[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| `sad` | `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]` |
| `afraid` | `[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]` |
| `disgusted` | `[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]` |
| `melancholic` | `[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]` |
| `surprised` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]` |
| `calm` | `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]` |

**Tip**: For best results, keep the sum of emotion vector values ≤ 0.8.

---

### Example Client Code

```python
import requests
import base64

def encode_audio(path: str) -> dict:
    with open(path, "rb") as f:
        return {
            "data": base64.b64encode(f.read()).decode("utf-8"),
            "format": "wav",
        }

# Example 1: Basic synthesis with emotion vector
response = requests.post("http://localhost:8002/synthesize", json={
    "text": "欢迎使用 IndexTTS 语音合成服务。",
    "spk_audio_prompt": encode_audio("speaker.wav"),
    "emo_vector": [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "emo_alpha": 0.8,
})

with open("output.wav", "wb") as f:
    f.write(response.content)

# Example 2: Emotion-tagged text (service-layer composition)
response = requests.post("http://localhost:8002/synthesize", json={
    "text": "<happy>你好，欢迎使用我们的服务！</happy><calm>如有任何问题，请随时联系我们。</calm>",
    "spk_audio_prompt": "speaker.wav",
    "use_emotion_tags": True,
})

# Example 3: Token extraction and synthesis
tokens = requests.post("http://localhost:8002/extract_tokens", json={
    "audio": "input.wav",
}).json()

response = requests.post("http://localhost:8002/synthesize_from_tokens", json={
    "codes": tokens["codes"],
    "text": "Hello world.",
    "spk_audio_prompt": "speaker.wav",
})

```

## Source adapter

`upstream/` contains the pinned official IndexTTS checkout. `adapter.py` extends
its inference class with stateless conditioning, continuation, and token-based
synthesis. Unchanged model components are imported directly from upstream.
Continuation requests use the upstream HuggingFace generation path because the
acceleration engine does not consume prefix tokens.

The CUDA extension is built from official upstream sources using PyTorch's cache.
Build products use `TORCH_EXTENSIONS_DIR` when supplied, otherwise the shared
cache under `ARTIFACT_DIR`, with a tool and PyTorch version subdirectory. Model files use the
explicit model directory and HuggingFace cache configuration.
