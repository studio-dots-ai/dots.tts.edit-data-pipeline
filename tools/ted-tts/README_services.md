# TED-TTS Service

FastAPI wrapper for TED-TTS, a training-free intra-utterance emotion and duration control layer built on IndexTTS2.

## Environment

- Directory: `tools/ted-tts/`
- Env: `envs/pipe-indextts/.venv/bin/python`
- Default port: `8035`
- Default checkpoints: `$CHECKPOINTS_DIR/index-tts`; configure the external model directory or pass `--model-dir` and `--cfg-path`

TED-TTS introduces no new model weights. It reuses the IndexTTS2 checkpoint layout.

## Start

Set `TED_PYTHON_BIN` to an existing pipe-indextts interpreter, using its absolute
path when running from a worktree. Do not create or synchronize a new environment.

```bash
ARTIFACT_DIR=/path/to/artifacts CHECKPOINTS_DIR=/path/to/model-cache CUDA_VISIBLE_DEVICES=0 \
  "$TED_PYTHON_BIN" tools/ted-tts/api_service.py --port 8035
```

## Endpoints

### `GET /health`

Returns service readiness and model config.

### `GET /info`

Returns sample rate, supported modes, emotion vector order, and duration-token ratio.

### `POST /synthesize`

Synthesizes speech. `spk_audio_prompt` and `emo_audio_prompt` accept either a path string or an encoded object:

```json
{"data": "<base64 audio bytes>", "format": "wav"}
```

Request fields:

| Field | Type | Notes |
|---|---|---|
| `mode` | `0..5` | TED-TTS mode |
| `text` | string | Use `|` for segment boundaries in modes 3/4/5 |
| `spk_audio_prompt` | string/object | Required speaker prompt |
| `emo_audio_prompt` | string/object | Required for mode 1 |
| `emo_vector` | list[float] | Required for mode 2, 8 dims: happy, angry, sad, fear, hate, low, surprise, neutral |
| `emo_text` | string | Required for modes 3/4/5, `|` separated |
| `target_seconds` | list[float] | Duration-control modes 4/5 are temporarily disabled |
| `duration_mode` | string | Duration-control modes 4/5 are temporarily disabled |
| `eos_control_policy` | string | Duration-control modes 4/5 are temporarily disabled |
| `output_format` | string | `wav` or `json` |

Example mode 3 request:

```bash
curl -X POST http://localhost:8035/synthesize \
  -H 'Content-Type: application/json' \
  -o ted_mode3.wav \
  -d '{
    "mode": 3,
    "text": "Hello, nice to meet you!|We can get along peacefully.",
    "spk_audio_prompt": "tools/ted-tts/upstream/datasets/Ref/0011_000001.wav",
    "emo_text": "happy: cheerful and bright|neutral: calm and steady",
    "seed": 42,
    "output_format": "wav"
  }'
```

### `POST /adapt_tagged`

Converts canonical `<emo>` tagged text to TED-TTS segment inputs without
loading or running the model. This endpoint is useful for validating the tag
adapter used by the emotion edit pipeline.

```json
{
  "tagged_text": "<emo type=\"happy\" level=\"1\">Hello</emo><emo desc=\"angry and tense\" type_ref=\"angry\">there</emo>"
}
```

Response fields:

| Field | Type | Notes |
|---|---|---|
| `text` | string | `|`-separated segment text for TED-TTS |
| `emo_text` | string | `|`-separated natural-language emotion prompts |
| `segments` | list | Per-segment tag, control mode, level/alpha, text, and rendered prompt |
| `target_seconds` | list/null | Per-segment duration targets when every tag includes `dur` |

### `POST /synthesize_tagged`

Synthesizes canonical `<emo>` tagged text directly. The adapter lives in this
tool: callers send tags, and TED-TTS internally converts them to natural
language emotion prompts before QwenEmotion produces continuous emotion
vectors. Without `dur`, the endpoint uses TED-TTS mode 3 (`use_emo_text=True`)
and does not construct one-hot emotion vectors. Tagged duration control and
`/synthesize` modes 4/5 are temporarily disabled because the upstream TED
duration path has a known conditioning slot-order bug. The expected ordering is
`[duration_ctrl, duration_free]` to match the IndexTTS2 duration-control fix;
the old TED path used `[duration_free, duration_ctrl]`. Use no-duration modes
0/1/2/3 until the duration path is revalidated.

```bash
curl -X POST http://localhost:8035/synthesize_tagged \
  -H 'Content-Type: application/json' \
  -o ted_tagged.wav \
  -d '{
    "tagged_text": "<emo type=\"happy\" level=\"1\">Hello, nice to meet you.</emo><emo desc=\"angry, tense, forceful delivery\" type_ref=\"angry\">I am very upset now.</emo>",
    "spk_audio_prompt": "tools/ted-tts/upstream/datasets/Ref/0011_000001.wav",
    "seed": 42,
    "output_format": "wav"
  }'
```

Duration-control example:

```bash
curl -X POST http://localhost:8035/synthesize_tagged \
  -H 'Content-Type: application/json' \
  -o ted_tagged_duration.wav \
  -d '{
    "tagged_text": "<emo desc=\"soft, slower sadness\" dur=\"1.8\">Hello, nice to meet you.</emo><emo desc=\"tense, faster anger\" dur=\"1.4\">I am very upset now.</emo>",
    "spk_audio_prompt": "tools/ted-tts/upstream/datasets/Ref/0011_000001.wav",
    "seed": 42,
    "output_format": "wav"
  }'
```

## Notes

- Run this service with `pipe-indextts`; do not install TED-TTS as a second editable `indextts` package in that env.
- The service imports its adjacent official `upstream/indextts` package, isolating TED-TTS from the installed IndexTTS package.
- If checkpoints are missing, ask before downloading new model files.

## Official source layout

`upstream/` pins official Simon-leong/TED-TTS commit
`36ffc3e2de346156baa7d60a1749ca4a9365625b`. The service, tag adapter and tests
are owned by this superproject. No environment synchronization is required:
invoke `api_service.py` with an existing pipe-indextts interpreter and set
`CHECKPOINTS_DIR` explicitly when sharing model caches.

`adapter.py` imports the adjacent official source directly. It keeps the active
service adaptations small: duration-conditioning slot order, per-generation
state reset, explicit QwenEmotion device placement,
and BigVGAN CUDA extension loading. The original inference and emotion parsing
methods remain in `upstream/`. Modes 4/5 and tagged duration requests retain their
501 responses.

The service does not consume the optional segment-length diagnostics returned
by upstream inference; these statistics are disabled after decoding completes.

CUDA compilation reads the official `.cpp` and `.cu` sources. PyTorch manages
compilation and caching, with outputs under
`$ARTIFACT_DIR/shared/cache/ted-tts/`. An explicit `TORCH_EXTENSIONS_DIR` selects
that cache root instead, with a tool-specific `ted-tts` subdirectory. The
existing interpreter's `bin` directory supplies `ninja` when necessary.

TED has two different BigVGAN kernels with the same extension name. Each source
has its own build directory; the first loaded kernel is reused by both loader
paths, preserving the service's existing preload behavior.

### Callers and validation

The emotion distributed pipeline and Agent entry call this service through
`TedTTSClient.synthesize_tagged`. Speaker audio, emotion descriptions, sampling
parameters, method and mel-token limit retain their existing behavior. Public
`/synthesize` modes 0–3, `/synthesize_tagged`, `/adapt_tagged`, `/health` and `/info`
remain available.

Tests exercise HTTP request forwarding and audio responses, duration rejection,
conditioning tensor order, generation state, emotion-device placement and CUDA
loading. Run them with an existing `pipe-indextts` interpreter and place pytest
outputs under the external artifact directory. Full synthesis validation also
requires the existing model cache and a GPU.
