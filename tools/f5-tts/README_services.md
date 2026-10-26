# F5-TTS synthesis service

The superproject owns `f5_service/`, the HTTP entrypoint, tests and patches.
`upstream/` pins official SWivid/F5-TTS commit
`27e20fcf39fbce3f187a61d4a3e92065d2d9b947`. See `upstream.json` for provenance.

## Start

Initialize only this dependency (the default Vocos backend does not need the
upstream optional BigVGAN submodule):

```bash
git submodule update --init tools/f5-tts/upstream
/path/to/existing/pipe-gpu/.venv/bin/python tools/f5-tts/api_service.py --port 8001
```

Reuse an existing environment; do not run environment synchronization in a new
worktree. For offline execution, set `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1` and `HF_HUB_CACHE` to the existing model cache.
The service uses F5TTS_v1_Base with the 24 kHz Vocos vocoder. Prepare these
files in the Hugging Face cache selected by `HF_HUB_CACHE`:

| Public model repository | Required files |
|---|---|
| `SWivid/F5-TTS` | `F5TTS_v1_Base/model_1250000.safetensors` |
| `charactr/vocos-mel-24khz` | `config.yaml`, `pytorch_model.bin` |

The model configuration and vocabulary come from the pinned upstream source.
Model weights are provisioned separately under their respective licenses.

CLI options remain `--host` (0.0.0.0), `--port` (8001), `--workers` (1), and
`--log-level` (info). Each worker loads its own model.

## API contract

- `GET /`: service name, status and docs link.
- `GET /health`: readiness and model/vocoder metadata.
- `POST /sample`: TTS or mask-based editing. `/docs` provides full schemas.
- `POST /sample_batch`: sequential batch inference with JSON output; an item failure
  fails the batch. WAV batch output is rejected before inference.

Required request fields are `cond` and `text`. Tensor objects contain base64
`data`, `shape`, and `dtype` (float32 default). Text accepts raw strings or
pre-tokenized token lists. Request controls include:

- `duration`, `duration_sec`, `input_sample_rate`;
- `edit_mask`, `edit_spans_sec` (mutually exclusive);
- `preserve_unmasked_waveform`, `preserve_waveform`,
  `preserve_waveform_sample_rate`, `preserve_spans_sec`;
- `source_crossfade_sec`, `regen_crossfade_sec`;
- `steps`, `cfg_strength`, `sway_sampling_coef`, `seed`;
- `output_format` (`mel`, `numpy`, `wav`, `both`), `trace_id`.

An edit mask uses True for preserved frames and False for regenerated frames.
Preserve mode splices onto the original waveform timeline, retains its exact
length/sample rate, and preserves samples outside the edit/crossfade regions.
The single-call edit duration limit and inference serialization lock remain.
Tests cover model import provenance, exact duration, preservation of unedited
frames, supported request formats and propagation of batch failures.

## Runtime adaptation

`f5_service/core.py` verifies the official HEAD and clean checkout, then selects this checkout's official package even when the
shared environment has an editable install pointing at another checkout. It
rejects an already imported foreign F5 package instead of silently mixing code.

The service calls the official model loader and `CFM.sample()` directly. For
waveform-preserving edits, it passes the requested frame count as both `duration`
and `max_duration`, so upstream's duration clamp prevents automatic extension.
The HTTP layer validates conditioning length and the single-call editing limit
before inference. Ordinary synthesis retains the official duration defaults.

### Service capabilities

The pipelines and `F5TTSClient` use the synthesis service and `/sample`.
Boundary smoothing uses the same endpoint with exact waveform-preservation checks.
The adapter retains crossfade utilities and the active service tests.

## Validation

Start the service before running all tests (never use missing-service skips as
validation):

```bash
SERVICE_URL=http://127.0.0.1:8001 /path/to/existing/pipe-gpu/.venv/bin/python \
  -m pytest tools/f5-tts/tests -v
```

Tests verify API response formats, official import provenance, isolated model loading,
exact duration errors, and preservation of masked mel frames. Test artifacts belong in the
user-selected artifact directory, not inside an environment or upstream checkout.

Tests cover WAV and numpy responses, deterministic fixed-seed synthesis, exact
duration, and boundary smoothing through `F5TTSClient.smooth_boundaries`.
Smoothing checks identical repeats, unchanged sample rate and length,
byte-equivalent PCM outside the edit span, and a change within the span.

### Failure behavior

Batch inference accepts JSON output formats (`mel`, `numpy`, `both`). WAV batch requests are rejected before inference. Any item failure fails the batch request; earlier items are not returned as a completed batch.
