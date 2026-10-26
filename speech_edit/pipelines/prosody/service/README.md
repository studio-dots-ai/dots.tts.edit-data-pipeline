# Prosody Edit Pipeline Service

This service executes one prebuilt prosody tagged instruction through the
shared prosody pipeline core. It accepts source audio plus `instruction_st`
and returns edited audio, edit metadata, and optional evaluation metrics.

## Run

Set `ARTIFACT_DIR` to an absolute directory outside the checkout. The launcher
uses the service graph in `configs/service/prosody_edit_pipeline/` and the shared runtime
process lifecycle. `defaults.yaml` declares generation dependencies;
`evaluate.yaml` adds ASR, quality, speaker similarity and WDTW (including F0).
Generation uses GPU 0; evaluation models use GPU 1. Override the individual
service declarations to use other devices, ports or existing HTTP services.

```bash
# Generation and evaluation dependencies.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/prosody_edit_pipeline/evaluate

# Generation only.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/prosody_edit_pipeline/defaults

# Resolve without starting processes, using existing environments.
/path/to/envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/prosody_edit_pipeline/evaluate --env-root /path/to/envs --dry-run

# Override the API port and output directory through the canonical config.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/prosody_edit_pipeline/evaluate \
  service.prosody_edit_pipeline.port=8036 \
  service.prosody_edit_pipeline.params.output_dir=services/prosody
```

The default API port is `8036`. Dependency URLs are derived from the same
config declarations as their launch commands. A required service or evaluation
failure fails the request. Logs are stored beneath `ARTIFACT_DIR`; interrupting
the launcher stops its managed child processes.

## Request and response

### POST `/execute`

Required fields:

| field | type | meaning |
|---|---|---|
| `audio.data` | string | base64-encoded WAV bytes. |
| `source_text` | string | transcript matching the source audio. |
| `instruction_st` | string | tagged instruction using `<pitch>` and/or `<rate>` tags. |

Supported tags:

```text
<pitch, semitones=3>word</pitch>
<rate, factor=0.8>phrase</rate>
```

Common optional fields:

| field | type | meaning |
|---|---|---|
| `language` | string | `en`/`zh` hint for alignment and evaluation. |
| `sample_id` | string | stable request id. |
| `generation_id` | string | generated wave id, default `1`. |
| `evaluate` | bool/null | default follows launcher mode. |
| `persist` | bool | write WAV/metadata artifacts under `output_dir`. |
| `pitch_method` | string | pitch conversion method, default `psola+indextts2_cco`. |
| `psola_indextts2_cco_pitch_boost` | string | IndexTTS2 CCO pitch boost, default `mul:1.6`. |
| `smooth_policy` | string | boundary smoothing policy. |
| `crossfade_ms` | number | splice crossfade duration. |

Unknown fields are rejected. The service does not support parquet saving.

When `evaluate=true`, the response includes the usual ASR/WER, UTMOS, speaker
similarity, and WDTW metrics plus `region_effects`,
`duration_l1_mean_sec`, `pitch_l1_mean_semitones`, and
`region_effects_status`. Pitch effect metrics require the F0 Evaluation service
from `F0_URL`; rate-only regions can still report duration accuracy without F0.

Minimal Python request:

```python
import base64
import requests

payload = {
    "audio": {
        "data": base64.b64encode(open("source.wav", "rb").read()).decode("utf-8"),
        "format": "wav",
    },
    "source_text": "hello beautiful world",
    "instruction_st": "hello <pitch, semitones=3>beautiful</pitch> world",
    "language": "en",
    "evaluate": False,
}

response = requests.post("http://127.0.0.1:8036/execute", json=payload, timeout=300)
response.raise_for_status()
result = response.json()
open("edited.wav", "wb").write(base64.b64decode(result["audio"]["data"]))
```

### Output storage

Set `ARTIFACT_DIR` to an absolute directory outside the repository before launching the service. Default output and logs are placed in its automatically derived worktree namespace. Relative configured output directories and request output paths resolve in that namespace; absolute overrides must remain beneath `ARTIFACT_DIR`. See [artifact storage](../../../../docs/environment-setup.md#artifact-storage).
