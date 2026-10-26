# Text Edit Pipeline Service

This service executes prebuilt tagged text-edit instructions through the shared
text editing core. Callers send source audio plus `instruction_st`;
the service returns generated audio, span metadata, and optionally evaluation
metrics. The service does not accept `operations` or edit graphs.

This service is for single-sample execution. Bulk
dataset generation must use the distributed pipeline runtime; this service does
not replace the distributed launcher or write distributed parquet output.

## Run

Set `ARTIFACT_DIR` to an absolute directory outside the checkout. The launcher
uses the service graph in `configs/service/text_edit_pipeline/` and the shared runtime
process lifecycle. `defaults.yaml` declares generation dependencies;
`evaluate.yaml` adds ASR, quality, speaker similarity and WDTW (including F0).
Generation uses GPU 0; evaluation models use GPU 1. Override the individual
service declarations to use other devices, ports or existing HTTP services.

```bash
# Generation and evaluation dependencies.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/text_edit_pipeline/evaluate

# Generation only.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/text_edit_pipeline/defaults

# Resolve without starting processes, using existing environments.
/path/to/envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/text_edit_pipeline/evaluate --env-root /path/to/envs --dry-run

# Override the API port and output directory through the canonical config.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/text_edit_pipeline/evaluate \
  service.text_edit_pipeline.port=8038 \
  service.text_edit_pipeline.params.output_dir=services/text
```

The default API port is `8038`. Dependency URLs are derived from the same
config declarations as their launch commands. A required service or evaluation
failure fails the request. Logs are stored beneath `ARTIFACT_DIR`; interrupting
the launcher stops its managed child processes.

## Request and response

### Request Defaults

The service uses the following defaults:

```json
{
  "duration_mode": "estimate",
  "expand_radius": 0.4,
  "merge_gap": 0.1,
  "nfe_step": 32,
  "cfg_strength": 2.0,
  "sway_sampling_coef": -1.0,
  "seed": 42,
  "source_crossfade_sec": 0.02,
  "regen_crossfade_sec": 0.0
}
```

`duration_mode="synthesis"` is supported, but must be requested explicitly.

### POST `/execute`

Required request fields:

| field | type | meaning |
|---|---|---|
| `audio.data` | string | base64-encoded WAV bytes. |
| `source_text` | string | transcript of the source audio. |
| `instruction_st` | string | tagged edit instruction, e.g. `<sub targ="kind">brave</sub>`. |

Common optional fields:

| field | type | meaning |
|---|---|---|
| `language` | string | source language hint such as `en`, `eng`, `zh`, or `zho`. |
| `sample_id` | string | stable request/sample id for outputs. |
| `generation_id` | string | generation key, default `000`. |
| `evaluate` | bool/null | request metrics; default follows launcher mode. |
| `persist` | bool | write generated artifacts under `output_dir`. |
| `save_parquet` | bool | must be false; use distributed save workers for parquet output. |
| `output_dir` | string/null | override artifact output directory. |
| `duration_mode` | string | `estimate`, `synthesis`, or `generation`. |
| `synthesis_mode` | string | `one_shot` or `per_op`; default `one_shot`. |
| `debug` | bool | write and return per-step debug artifact metadata. |
| `debug_output_dir` | string/null | override debug artifact root. |
| `expand_radius` | number | seconds to expand edit spans before generation. |
| `merge_gap` | number | merge expanded spans when the gap is smaller than this. |
| `nfe_step` | integer | F5 sampling steps. |
| `cfg_strength` | number | F5 CFG strength. |
| `sway_sampling_coef` | number | F5 sway sampling coefficient. |
| `seed` | integer | F5 random seed. |
| `source_crossfade_sec` | number | F5 source-boundary crossfade. |
| `regen_crossfade_sec` | number | F5 regenerated-region crossfade. |

Unknown fields are rejected. In particular, `operations` and `edit_operations`
are invalid service inputs.

Minimal Python request:

```python
import base64
import requests

wav_path = "source.wav"
payload = {
    "audio": {
        "data": base64.b64encode(open(wav_path, "rb").read()).decode("utf-8"),
        "format": "wav",
    },
    "source_text": "hello brave world",
    "instruction_st": "hello <sub targ=\"kind\">brave</sub> world",
    "language": "en",
    "evaluate": False,
}

response = requests.post("http://127.0.0.1:8035/execute", json=payload, timeout=300)
response.raise_for_status()
result = response.json()
open("generated.wav", "wb").write(base64.b64decode(result["audio"]["data"]))
```

Response fields:

| field | meaning |
|---|---|
| `status` | `ok` on success. |
| `pipeline_type`, `pipeline_version` | Text pipeline identity. |
| `sample_id`, `generation_id` | request/result ids. |
| `target_text` | rendered target transcript from `instruction_st`. |
| `audio.data` | base64-encoded generated WAV bytes. |
| `source_spans` | source edit spans in seconds. |
| `target_spans` | target timeline spans in seconds. |
| `target_aligned_spans` | aligned generated-audio target spans. |
| `eval_results` | metrics when `evaluate=true`; otherwise empty. |
| `metadata` | span plan, F5 request metadata, persistence metadata, and pair metadata. |
| `debug` | debug manifest when `debug=true`; otherwise null. |

### Output storage

Set `ARTIFACT_DIR` to an absolute directory outside the repository before launching the service. Default output and logs are placed in its automatically derived worktree namespace. Relative configured output directories and request output paths resolve in that namespace; absolute overrides must remain beneath `ARTIFACT_DIR`. See [artifact storage](../../../../docs/environment-setup.md#artifact-storage).
