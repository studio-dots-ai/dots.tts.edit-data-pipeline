# Pause Edit Pipeline Service

This service executes one prebuilt pause-edit structured instruction through the
shared pause-edit synthesis core. It accepts source audio plus `instruction_st`
and returns edited audio, pause-edit metadata, and optional evaluation metrics.
One `instruction_st` may contain between 1 and 3 pause insertions, reductions,
or a mixture of both.

## Run

Set `ARTIFACT_DIR` to an absolute directory outside the checkout. The launcher
uses the service graph in `configs/service/pause_edit_pipeline/` and the shared runtime
process lifecycle. `defaults.yaml` declares generation dependencies;
`evaluate.yaml` adds ASR, quality, speaker similarity and WDTW (including F0).
Generation uses GPU 0; evaluation models use GPU 1. Override the individual
service declarations to use other devices, ports or existing HTTP services.

```bash
# Generation and evaluation dependencies.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/pause_edit_pipeline/evaluate

# Generation only.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/pause_edit_pipeline/defaults

# Resolve without starting processes, using existing environments.
/path/to/envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/pause_edit_pipeline/evaluate --env-root /path/to/envs --dry-run

# Override the API port and output directory through the canonical config.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/pause_edit_pipeline/evaluate \
  service.pause_edit_pipeline.port=8037 \
  service.pause_edit_pipeline.params.output_dir=services/pause
```

The default API port is `8037`. Dependency URLs are derived from the same
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
| `instruction_st` | string | pause structured instruction using one or more `<pause act="ins|red" level="1|2|3"/>` tags. |

Example single-pause instruction:

```text
Hello<pause act="ins" level="2"/> world.
```

Example multi-pause instruction:

```text
Hello<pause act="ins" level="1"/> world,<pause act="ins" level="2"/> again.
```

Example pause reduction:

```text
Hello<pause act="red" level="2"/> world.
```

Reduction aligns the adjacent anchor words, removes up to the requested level
duration from the center of the measured gap, and locally smooths the resulting
stitch. It never removes aligned speech when the gap is shorter than requested.

Common optional fields:

| field | type | meaning |
|---|---|---|
| `sample_id` | string | stable request id. |
| `profile_id` | integer | stable profile id, default `1`. |
| `language` | string | optional language hint stored in metadata. |
| `evaluate` | bool/null | default follows launcher mode. |
| `persist` | bool | write WAV/metadata artifacts under `output_dir`. |
| `synth_method` | string | `single_region`, `dual_boundary`, or `insert_only`. |
| `expand_radius` | number | method-dependent context margin. |
| `center_regen_second_pass` | bool | dual_boundary-only second pass. |
| `center_regen_margin_sec` | number/null | explicit dual_boundary center margin. |

Unknown fields are rejected.

In generation-only mode, `/execute` defaults to `evaluate=false`. A request with
`evaluate=true` is rejected because evaluation services are not running. In
evaluation mode, `eval_results` includes service-computed WER/CER, SpkSim,
UTMOS, WDTW, and pause measurement fields.

Use the distributed runtime for parquet dataset output.

### Output storage

Set `ARTIFACT_DIR` to an absolute directory outside the repository before launching the service. Default output and logs are placed in its automatically derived worktree namespace. Relative configured output directories and request output paths resolve in that namespace; absolute overrides must remain beneath `ARTIFACT_DIR`. See [artifact storage](../../../../docs/environment-setup.md#artifact-storage).
