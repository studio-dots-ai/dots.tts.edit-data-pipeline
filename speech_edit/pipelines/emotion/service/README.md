# Emotion Edit Pipeline Service

This service executes one structured emotion edit through the shared
emotion/style pipeline synthesis and assembly core. Callers provide source audio,
`source_text`, and a partial `instruction_st` that marks the target emotion
spans.

## Run

Set `ARTIFACT_DIR` to an absolute directory outside the checkout. The launcher
uses the service graph in `configs/service/emotion_style_edit_pipeline/` and the shared runtime
process lifecycle. `defaults.yaml` declares generation dependencies;
`evaluate.yaml` adds ASR, WER, quality, speaker similarity and WDTW (including F0).
Generation uses GPU 0; evaluation models use GPU 1. Override the individual
service declarations to use other devices, ports or existing HTTP services.

```bash
# Generation and evaluation dependencies.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/emotion_style_edit_pipeline/evaluate

# Generation only.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/emotion_style_edit_pipeline/defaults

# Resolve without starting processes, using existing environments.
/path/to/envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/emotion_style_edit_pipeline/evaluate --env-root /path/to/envs --dry-run

# Override the API port and output directory through the canonical config.
envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/emotion_style_edit_pipeline/evaluate \
  service.emotion_style_edit_pipeline.port=8039 \
  service.emotion_style_edit_pipeline.params.output_dir=services/emotion
```

The default API port is `8039`. Dependency URLs are derived from the same
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
| `instruction_st` | string | partial tagged target-emotion instruction. |

`language` is optional for English/Chinese requests. When omitted, the service
infers `eng` or `zho` from `source_text` and passes the resolved code to the
aligner.

Example instruction:

```text
Please read <emo type="happy" level="2">this part</emo> with a smile.
Please read <emo desc="quiet sadness, low energy" type_ref="sad">this part</emo> softly.
```

Only canonical `<emo>` tags are accepted in external `instruction_st`. Leave
unchanged spans untagged. `type+alpha/level` and `desc` are mutually exclusive;
pitch/rate/style/unknown tags are rejected, and `<emo type="none">` is an
internal full-tagged representation that is rejected at the service boundary.
The service strips tags and verifies the result equals `source_text`. In
description mode, every changed `<emo desc="...">` segment must include
`type_ref`, even when the request routes to IndexTTS2. Changed segment metadata
stores `target_emotion_desc` and stores `target_tag` from `type_ref` as the
evaluation reference. The pipeline service derives the synthesis backend from
`synthesis_mode` and `emotion_condition_mode`. Duration-tagged requests are
currently unsupported.

Common optional fields:

| field | type | meaning |
|---|---|---|
| `language` | string | Optional `en`/`zh` hint for alignment and evaluation; inferred when omitted. |
| `sample_id` | string | stable request id. |
| `generation_id` | string | target wave id, default `target`. |
| `evaluate` | bool/null | default follows launcher mode. |
| `persist` | bool | write source/synthetic target/edited WAVs plus metadata. |
| `return_synthetic_target_audio` | bool | include donor audio in response. |
| `max_candidates` | int | synthesis attempts for the synthetic target. |
| `synthesis_mode` | string/null | `single`, `multi_emo_normal`, `multi_emo_frequent`, or `infer`. |
| `emotion_condition_mode` | string/null | `type_level`, `desc`, or `infer`. |
| `multi_frequent_smoothing_radius_sec` | number | Accepted from agent clients; current TED synthesis does not use this field. |
| `aligner_backend` | string/null | `qwen3`; default follows `ALIGNER_BACKEND`; when supplied, it must match the launched stack backend. |

Response fields include edited `audio`, `target_tagged_text`, `edit_tagged_text`,
`changed_segments`, `metadata`, and optional `eval_results`.

### Backend Routing

The service routes `single/type_level`, `single/desc`,
`multi_emo_normal/type_level`, and `multi_emo_normal/desc` to IndexTTS2.
`multi_emo_frequent/desc` routes to TED-TTS. `multi_emo_frequent/type_level`
is rejected.

### TED-TTS Donor Synthesis

When a request routes to TED-TTS, the service synthesizes the full target tagged
text once as a donor utterance. It does not synthesize unchanged spans with a
separate neutral pass. The final edited audio is assembled from the user's
original source audio: unchanged intervals stay from the source, and only
`<target>` intervals are cut from the synthetic donor and spliced into place.

TED requests must use description-mode emotion controls with `type_ref`. The
service converts unchanged internal spans to neutral description prompts. TED
routing and generation limits are configured by
`ted_estimated_duration_soft_limit_sec` and `ted_max_generated_duration_sec` in
`configs/pipeline/emotion_style_edit/defaults.yaml` (17 and 22 seconds).
The service profile may override these values under `params.pipeline`.

### Output storage

Set `ARTIFACT_DIR` to an absolute directory outside the repository before launching the service. Default output and logs are placed in its automatically derived worktree namespace. Relative configured output directories and request output paths resolve in that namespace; absolute overrides must remain beneath `ARTIFACT_DIR`. See [artifact storage](../../../../docs/environment-setup.md#artifact-storage).
