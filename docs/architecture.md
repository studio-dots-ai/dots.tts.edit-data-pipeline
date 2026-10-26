# Architecture and extension boundaries

![Distributed pipeline runtime overview](images/distributed-pipeline-overview.png)

The diagram illustrates one stage topology; each pipeline declares its own graph.

## Layers and ownership

### Pipeline and service layers

The four pipelines generate text, emotion/style, prosody and pause edits. They own sampling, task construction, composition, evaluation and persistence. Distributed workers execute stages using PostgreSQL runtime state and queues; agent-facing pipeline services expose interactive editing APIs.

Tool services perform model inference or signal processing. HTTP clients in `speech_edit/clients/services.py` encode requests and decode responses. Shared Python support modules are also used where appropriate; tool endpoints should not absorb pipeline orchestration.

The unedited region must retain its original waveform and timing. Editing and boundary-smoothing tests verify this directly, including sample counts and PCM equality outside the requested span. Preserve all request fields, defaults and output metadata when changing an adapter.

Emotion tags carry intensity information. The pipeline owns the mapping from `level` to `alpha`; the synthesis service receives and preserves the resulting per-segment alpha values. A parser mismatch should be corrected at the responsible interface rather than discarding intensity controls.

### Package ownership

- `speech_edit/clients/` owns HTTP request encoding, response decoding and service-level errors and retries. It does not start services or orchestrate edits.
- `speech_edit/data/` owns language identifiers, text markup, dataset source resolution, input sampling and generic parquet row representation. It does not depend on pipelines, tool clients or the execution runtime.
- `speech_edit/evaluation/` owns shared audio metrics and their service orchestration. Pipeline-specific edit metrics and result assembly stay in the owning pipeline.
- `speech_edit/runtime/` owns stage execution, queue leases, run control, local process lifecycle and intermediate artifact cleanup. Its `services/` package starts and monitors configured service dependencies. Runtime parquet output coordinates part writes with queue acknowledgements and runtime metadata.
- `speech_edit/pipelines/` owns the four editing flows, their task-specific output rows, stage adapters and interactive APIs. Pipeline adapters implement the runtime interface; editing semantics stay in each pipeline.
- `configs/` contains the Hydra pipeline, stage, service and source declarations consumed by the runtime and console.

Text markup parsing and rendering live in `speech_edit/data/text_markup.py`, used by both the text pipeline and WDTW. Edit planning and audio composition remain pipeline responsibilities. Task-specific emotion, prosody and pause tag contracts stay with their pipelines.

## External source layout

```text
tools/<tool>/
    upstream/          # official repository at the pinned gitlink
    api_service.py     # project service entrypoint, when applicable
    patches/           # maintained source patches, when required
    tests/             # service and adapter contract tests
    README_services.md
```

The six pinned external tools are F5-TTS, IndexTTS2, TED-TTS, SeedVC, Qwen3-TTS and Qwen3-ASR. `.gitmodules` records official source URLs; manifests and gitlinks record commit identity. Preserve upstream licenses and resource files.

Services load the official implementation from their relative `upstream/` directory. Tool-local adapters and in-memory patches provide the required inference changes; their service documentation describes the supported behavior. Keep adaptations outside the pinned official submodule and generated build outputs in the configured artifact storage.

When updating an upstream commit, audit reachable call sites, endpoint schemas, defaults, forwarded arguments, inference controls and errors. Verify import provenance so a previously installed editable package cannot silently replace the selected source. Test the actual deployed modes, including required alignment and smoothing dependencies.

## Development

### Validation

Use fixed seeds where applicable and verify service contracts, parameter forwarding and preservation of unedited waveform regions. Missing models or unhealthy dependencies must be reported, not counted as successful validation. Database-backed and real pipeline tests may require explicitly configured services; report their execution separately from unit tests.

Each service documents its API, model requirements and launch flags. Consult the [service guide](services-guide.md), [environment guide](environment-setup.md) and [distributed runtime guide](../speech_edit/runtime/README.md) before extending a calling pattern.

### Tests and generated output

Project tests mirror ownership under `tests/clients`, `tests/data`, `tests/evaluation`, `tests/runtime`, `tests/pipelines`, `tests/viewers` and `tests/scripts`. Tool-specific tests stay under each tool's `tests/` directory.

Run and interactive-service output lives under the worktree namespace derived from `ARTIFACT_DIR`; viewers use the same default data root. Shared artifacts use the root's `shared/` namespace when explicitly selected by their caller. Generated files are independent of the source checkout. See [artifact storage](environment-setup.md#artifact-storage).

Each pipeline exposes its HTTP API through `service/api_service.py`. Use `python -m speech_edit.runtime.services --config service/<pipeline>_pipeline/evaluate` to launch its configured dependency graph, or start the API directly against existing services. Pipeline-specific maintenance commands live under that pipeline's `scripts/` directory.
