# Text Edit Pipeline

Bulk tagged text-edit generation is launched only through the distributed
pipeline runtime under `speech_edit/runtime/`.

The pipeline adapter declares the runnable stages:

- `sample`: reads source parquet payloads and selects generation specs.
- `plan`: calls the LLM and emits audio-free tagged edit plans.
- `synthesize`: executes tagged plans through alignment and F5-TTS.
- `eval`: computes optional evaluation metadata.
- `save`: writes distributed parquet parts through `StageSpec.parquet_output`.

Parquet output is owned by the distributed save stage. The legacy local Data
Saver path and sample-tree-to-parquet path are not supported.

Use the distributed pipeline console for normal large runs. For local debugging,
compose a `text_edit` distributed run config and run workers with the
shared runtime entrypoints documented in
`speech_edit/runtime/README.md`.

The FastAPI service under `service/` is a single-request execution
tool. It is not a bulk pipeline launcher.
