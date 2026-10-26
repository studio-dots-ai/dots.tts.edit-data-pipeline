# Emotion/Style Editing Pipeline

This pipeline creates paired speech-edit data for fine-grained emotion edits.
Each pair keeps the unedited regions from the source audio and changes only the
tagged target span.

## Supported Entrypoints

The active entrypoints are:

- Distributed pipeline: `stage_adapter.py`, launched through
  `speech_edit/runtime`.
- Pipeline-as-service: `service/api_service.py`, launched through `speech_edit.runtime.services`, used for
  request-level structured emotion edits.

## Run the pipeline

### Distributed Pipeline

The distributed adapter runs the pipeline as six runtime stages:

1. `sample`: reads source parquet rows and builds generation batches.
2. `plan`: calls the LLM service to produce canonical `<emo>` tagged texts and
   paired edit instructions.
3. `synth_index`: consumes `emotion.plan_to_synth.index` and runs IndexTTS2
   synthesis for Index-routed profiles.
4. `synth_ted`: consumes `emotion.plan_to_synth.ted` and runs TED-TTS synthesis
   for TED-routed profiles.
5. `eval`: computes ASR, emotion, quality, speaker similarity, WDTW, and
   alignment-backed metrics according to config.
6. `save`: writes parquet rows through the distributed runtime output sink.

Both synth stages use the same adapter code and write to `emotion.synth_to_eval`;
the split queues make backend-specific worker allocation and backlog debugging
explicit.

The default config is
`configs/pipeline/emotion_style_edit/defaults.yaml`.

#### Speaker Parquet Sources

`pipeline_config.speaker_parquet_sources` optionally supplies an external
speaker prompt pool for the distributed `sample` stage. Supported forms are:

- a `speaker_timbre_library` root containing `zh/` and/or `en/`;
- a language manifest such as `speaker_timbre_library/en/parquet_files.list`;
- explicit parquet files or list files whose path contains `/zh/` or `/en/`.

When configured, the sample stage lazily creates one cyclic speaker sampler per
language in each worker. Generation text still comes from the main source queue;
only the speaker prompt is sampled from the external speaker pool. If no speaker
source is configured, the sample stage uses the
shortest segment in each sampled batch becomes the speaker prompt and the other
segments provide generation text.

### Pipeline-as-service

The service accepts source audio, `source_text`, and a partial
`instruction_st`. It expands the request into:

- a full source tagged text with unchanged regions marked as `none`;
- a full target tagged text containing the requested target emotion spans;
- an assembly markup string where changed spans are wrapped in `<target>`.

For TED-TTS routes, the service synthesizes one full target donor utterance.
The final edited audio still preserves the user's original source audio outside
the changed spans; only `<target>` intervals are cut from the synthetic donor
and spliced into the original source audio.

See `service/README.md` for request fields and launcher details.
## Edit representation

### Tagged Text

The canonical tag format is XML-like:

```text
<emo type="happy" level="2">text</emo>
<emo desc="bright, relieved, smiling delivery" type_ref="happy">text</emo>
<emo type="none">neutral text</emo>
```

`type+level` and `desc+type_ref` are mutually exclusive. Full tagged texts must
cover the whole plain text. Index neutral regions use `<emo type="none">...</emo>`;
TED description-mode neutral regions use a neutral natural-language `desc` and
a reference `type_ref`.

### Synthesis Profiles

Each sample/request has an explicit synthesis profile:

- `synthesis_mode`: `single`, `multi_emo_normal`, or `multi_emo_frequent`.
- `emotion_condition_mode`: `type_level` or `desc`.
- `synthesis_backend`: derived internal routing metadata, `index` or `ted`.

Backend routing is determined by the profile matrix:

| synthesis mode | `type_level` | `desc` |
|---|---|---|
| `single` | Index | Index |
| `multi_emo_normal` | Index | Index |
| `multi_emo_frequent` | illegal | TED |

#### IndexTTS2 Backend

IndexTTS2 supports `single` and `multi_emo_normal` profiles with
`type+level` or natural-language `desc+type_ref` emotion controls. The pipeline
maps level to the IndexTTS2 intensity value before synthesis; description
profiles pass the natural-language prompt as `emo_text`. `multi_emo_normal`
uses per-segment synthesis, concatenation, and boundary smoothing. IndexTTS2
can handle longer utterances than the current TED-TTS tagged path.

#### TED-TTS Backend

TED-TTS is used only for `multi_emo_frequent/desc` in the active dataset
pipeline. It uses TED-TTS mode 3 / no-duration whole-tagged synthesis with
natural-language `desc` controls plus `type_ref` evaluation labels.

TED-TTS has a shorter synthesis length budget in this integration. The sampler
uses a soft estimated-duration limit of 17 seconds for TED routing; actual TED
generation uses a separate hard cap, default `ted_max_generated_duration_sec=22`
(about `max_mel_tokens=1100`). Longer text
should remain on Index-backed `single` or `multi_emo_normal` until a dedicated
length-routing policy is added.

#### Distributed Run Statistics

Distributed parquet rows include a JSON `pair_metadata` column with
`sample_id`, `language`, `synthesis_profile`, `synthesis_backend`,
`synthesis_mode`, `emotion_condition_mode`, `prompted_edit_mode`,
`estimated_duration_sec`,
`ted_estimated_duration_soft_limit_sec`, `ted_max_generated_duration_sec`,
`ted_max_mel_tokens`, `selected_text_window`, `pair_change_summary`, and
`speaker_prompt_metadata`. `prompted_edit_mode` is the edit intent sampled before
the LLM call; actual tag-level differences are recorded in each row's `tag_diff`
column rather than post-hoc classified into type/intensity buckets. Use these
fields for run-wide sanity checks before listening:

```bash
envs/pipe-gpu/.venv/bin/python - <<'PY'
from collections import Counter
from pathlib import Path
import json
import pandas as pd

from speech_edit.runtime.paths import resolve_output_path
run_dir = resolve_output_path("distributed_pipeline_runs/emotion_style_edit/<run_id>")
rows = []
for path in run_dir.glob("parquet/worker_*/part-*.parquet"):
    rows.append(pd.read_parquet(path))
df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
meta = df["pair_metadata"].map(json.loads) if "pair_metadata" in df else []
print("rows", len(df))
print("samples", len({m.get("sample_id") for m in meta}))
print("profiles", Counter((m.get("synthesis_mode"), m.get("emotion_condition_mode")) for m in meta))
print("backends", Counter(m.get("synthesis_backend") for m in meta))
print("modes", Counter(m.get("synthesis_mode") for m in meta))
print("condition_modes", Counter(m.get("emotion_condition_mode") for m in meta))
print("prompted_edit_modes", Counter(m.get("prompted_edit_mode") for m in meta))
print("ted_estimated_sec", [
    round(float(m.get("estimated_duration_sec") or 0.0), 2)
    for m in meta
    if m.get("synthesis_backend") == "ted"
][:20])
PY
```

For success/failure, inspect distributed queue states in the runtime DB and the
worker logs for dead items. For eval summaries, aggregate `eval_results` from
the same parquet rows. Description-mode samples use `type_ref` as the coarse
emotion classifier target, while ASR, UTMOS, speaker similarity, and WDTW remain
available when their services are enabled.
