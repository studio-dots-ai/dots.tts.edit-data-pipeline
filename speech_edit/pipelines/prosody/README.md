# Prosody Editing Pipeline

The prosody pipeline creates paired speech samples with local pitch and speaking-rate
edits. It changes the tagged regions and preserves the surrounding audio, subject
to the configured boundary smoothing. Pitch, rate, and mixed edits can target a
word, phrase, clause, or sentence.

## Use the pipeline

For dataset generation, choose a configuration under
[`configs/pipeline/prosody_edit/`](../../../configs/pipeline/prosody_edit/) in the
[distributed console](../../../viewers/distributed-pipeline-console/README.md).
Prepare Parquet inputs according to the
[input data guide](../../../configs/README.md#input-data), then configure model
paths, worker resources and service environments for your deployment.

For one editing request, use the [HTTP service](service/README.md). It accepts
source audio, its transcript and a tagged instruction:

```text
hello <pitch, semitones=3>beautiful</pitch> world
please <rate, factor=0.8>slow this phrase</rate>
```

The service documentation covers startup, request fields and output persistence.
Both entrypoints read business defaults from
[`defaults.yaml`](../../../configs/pipeline/prosody_edit/defaults.yaml);
dataset variants and service profiles supply explicit overrides.

## Processing and output

### Stage flow

1. **Sample:** read utterances from the configured source and apply the language
   and duration filters.
2. **Plan:** select edit kinds, regions and parameter candidates using the YAML
   distributions. The LLM generates tagged text and instructions. Tag validation
   checks the requested regions and preservation of the source transcript.
3. **Synthesize:** align tagged regions through the aligner service, apply the
   configured pitch or rate transformation, and splice the results into the
   source audio. Boundary smoothing uses F5-TTS when enabled by the smoothing
   policy. Required service errors fail the operation.
4. **Evaluate:** compute the configured content, quality, speaker, temporal and
   region-effect metrics. Content evaluation calls ASR and the WER/CER service;
   shared orchestration lives in `speech_edit/evaluation/audio.py`.
5. **Save:** construct source/target pairs between the original and each edited
   wave and write Parquet parts through the runtime writer. Metrics accompany
   the rows so downstream consumers can apply their own selection criteria.

### Service dependencies

| Operation | Service |
|---|---|
| Edit planning | LLM gateway |
| Region alignment | Aligner gateway and Qwen3 forced aligner |
| Pitch transformation | Pitch conversion; IndexTTS2 or Qwen3-TTS according to the selected method |
| Rate transformation | Speech-rate conversion; IndexTTS2 for the codec method |
| Boundary smoothing | F5-TTS |
| Evaluation | ASR, WER/CER, UTMOS, speaker similarity, WDTW and F0 evaluation |

The exact service graph is declared in the
[stage profiles](../../../configs/stage/) and
[service profiles](../../../configs/service/). See the
[service guide](../../../docs/services-guide.md) for environments and endpoints.

### Persisted results

Distributed output consists of Parquet rows containing paired audio, transcripts,
instructions and edit/evaluation metadata. `pipeline_config.output_dataset_name`
owns the dataset label. Runtime metadata records completed Parquet parts after
successful writes.

Interactive requests return edited audio and metadata. Optional persistence writes
WAV and JSON files beneath `ARTIFACT_DIR`; it does not invoke the distributed
Parquet writer. The [distributed pipeline console](../../../viewers/distributed-pipeline-console/README.md)
can inspect completed distributed datasets.

## Code ownership

| File | Responsibility |
|---|---|
| `input_generator.py`, `prompts.py` | Sample planning and instruction generation |
| `tag_utils.py` | Tag parsing and validation |
| `editor.py` | Alignment, configured transforms, splicing and smoothing |
| `pipeline.py` | Per-sample editing and evaluation orchestration |
| `parquet_rows.py` | Dataset row construction |
| `sample_output.py` | Interactive sample persistence |
| `stage_adapter.py` | Distributed stage integration |
| `service/` | Interactive HTTP interface |
