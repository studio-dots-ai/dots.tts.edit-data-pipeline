# Distributed Pipeline Config Guide

This directory contains Hydra-composed configs for distributed speech-editing
pipelines. Configs are runtime contracts: the local runner, worker entrypoint,
console, and pipeline adapters all read the resolved run config.

## Layout

- `pipeline/<pipeline>/`: complete pipeline configs. `defaults.yaml` owns the
  pipeline's business parameters, stage composition, queue topology and worker
  counts. Dataset variants inherit these defaults and override their settings.
- `pipeline/defaults.yaml`: shared runtime settings composed by each pipeline.
- `stage/`: reusable stage profiles. These declare worker
  concurrency, parquet output policy, queue-store pool settings, and claimed
  services.
- `service/`: reusable local service startup profiles. These own service
  command templates, ports, URLs, health checks, startup timeouts, GPU
  visibility, service params, and service-to-service dependencies.
  `defaults.yaml` contains common service settings; `model_defaults.yaml`
  adds the explicit external model and cache paths for local model services.
- `source/`: reusable source dataset profiles.

The console lists `pipeline/*/*.yaml`, including each pipeline's defaults and
its dataset variants. Shared runtime, stage, service and source fragments remain
outside this directory level. For example, submit
`pipeline/text_edit/emilia_en_multiop`.

Hydra package targets are important:

```yaml
defaults:
  - /pipeline/defaults@_global_
  - /stage/plan/gemma4_mtp_tp2@_global_.stages.plan
  - /service/llm_service/gemma4_mtp_tp2@services.llm
  - /stage/defaults@_here_
  - _self_
```

- Use `_global_` from pipeline configs so composed values land at top-level
  keys such as `stages.plan`.
- Use `_here_` from stage configs so `stage/defaults.yaml` fields merge into
  the current stage object.
- Put `_self_` last when the local file should override defaults.

## Configure a deployment

### Input data

Keep input datasets outside the repository, at a location accessible to every
worker. The examples use Emilia. Set `source.inputs` in
`configs/source/emilia.yaml`, or override it in a pipeline YAML:

```yaml
source:
  inputs:
    - path: /path/to/data/parquet
```

Each path may select a directory of Parquet shards, one `.parquet` file, or a
`.list` manifest. A manifest is optional: it contains one shard path per line,
with paths relative to the manifest's directory or absolute paths. Blank lines
and `#` comments are ignored. Directories are scanned recursively.

For example, the directory above can contain:

```text
/path/to/data/parquet/
  part-00000.parquet
  part-00001.parquet
```

The recommended schema stores one utterance per row, with these three columns:

| Column | Type | Content |
| --- | --- | --- |
| `audio` | binary | Complete WAV or FLAC file bytes, not a path or a raw sample array. |
| `text` | string | Nonempty transcript of that utterance. |
| `duration` | number | Duration in seconds, consistent with the audio; used for filtering. |

The reader obtains the sample rate and final duration from the decoded audio.
All four pipelines consume this schema; no additional source metadata is
required. The configured language and duration filters still apply.

The reader also supports the following compact nested representation, using two
JSON-string columns. Each row contains one utterance; only the first entry in
`segments` is read. The `chunk_id` must match the key in `audios`:

```json
{
  "segments": "[{\"chunk_id\":\"000000\",\"text\":\"Hello world.\",\"duration\":3.2}]",
  "audios": "{\"000000\":\"<base64-encoded WAV or FLAC bytes>\"}"
}
```

Emotion editing can use a separate speaker-reference library with the same
Parquet schema. Its reference utterances must be 2–30 seconds long. The provided
language-specific configs set `pipeline_config.speaker_parquet_sources` directly
to external manifest paths:
Speaker manifests must reside under an `en` or `zh` directory, as the loader uses
that directory component to select the language:

```text
/path/to/speakers/
  en/parquet_files.list
  en/part-00000.parquet
  zh/parquet_files.list
  zh/part-00000.parquet
```

Each list can contain `part-00000.parquet`. To use input utterances themselves
as speaker references, set `pipeline_config.speaker_parquet_sources: []` and
provide at least two eligible utterances per sampled group.

Preset input paths are `/path/to/...` placeholders. Replace them in YAML before
running. No dataset or manifest is bundled with the code. Input locations are
separate from `ARTIFACT_DIR`, which holds generated outputs and runtime state.

### Worker environments and PostgreSQL

Set `worker_python_bin` and each service's `python_bin` to existing environments.
Connect the console and workers to an external PostgreSQL database using
`DIST_PIPELINE_DATABASE_URL`. Database credentials belong in the process
environment, not in a committed config. Runs are isolated by `run_id`.

`stages.<name>.max_workers` sets the number of processes started by the local
runner. Each process executes one stage. Allocate CPU, memory and devices
through your execution environment.
For multiple machines, launch stage entrypoints yourself with the same effective
run config and database. Inputs and `ARTIFACT_DIR` must be accessible from each
worker host. See the [runtime guide](../speech_edit/runtime/README.md).

## Author configuration

### Pipeline Configs

Pipeline configs should start from `pipeline/defaults.yaml`, compose all stages
into `stages.<stage_name>`, and then declare the pipeline-specific runtime
surface.

Variants under `pipeline/<pipeline>/` compose their `defaults.yaml` and supply
dataset and task settings. Variants may inherit another variant in the same
directory. `pipeline_config` in the pipeline defaults is also the parameter source
for interactive pipeline services; service profiles explicitly reference it and supply overrides.

Important fields:

- `pipeline`: logical pipeline id, for example `prosody_edit`.
- `adapter_class`: import path for the pipeline adapter. The adapter owns stage
  business logic and may provide custom `stage_specs(run_config)`.
- `run_dir`: output directory. The effective run config,
  logs, artifacts, parquet parts, and control files live under this tree.
- `runtime.run_dir` and `runtime.service_log_root`: usually derived from
  `run_dir`; service configs may interpolate these values.
- `runtime.artifact_cleanup.enabled`: controls shared cleanup of consumed queue
  payload artifacts. It defaults to `true`; set it to `false` only when
  retaining intermediate pickle payloads for debugging.
- `source`: source seeding contract. Prefer `inputs` plus `queue_name`,
  `rows_per_item`, and optional caps such as `max_items_per_input`.
- `pipeline_config`: business-level adapter settings, such as edit policy,
  language, generation modes, model routing choices, parameter distributions, and
  `output_dataset_name`.
- `queues`: queue names and queue-level limits. Keep queue capacity here, not in
  stage configs.
- `topology.bindings`: the only place that wires queues to stages.


Topology is explicit. The runtime must not infer graph edges from queue names,
stage names, stage order, or naming patterns. Use `producers` for every queue
with an upstream stage, even when there is only one producer.

Prosody edit configs may set `pipeline_config.parameter_policy` to control
pitch/rate value distributions. Use weighted pitch buckets with integer
`values`, weighted rate buckets with positive `ranges`, and
`values_per_wave` for the size of the candidate pool exposed to the LLM. Rate
ranges must stay fully below or above `1.0`. Prosody configs may also set
`pipeline_config.region_count_weights` to sample exact edit-region counts and
`pipeline_config.duration_bucket_policy.buckets` to weighted-sample source
utterance duration buckets without changing the hard `min_duration` /
`max_duration` filter.

Emotion style edit configs declare synthesis profiles. `pipeline_config.synthesis_profile_weights` controls
the sample-level distribution over legal `(synthesis_mode,
emotion_condition_mode)` pairs. The backend is derived internally: `single` and
`multi_emo_normal` with `type_level` or `desc` route to IndexTTS2, while
`multi_emo_frequent/desc` routes to TED-TTS. `multi_emo_frequent/type_level` is not supported. TED
length control is split into two knobs:
`ted_estimated_duration_soft_limit_sec` filters profile/window eligibility from
estimated text duration, while `ted_max_generated_duration_sec` is the hard
generation cap passed to TED-TTS and converted to mel tokens by the pipeline
core. The text duration estimator is controlled by
`duration_estimation_en_word_sec` and `duration_estimation_cjk_char_sec`.

```yaml
queues:
  source_items: {max_size: 4096}
  my.sample_to_plan: {max_size: 4096}
  my.plan_to_save: {max_size: 4096}

topology:
  bindings:
    source_items:
      consumers: [sample]
    my.sample_to_plan:
      producers: [sample]
      consumers: [plan]
    my.plan_to_save:
      producers: [plan]
      consumers: [save]
```

Source inputs accept Parquet files, lists or directories. Each `source.inputs`
entry declares a `path`:

```yaml
source:
  inputs:
    - path: /path/to/dataset.list
    - path: /path/to/parquet
  queue_name: source_items
  rows_per_item: 256
  shuffle_items: true
  shuffle_seed: 20260623
```

### Stage Configs

Stage configs are reusable runtime profiles. They should not contain
pipeline-specific business policy unless a pipeline deliberately overrides the
stage under `stages.<name>.pipeline_config`.

Important fields:

- `max_workers`: number of workers started locally for this stage.
- `worker_python_bin`: Python used to start the generic stage worker process.
  The shared default is `envs/pipe-cpu/.venv/bin/python`; model/tool Python
  environments belong in service configs instead.
- `num_clients_per_worker`: number of queue-processing clients inside one
  worker process. Default is `1`; the shared Gemma plan stage uses `16`.
- `lease_seconds`: queue lease duration.
- `queue_store.pool`: DB pool behavior for one worker process. This is
  independent from `num_clients_per_worker`.
- `services`: service profiles claimed by this stage.
- `parquet_output`: save-stage parquet batching and output settings.

Parquet output stages must keep `num_clients_per_worker: 1`. The generic worker
rejects multi-client parquet output because parquet part writing is worker-local
and should not share one writer across multiple queue clients.

Example stage override in a pipeline config:

```yaml
stages:
  plan:
    max_workers: 256
  save:
    parquet_output:
      flush_rows: 512
      flush_interval_sec: 600
```

### Service Configs

Service configs define the service graph for distributed pipeline workers.
Stage configs claim service profiles under `stages.<name>.services`. The resolved
service graph owns process startup, URL exposure, health checks, GPU visibility,
and environment injection. Keep service commands and lifecycle logic in these
profiles rather than pipeline configs or adapters.

Important fields:

- `config_key`: stable service config identity. For files under `service/`, use
  the path relative to `service/` without the `.yaml` suffix, for example
  `aligner_gateway/qwen3` or `f5_tts/defaults`.
- `dep_ref_key`: the caller-owned mapping key used when a service is composed
  under `services` or `dependencies`. This is not written inside the callee
  config; it is the key in YAML such as `services.aligner`. An alias belongs to
  one dependency edge, not to the referenced instance. Different callers may
  use the same alias, and one instance may be referenced through multiple
  aliases. Only one caller binding the same alias to different instances is an
  error.
- `port`, `url`, `health_url`: local endpoint contract. If `url` or
  `health_url` is omitted, the shared runtime derives them from `port` and
  `health_path`.
- `instance_key`: defined in `service/defaults.yaml` as `${.config_key}:${.port}`
  and resolved by Hydra before runtime code reads it. Services with the same
  instance key are the same local instance and start once. If two services use
  the same port with different config keys, or the same instance key with
  different commands, config resolution fails.
- `roots`, `instances`, and `callers`: runtime graph views. `roots` contains
  dependencies declared directly by the worker or pipeline service, keyed by their
  caller-scoped `dep_ref_key`; `instances` is the deduplicated startup registry,
  keyed by `instance_key`; each node's `callers` records the caller instance,
  edge alias, and config path that referenced it. A root caller has no caller
  instance.
- `startup_timeout_sec`: maximum wait time before the worker
  enters stage logic. Use long timeouts for model services with heavy local
  initialization instead of adding sleeps in callers.
- `gpu_ids`: worker-local GPU visibility used when starting the service. In a
  worker process, IDs refer to its `CUDA_VISIBLE_DEVICES`; in
  concurrent workers, `port_stride` offsets service ports while declared
  GPU IDs remain relative to each worker's visible devices.
- `python_bin`: Python used to start the service process. Python service command
  templates should start with `${shell_quote:${.python_bin}}` so the runtime can distinguish
  the generic CPU worker environment from the model/tool environment.
- `params`: service-local structured params used by the command template.
- `env`: service-local environment. Values are strings after Hydra resolution.
- `command`: shell command used by the shared service launcher.
- `dependencies`: recursive full service configs composed below a parent
  service. The resolver builds dependency edges once, starts dependencies before
  their callers, deduplicates instances, and derives health checks from that
  same graph. A service profile owns its internal dependencies and interpolates
  their endpoints into its own command. Recursive Hydra defaults cycles are
  invalid and fail during config composition.
- `managed: false`: declare an endpoint that is owned by the parent service or
  by an external process. The shared launcher reserves and exposes its URL/port
  values but does not start or stop a separate process.

Adapters consume `AdapterContext.services`, the direct-dependency view derived
from `ResolvedServiceGraph.roots`, and resolve endpoints with
`ctx.services.url("asr")`. Each service declares the dependencies it calls in its
own profile. Nested dependencies are private to that service; a stage declares
an additional direct dependency only when its own code calls it. The runtime
applies the same startup ordering and instance deduplication to every service.
Callers must not invent URLs from ports, inspect another service's dependencies,
or start helper processes outside the graph.

Pipelines may override service fields after composition:

```yaml
stages:
  plan:
    services:
      llm:
        port: 18012
        params:
          max_num_seqs: 4
```

### Save Stage And Dataset Names

Distributed save stages use `StageSpec.parquet_output`.

Parquet row metadata uses `pipeline_config.output_dataset_name` for
`from_dataset`; row processors require that name explicitly.

Example:

```yaml
pipeline_config:
  output_dataset_name: Emilia-ProsodyEdit-V1

stages:
  save:
    parquet_output:
      enabled: true
      output_dir: ${runtime.run_dir}/parquet
      dataset_name: ${pipeline_config.output_dataset_name}
      flush_rows: 256
      flush_bytes_mb: 256
      flush_interval_sec: 600
```

### Validation Checklist

- Compose the Hydra config through `load_run_config()` or the submit dry-run
  path before submitting a large run.
- Confirm every stage referenced in `topology.bindings` exists in `stages`.
- Confirm every queue in `topology.bindings` is declared in `queues`.
- Confirm service commands only appear under `service/` configs.
- Confirm save parquet stages have `parquet_output.enabled: true` and
  `num_clients_per_worker: 1`.
- Confirm `pipeline_config.output_dataset_name` is set for distributed parquet
  output.
