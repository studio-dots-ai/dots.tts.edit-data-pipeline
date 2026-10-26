# Environment setup

Dependency definitions and lockfiles live under `envs/`. Use a separate interpreter for each incompatible dependency stack.

| Environment | Purpose |
|---|---|
| `pipe-gpu` | F5-TTS, SeedVC, Qwen3-TTS, VAD and GPU evaluation services |
| `pipe-indextts` | IndexTTS2 and TED-TTS |
| `pipe-llm` | Local LLM gateway |
| `pipe-asr` | Qwen3-ASR and forced alignment |
| `pipe-cpu` | Pipeline workers, HTTP clients and CPU signal-processing services |

Consult the corresponding `pyproject.toml` and `uv.lock` for exact Python and dependency requirements. GPU services require a compatible NVIDIA driver; optional compiled kernels also need the build tools specified by their upstream projects.

## Python environments

### Install dependencies

Install [uv](https://docs.astral.sh/uv/), initialize submodules, and select the environment needed for the service:

```bash
git submodule update --init --recursive
bash scripts/setup_venvs.sh --env pipe-gpu --dry-run
bash scripts/setup_venvs.sh --env pipe-gpu
```

Use `--python /absolute/path/to/python` to select a compatible interpreter. Installation is an explicit operation; it can download packages and compile extensions. The setup script uses frozen dependency locks and uv's standard cache configuration. It does not install system packages or GPU drivers.

A single environment can also be installed directly:

```bash
(cd envs/pipe-cpu && uv sync --frozen)
```

Set `UV_CACHE_DIR` if a different package-cache location is needed. On a shared filesystem, uv can reuse cached packages across environments. Do not edit installed package files shared through hard links.

### Reuse an environment

Run a compatible existing interpreter directly:

```bash
/path/to/existing/environment/bin/python tools/vad-service/api_service.py --port 8005
```

For any pipeline service, invoke the shared launcher with an existing interpreter:

```bash
/path/to/envs/pipe-cpu/.venv/bin/python -m speech_edit.runtime.services \
  --config service/pause_edit_pipeline/defaults --env-root /path/to/envs
```

Run this from the checkout. `--env-root` applies to every repo-relative service
`python_bin` in the dependency graph. Absolute per-service paths remain explicit
overrides. The launcher itself runs in the interpreter used on the command line.

Distributed worker configuration accepts explicit Python paths; configure these for the current machine. Keep source worktrees separate from reusable environments and generated outputs. Do not synchronize an environment while other running services depend on it.

## Deployment storage and access

### Model caches and credentials

Provision the model files described in each tool's service documentation before starting inference. Use a shared cache when appropriate:

```bash
export CHECKPOINTS_DIR=/path/to/model-cache
export HF_HUB_CACHE="$CHECKPOINTS_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

`scripts/env.sh` provides the common model-cache and offline settings. Set
`CHECKPOINTS_DIR` explicitly; services do not assume a model directory inside the
checkout. Torch model weights use `$CHECKPOINTS_DIR/torch`; downloaded Python
module caches use `$ARTIFACT_DIR/shared/cache/hf_modules`. `scripts/env.sh` preserves explicit Hugging Face and Torch cache settings. Configured model services inherit
`configs/service/model_defaults.yaml`; override their `env` fields to select
different caches. These paths are saved in the resolved run configuration for
remote workers. Services without local models do not require `CHECKPOINTS_DIR`.
External API credentials and provider endpoints must be supplied through
environment variables; do not commit them. Keep datasets outside the checkout
and configure their paths using the [input data guide](../configs/README.md#input-data).

Use environment variables for shared storage, credentials and library runtime
settings. Dataset manifests, model choices, service parameters and worker resource
selection belong in YAML or explicit CLI arguments.

| Scope | Environment variables | Configuration source |
|---|---|---|
| Shared storage | `ARTIFACT_DIR`, `CHECKPOINTS_DIR` | Deployment shell; required when using artifacts or local models |
| Shared libraries | `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TORCH_HOME`, `HF_MODULES_CACHE`, `TORCH_EXTENSIONS_DIR`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` | Shared defaults or service `env` overrides |
| External API | `GEMINI_API_KEY`, `GEMINI_<MODEL>_API_KEY`, `GEMINI_API_URL` | Deployment secrets and provider endpoint |
| PostgreSQL | `DIST_PIPELINE_DATABASE_URL` | Connection to the external runtime database |

Worker identity, device assignments, console process metadata and per-process
vLLM caches are injected by launchers. They are runtime context rather than
additional user configuration. Qwen3-ASR compilation caches are unique to each
service process, live beneath `ARTIFACT_DIR`, and are removed on normal exit.

### Artifact storage

Set one absolute artifact root outside all checkouts of this repository:

```bash
export ARTIFACT_DIR=/path/to/artifacts/edit-data-pipeline
```

Set it in the launching shell or your deployment environment. The project does not automatically load `.env` files. Deployment secrets and machine-specific paths must remain outside tracked configuration.

The path helper in `speech_edit/runtime/paths.py` derives:

- `worktrees/git-<id>/`: pipeline runs, interactive-service output, viewer state, logs and validation evidence for this checkout.
- `shared/`: storage for reusable artifacts whose callers explicitly choose shared scope.

The ID hashes Git's administrative worktree identity, relative to its common Git directory. Renaming a branch or moving its checkout does not change it. Source exports without Git metadata use a hash of the resolved checkout path, so moving an export changes its namespace. Use one artifact root per repository.

Relative output overrides resolve under the worktree namespace. Absolute output overrides must remain inside `ARTIFACT_DIR` and outside repository checkouts; traversal and existing symlinks are checked. Resolution does not create directories. Missing configuration fails when an output location is needed; imports and `--help` remain available.

```bash
python speech_edit/runtime/paths.py --scope worktree
python speech_edit/runtime/paths.py --scope shared --path cache
```

Distributed run configurations freeze the resolved artifact root and output paths. Worker and PostgreSQL job launch commands propagate that root explicitly. The same storage must be accessible at those paths on remote workers. Viewers may read another explicitly selected data root while keeping their own cache and process state under `ARTIFACT_DIR`.

Pytest uses the configured artifact root automatically: its cache lives under
`worktrees/git-<id>/tests/pytest-cache`, and each invocation gets a unique
`tests/runs/<id>/tmp` directory. Explicit `--basetemp` and `-o cache_dir=...`
overrides must remain under `ARTIFACT_DIR`. Set `PYTHONDONTWRITEBYTECODE=1` to
avoid interpreter bytecode in the source tree.

```bash
PYTHONDONTWRITEBYTECODE=1 /path/to/environment/bin/python -m pytest tests
```

Source-index caches use `shared/source_indexes/` so runs from multiple worktrees
can reuse the same index. Writers use unique temporary files and atomically
replace the completed cache. Relative `source.source_index_cache` overrides are
resolved under `shared/`; absolute overrides must stay within `ARTIFACT_DIR`.
The effective run configuration's artifact root takes precedence over the
calling shell when seeding a configured run.

Tool-internal temporary files and model caches retain their existing settings.

### Network and input trust

Pipeline and tool APIs accept local filesystem paths and perform model inference;
the console can create runs and control task processing. These interfaces do not implement
user authentication. Run them on a trusted private network, with an authenticated
proxy or SSH tunnel for remote access. Viewers bind to `127.0.0.1` by default;
set `--host` explicitly only when network access is controlled.

Restrict PostgreSQL to the workers and operators that need it. Keep database
credentials, logs and resolved run configurations
in access-controlled artifact storage. Queue payloads use Python pickle and may
only come from trusted workers. Treat Hydra configuration, model checkpoints and
third-party model code as executable inputs; accept them only from trusted sources.

## Maintaining dependencies

### Local source dependencies

F5-TTS, IndexTTS2, Qwen3-TTS and Qwen3-ASR package dependencies point to their official `upstream/` directories. SeedVC uses its outer package definition; the aligner gateway runs as a repository script. Launch the outer service entrypoint documented for each tool so its source-selection and patch setup runs before importing model code.

Services import the official submodules from the current checkout's relative `upstream/` directories. Tool adapters implement the required inference changes. CUDA extension outputs belong in the configured external build cache; model weights use the explicit model paths.

Official submodule files must stay clean. See [Architecture](architecture.md) for the adapter and patch boundary.

### Dependency changes and troubleshooting

When changing dependencies, edit the relevant `pyproject.toml`, update its lockfile deliberately, and validate the affected service before sharing the change. Ordinary installation should use the existing frozen lock.

For import failures, first confirm that the selected interpreter belongs to the intended environment and that submodules are initialized. For compilation failures, check the upstream compiler/CUDA requirements and ensure the environment's existing build tools are on `PATH`. For missing models in offline mode, check the configured cache and required filenames rather than initiating an implicit download.
