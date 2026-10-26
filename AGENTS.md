# Contributor guidance

This repository builds datasets for text, emotion/style, prosody and pause editing. See [README](README.md), [architecture](docs/architecture.md), [environment setup](docs/environment-setup.md) and [service guide](docs/services-guide.md).

## Invariants

- Preserve waveform samples and timing outside the requested edited region.
- Check all callers before changing shared modules or service contracts. Preserve endpoint schemas, defaults, parameter forwarding and error behavior.
- Keep orchestration in pipelines and inference in tool services.
- Keep official `tools/*/upstream` submodules clean. Put adapters and reviewed patches in the outer tool directory and verify import provenance.
- Preserve upstream license files and attribution.

## Development

- Use the appropriate environment under `envs/`, or an existing compatible interpreter by absolute path. Do not modify global environments or synchronize shared environments used by running services.
- Do not download models or datasets implicitly. Configure model caches, credentials, service URLs and output paths explicitly.
- Keep credentials, local machine configuration, generated data, temporary scripts and validation logs outside the checkout. Use `ARTIFACT_DIR` and `speech_edit.runtime.paths` for generated paths; do not hardcode checkout or branch names.
- Keep one authoritative implementation for each behavior, field and default. Reuse it across callers; remove retired implementations rather than adding compatibility wrappers.
- Make focused changes and preserve unrelated edits. Use English for source code, comments and public documentation.
- Read working callers and tool documentation before changing an API. Keep emotion intensity, alignment and duration controls intact.

## Tests and documentation

- Run relevant unit and contract tests for changed behavior. Each endpoint and active calling pattern needs coverage.
- Prefer fixed seeds and meaningful contract, parameter and waveform-preservation assertions. Do not weaken assertions merely to make a changed implementation pass.
- Keep test inputs stable and report skipped integration dependencies separately from passing tests.
- Update the relevant tool's `README_services.md` and the service guide when changing supported APIs or launch requirements.
- Reuse existing environments for source-only worktree changes; new worktrees do not imply new environments.
