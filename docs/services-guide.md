# Services Guide

Model and signal-processing services expose FastAPI HTTP APIs. Pipelines call these services through HTTP clients and also use shared Python support modules. Ports below are defaults and can be overridden per deployment.

## Run services

### Starting Services

All services follow the same launch pattern — run the service file with the corresponding environment's Python:

```bash
# General format
envs/<env>/.venv/bin/python tools/<tool>/<service_file>.py [--port PORT] [other args]

# Example: start VAD service
envs/pipe-gpu/.venv/bin/python tools/vad-service/api_service.py --port 8005

# Example: start IndexTTS2 (requires GPU)
CUDA_VISIBLE_DEVICES=0 envs/pipe-indextts/.venv/bin/python \
    tools/index-tts/api_service.py --port 8002

# Example: start LLM service
CUDA_VISIBLE_DEVICES=0 envs/pipe-llm/.venv/bin/python \
    tools/llm-service/api_service.py --local-model Qwen/Qwen3.5-27B --port 8012

# Example: start Gemma 4 31B
CUDA_VISIBLE_DEVICES=0 envs/pipe-llm/.venv/bin/python \
    tools/llm-service/api_service.py --local-model google/gemma-4-31B \
    --gpu-memory-utilization 0.95 --max-model-len 4096 --max-num-seqs 4 --enforce-eager --port 8012
```

### Health Check

All services implement a `/health` endpoint:

```bash
curl http://localhost:8005/health
# {"status": "healthy", "model_loaded": true, ...}
```

### Pipeline service launchers

All four pipeline services use `python -m speech_edit.runtime.services` with
profiles under `configs/service/`. Invoke it with your chosen Python executable.
Pass `--dry-run` to inspect the graph, `--env-root /path/to/envs` to relocate all
service interpreters, and dotted config overrides to change ports, devices or
dependency endpoints. Set `ARTIFACT_DIR` to external output storage. Select
`--config service/<pipeline>_pipeline/defaults` for generation only, or
`--config service/<pipeline>_pipeline/evaluate` to include evaluation dependencies.

Required inference or evaluation errors propagate to the request. Pipelines do
not switch algorithms after a service failure or serialize failed metrics as
successful results. Batch F5, UTMOS and speaker-similarity requests fail if any
item fails. See each service README for supported request formats.
### Pipeline parameters

Each pipeline defines its execution and sampling parameters in
`configs/pipeline/<pipeline>/defaults.yaml`. Distributed runs compose this
file with its `pipeline_config` mapping. Agent service profiles explicitly reference
that file through `params.pipeline_defaults` and keep only overrides in
`params.pipeline`. The YAML service launcher passes `--pipeline-defaults` and
`--pipeline-parameters`; the service merges them before handling requests.

Deployment-specific overrides belong in the composing YAML. For example, the
emotion Agent service requests three candidates while distributed generation
requests one; the pause Agent service disables the second regeneration pass.
HTTP execution controls inherit the configured value when omitted or null;
explicit request values override it. Core functions require resolved parameters
rather than supplying another set of business defaults. Fixed algorithm bounds
and tag definitions remain in their owning implementation.

### Multi-GPU Deployment

Distributed pipeline configs define stage workers and service dependencies. See the [runtime guide](../speech_edit/runtime/README.md) and [configuration guide](../configs/README.md).

## Service reference

### Service Overview

| Port | Service | Env | Directory | Description |
|------|---------|-----|-----------|-------------|
| 8001 | F5-TTS Synthesis | pipe-gpu | `tools/f5-tts/` | Flow-matching TTS with mask prediction |
| 8002 | IndexTTS2 | pipe-indextts | `tools/index-tts/` | Zero-shot TTS with speaker cloning + emotion control |
| 8003 | SeedVC | pipe-gpu | `tools/seed-vc/` | Voice conversion |
| 8004 | Speech Rate Conversion | pipe-cpu | `tools/speech_rate_conversion/` | Time stretching (WSOLA signal-level + token-level) |
| 8005 | VAD | pipe-gpu | `tools/vad-service/` | Voice activity detection and silence trimming |
| 8006 | Volume Normalizer | pipe-cpu | `tools/volume_normalizer/` | Audio loudness normalization |
| 8007 | Qwen3-TTS | pipe-gpu | `tools/qwen3-tts/` | Preset speaker synthesis + zero-shot voice cloning |
| 8010 | ASR Service | pipe-gpu / pipe-asr | `tools/asr-service/` | Unified ASR gateway (current backend: Qwen3-ASR) |
| 8011 | WER/CER | pipe-cpu | `tools/wer_calc/` | Text error rate calculation |
| 8012 | LLM Service | pipe-llm | `tools/llm-service/` | Unified LLM gateway (Gemini API / local Qwen3.5, Qwen3-Omni, Gemma 4 via vLLM) |
| 8017 | UTMOS | pipe-gpu | `tools/utmos/` | Speech quality MOS scoring (1-5) |
| 8018 | Boundary Pause Smoother | pipe-cpu | `tools/boundary-pause-smoother/` | Audio splice smoothing (`smooth_only` seam regen or `adjust` pause-interval editing + F5-TTS regeneration) |
| 8019 | Speaker Similarity (WavLM) | pipe-gpu | `tools/spk-sim-eval/` | Speaker similarity evaluation (WavLM cosine similarity) |
| 8021 | Audio Assembly | pipe-cpu | `tools/audio-assembly/` | Audio splicing + assembly (aligner gateway queries + smoothing) |
| 8027 | Pitch Conversion | pipe-cpu | `tools/pitch_conversion/` | Pitch conversion (PSOLA + optional Qwen3-TTS/IndexTTS2 codec refinement) |
| 8028 | Qwen3 Forced Aligner | pipe-asr | `tools/qwen3-asr/` | Standalone Qwen3 forced alignment service |
| 8029 | Aligner Gateway | pipe-cpu | `tools/aligner-service/` | Unified aligner gateway delegating to the Qwen3 backend |
| 8032 | WDTW Metric | pipe-gpu | `tools/wdtw/` | Word-level DTW temporal fidelity metric for speech editing |
| 8034 | F0 Evaluation | pipe-cpu | `tools/f0_eval/` | Per-span voiced F0 summaries via Praat/Parselmouth |
| 8035 | TED-TTS | pipe-indextts | `tools/ted-tts/` | IndexTTS2-based intra-utterance emotion and duration control, including tagged emotion synthesis |
| 8037 | Emotion Edit Pipeline Service | pipe-gpu + pipeline stack | `speech_edit/pipelines/emotion/service/` | Source-audio emotion edit API backed by IndexTTS2/TED-TTS donor synthesis + Audio Assembly |

> F5-TTS provides the `/sample` synthesis/editing endpoint; pipeline orchestration lives in the four pipelines.
> Emotion Edit Pipeline Service (8037) launches its own dependency stack on override ports by default; see `speech_edit/pipelines/emotion/service/README.md`.

### Per-Service Documentation

Each tool directory contains a `README_services.md` with complete API documentation, request/response schemas, and usage examples:

| Service | Documentation |
|---------|---------------|
| F5-TTS | [tools/f5-tts/README_services.md](../tools/f5-tts/README_services.md) |
| IndexTTS2 | [tools/index-tts/README_services.md](../tools/index-tts/README_services.md) |
| SeedVC | [tools/seed-vc/README_services.md](../tools/seed-vc/README_services.md) |
| Qwen3-TTS | [tools/qwen3-tts/README_services.md](../tools/qwen3-tts/README_services.md) |
| ASR Service | [tools/asr-service/README_services.md](../tools/asr-service/README_services.md) |
| Qwen3-ASR | [tools/qwen3-asr/README_services.md](../tools/qwen3-asr/README_services.md) |
| Aligner Gateway | [tools/aligner-service/README_services.md](../tools/aligner-service/README_services.md) |
| LLM Service | [tools/llm-service/README_services.md](../tools/llm-service/README_services.md) |
| VAD | [tools/vad-service/README_services.md](../tools/vad-service/README_services.md) |
| Speech Rate | [tools/speech_rate_conversion/README_services.md](../tools/speech_rate_conversion/README_services.md) |
| Volume Normalizer | [tools/volume_normalizer/README_services.md](../tools/volume_normalizer/README_services.md) |
| WER/CER | [tools/wer_calc/README_services.md](../tools/wer_calc/README_services.md) |
| WDTW Metric | [tools/wdtw/README_services.md](../tools/wdtw/README_services.md) |
| UTMOS | [tools/utmos/README_services.md](../tools/utmos/README_services.md) |
| Boundary Pause Smoother | [tools/boundary-pause-smoother/README_services.md](../tools/boundary-pause-smoother/README_services.md) |
| Speaker Similarity | [tools/spk-sim-eval/README_services.md](../tools/spk-sim-eval/README_services.md) |
| Audio Assembly | [tools/audio-assembly/README_services.md](../tools/audio-assembly/README_services.md) |
| Pitch Conversion | [tools/pitch_conversion/README_services.md](../tools/pitch_conversion/README_services.md) |
| F0 Evaluation | [tools/f0_eval/README_services.md](../tools/f0_eval/README_services.md) |
| TED-TTS | [tools/ted-tts/README_services.md](../tools/ted-tts/README_services.md) |
| Emotion Edit Pipeline Service | [speech_edit/pipelines/emotion/service/README.md](../speech_edit/pipelines/emotion/service/README.md) |
