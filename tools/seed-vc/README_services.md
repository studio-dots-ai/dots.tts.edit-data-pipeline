# SeedVC Voice Conversion Service

## Overview

Voice conversion service — converts source audio's speaker identity to match a reference speaker's characteristics while preserving the original content and prosody.

**Environment**: `envs/pipe-gpu`
**Port**: 8003 (default)

## Launch

```bash
source envs/pipe-gpu/.venv/bin/activate
cd tools/seed-vc
python seed_vc_service.py --port 8003
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Root info |
| GET | `/health` | Health check |
| POST | `/convert` | Voice conversion |

---

### POST `/convert`

#### Input (`ConvertRequest`)

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `source_audio` | EncodedAudio or str | Yes | — | Source audio to convert |
| `reference_audio` | EncodedAudio or str | Yes | — | Reference audio for target speaker |
| `diffusion_steps` | int | No | 30 | Number of diffusion steps |
| `length_adjust` | float | No | 1.0 | Length adjustment factor |
| `inference_cfg_rate` | float | No | 0.7 | Classifier-free guidance rate |
| `seed` | int | No | None | Random seed for reproducible inference |
| `output_format` | str | No | `wav` | `wav` or `numpy` |

#### Limitations

- **Reference audio**: Truncated to **25 seconds** internally. Longer references are silently clipped.
- **Source audio**: Inputs longer than **30 seconds** are chunked with overlapping windows and crossfaded.

#### Output

- `wav`: Binary WAV response with headers `X-Sample-Rate`, `X-Duration-Seconds`
- `numpy`: JSON with `EncodedTensor` + `sample_rate`

#### Example

```python
import requests, base64

def encode_audio_file(path):
    with open(path, "rb") as f:
        return {"data": base64.b64encode(f.read()).decode(), "format": "wav"}

resp = requests.post("http://localhost:8003/convert", json={
    "source_audio": encode_audio_file("source.wav"),
    "reference_audio": encode_audio_file("reference.wav"),
    "output_format": "wav",
})
with open("converted.wav", "wb") as f:
    f.write(resp.content)
```

## Official source layout

The service wrapper lives in this directory; `upstream/` pins the official source. Initialize that submodule before launching. Reuse an existing uv interpreter by absolute path in a new worktree. The service activates and verifies this checkout before importing model code, so a previous editable installation cannot silently supply the implementation. Pinned source revisions and patches are recorded in `upstream.json`; regression contracts live in `tests/`.
