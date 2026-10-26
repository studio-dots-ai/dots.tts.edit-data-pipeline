# F0 Evaluation Service

CPU-only FastAPI service for per-span fundamental-frequency summaries. It is
used both for edited-region pitch evaluation and WDTW-F0 preservation scoring.

| Field | Value |
|---|---|
| Default port | `8034` |
| Environment | `pipe-cpu` |
| Backend | Praat/Parselmouth |

## Start

```bash
PYTHONPATH="$PWD" envs/pipe-cpu/.venv/bin/python tools/f0_eval/api_service.py --port 8034
```

## `POST /analyze`

Request:

```json
{
  "audio": {"data": "<base64 wav>", "format": "wav"},
  "spans": [
    {"start": 0.25, "end": 0.75, "label": "edited_word"}
  ],
  "f0_min_hz": 50,
  "f0_max_hz": 600
}
```

Response:

```json
{
  "status": "success",
  "sample_rate": 24000,
  "duration_sec": 1.0,
  "spans": [
    {
      "start": 0.25,
      "end": 0.75,
      "duration_sec": 0.5,
      "label": "edited_word",
      "min_f0_hz": 218.7,
      "max_f0_hz": 223.4,
      "median_f0_hz": 220.1,
      "mean_f0_hz": 221.3,
      "voiced_fraction": 0.96,
      "num_frames": 48,
      "num_voiced_frames": 46
    }
  ]
}
```

If a span has no voiced frames, `min_f0_hz`, `max_f0_hz`, `median_f0_hz`, and
`mean_f0_hz` are `null`, and `voiced_fraction` is `0.0`.

Praat requires at least three periods at the requested pitch floor. Spans
with zero duration or shorter than `3 / f0_min_hz` are therefore returned as
valid empty-F0 results with zero frames instead of producing a validation or
service error.
