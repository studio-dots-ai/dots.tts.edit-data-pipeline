# Audio Assembly Service

This service assembles a new waveform from two renditions of the same transcript.

- `source_audio`: audio used for untagged spans
- `target_audio`: audio used for `<target>...</target>` spans
- `edit_tagged_text`: full transcript with target spans marked inline

The service strips `<target>` tags to recover the base transcript, then either:

- uses caller-provided time spans directly when the manual-span fast path is eligible, or
- sends full-text queries to the aligner gateway for both audios.

In both cases it splices alternating source/target spans and smooths splice junctions through the boundary pause smoother.

By default, the service preserves the entire first and last segment of the final assembled audio. This applies to both aligner-derived spans and caller-provided manual spans.

## Start

From the repo root:

```bash
envs/pipe-cpu/.venv/bin/python tools/audio-assembly/api_service.py --port 8021
```

With custom upstream services:

```bash
envs/pipe-cpu/.venv/bin/python tools/audio-assembly/api_service.py \
  --port 8021 \
  --aligner-url http://localhost:8029 \
  --smoother-url http://localhost:8018
```

## API

### `POST /assemble`

Request body:

```json
{
  "source_audio": {"data": "<base64>", "format": "wav"},
  "target_audio": {"data": "<base64>", "format": "wav"},
  "edit_tagged_text": "<target>Monday,</target> there's gonna be haze, but Tuesday, look for <target>thunderstorms.</target>",
  "time_spans_from_source": [[0.5, 2.0]],
  "time_spans_from_target": [[0.0, 0.6], [2.3, 3.1]],
  "preserve_outer_boundaries": true,
  "language": "eng",
  "aligner_backend": "qwen3",
  "smoothing": {
    "regen_margin_sec": 0.5,
    "smooth_continuation_boundaries": true,
    "f5_steps": 32,
    "f5_cfg_strength": 2.0,
    "f5_sway_sampling_coef": -1.0,
    "f5_seed": 42
  },
  "output_format": "wav",
  "debug": false,
  "trace_id": "example-trace"
}
```

Rules:

- `edit_tagged_text` must be the full transcript.
- Only `<target>...</target>` markup is allowed.
- Nested tags are rejected.
- Adjacent target spans are merged internally before alignment.
- The service uses char-level alignment for robustness on mid-phrase edits.
- `aligner_backend` is optional and selects the aligner gateway backend: `auto` (default), or `qwen3`.
- `language` is optional and is forwarded to the aligner gateway when provided.
- `smoothing.regen_margin_sec` is optional. When omitted, Boundary Pause Smoother uses its default `regen_margin_sec=0.15`.
- `smoothing.smooth_continuation_boundaries` defaults to `true`. Smoothing only runs when this flag is true and there is at least one splice point.
- `preserve_outer_boundaries=true` keeps the full first and last assembled segment, regardless of whether they come from `source_audio` or `target_audio`.
- Set `preserve_outer_boundaries=false` to trim all segments strictly to aligner answers or provided manual spans.

### Optional manual spans

You may also provide:

- `time_spans_from_source`: one `[start_sec, end_sec]` pair for each plain-text source segment
- `time_spans_from_target`: one `[start_sec, end_sec]` pair for each raw `<target>...</target>` span in the request text

The service skips alignment only when all of the following are true:

- both span fields are provided
- the raw request text has no abstract-adjacent targets, meaning there is no `</target>` followed only by whitespace and then `<target>`
- `time_spans_from_target` count matches the number of raw `<target>` spans
- `time_spans_from_source` count matches the expected number of source segments
- both span lists are monotonic, non-overlapping, in-range, and every span satisfies `start < end`

Invalid or ineligible manual spans are rejected. Omit both span fields to request alignment-based assembly.

When `preserve_outer_boundaries=true`, the manual-span path still overrides the first segment start to `0.0` and the last segment end to the corresponding audio duration.

### WAV response

When `output_format="wav"`, the response body is WAV audio with headers:

- `X-Sample-Rate`
- `X-Duration-Sec`
- `X-Splice-Points`
- `X-Target-Spans`
- `X-Trace-Id` when provided

### JSON response

When `output_format="json"`, the response contains:

- encoded WAV audio
- sample rate and duration
- splice point count
- resolved target spans with keys:
  - `text`
  - `start_char`
  - `end_char`
  - `query`
  - `source_answer`
  - `target_answer`
- assembly details

For passthrough-target mode, `source_answer` is `null` and `target_answer` covers the full target audio duration.

## Caller compatibility

The existing `edit_tagged_text`-only workflow remains unchanged. The manual span fields and `preserve_outer_boundaries` are optional and orthogonal to the existing smoothing, debug, output format, and tracing options.

## Debug output

When `debug=true`, the service writes:

- `source.wav`
- `target.wav`
- `00_pre_smoothing.wav`
- `01_assembled.wav`
- `debug_info.json`

If smoothing runs with debug enabled, smoother artifacts are written under `smoothing/` inside the chosen debug directory.

## Failure behavior

Manual assembly requires both source and target spans. Invalid spans or a layout that cannot be represented by the supplied spans are rejected before alignment or smoothing.
