# WDTW Metric Service

The service returns three independent metrics from one source/target alignment:

- `wdtw`: the legacy Word-level Dynamic Time Warping metric.
- `wdtw_dur`: duration preservation over corrected preserved-word masks.
- `wdtw_f0`: equal-word-weighted min/max/mean F0 preservation error in semitones.

The AST paper leaves the local DTW cost unspecified. This implementation uses a duration cost:

- same normalized word: `abs(source_duration - target_duration)`
- different normalized words: `source_duration + target_duration + word_mismatch_penalty`

By default the service computes `mode="preserved"`, which uses the LCS between source and target transcripts and scores only unchanged words. If `edit_st_inst` is provided, WDTW also excludes words covered by the structured edit instruction. `mode="all"` mirrors the paper pseudocode more directly by scoring all aligned transcript words and does not apply `edit_st_inst` masking.

## Environment

- Python environment: `envs/pipe-gpu` recommended by the current task; `envs/pipe-cpu` also works because the metric itself is CPU-only.
- Default port: `8032`
- Dependencies: an aligner gateway is required; an F0 service is optional at
  the API level and configured by the standard service profile. WDTW-F0 is
  unavailable when audio or the F0 dependency is missing, without failing the
  other two metrics.

## Start

```bash
envs/pipe-gpu/.venv/bin/python tools/wdtw/api_service.py \
  --port 8032 \
  --aligner-url http://localhost:8029 \
  --aligner-backend qwen3 \
  --f0-url http://localhost:8034
```

## Endpoints

### `GET /health`

Returns service status and whether the optional F0 dependency is configured.

### `GET /info`

Returns service metadata and supported modes.

### `POST /compute_wdtw`

Compute WDTW for one source/target pair.

Request with precomputed alignments:

```json
{
  "source_transcript": "hello world",
  "target_transcript": "hello brave world",
  "source_alignments": [
    {"char": "hello", "start": 0.0, "end": 0.4},
    {"char": "world", "start": 0.4, "end": 0.9}
  ],
  "target_alignments": [
    {"char": "hello", "start": 0.0, "end": 0.4},
    {"char": "brave", "start": 0.4, "end": 0.7},
    {"char": "world", "start": 0.7, "end": 1.2}
  ],
  "mode": "preserved"
}
```

Request with audio paths, using the aligner gateway:

```json
{
  "source_audio": "/abs/path/source.wav",
  "target_audio": "/abs/path/target.wav",
  "source_transcript": "hello world",
  "target_transcript": "hello brave world",
  "language": "eng",
  "mode": "preserved"
}
```

Request with one cached alignment and audio for the missing side:

```json
{
  "source_transcript": "hello world",
  "target_transcript": "hello brave world",
  "source_alignments": [
    {"word": "hello", "start": 0.0, "end": 0.4},
    {"word": "world", "start": 0.4, "end": 0.9}
  ],
  "target_audio": "/abs/path/target.wav",
  "language": "eng"
}
```

If both alignments are provided, no aligner call is made. If either side is
missing an alignment, provide that side's audio so the service can align only
the missing side. Cached source/target alignments must use compatible levels
(for example both word-like or both char-like).

The aligner URL and backend are fixed when the WDTW service starts. Requests
cannot inspect or override this internal dependency. The service always requests
`word_level=true` from the aligner gateway and rejects token-level responses.

Optional `edit_st_inst`:

```json
{
  "source_transcript": "hello small world again",
  "target_transcript": "hello big world again",
  "edit_st_inst": "hello <sub targ=\"big\">small</sub> <happy>world</happy> again",
  "source_alignments": [
    {"char": "hello", "start": 0.0, "end": 0.4},
    {"char": "small", "start": 0.4, "end": 0.7},
    {"char": "world", "start": 0.7, "end": 1.2},
    {"char": "again", "start": 1.2, "end": 1.8}
  ],
  "target_alignments": [
    {"char": "hello", "start": 0.0, "end": 0.4},
    {"char": "big", "start": 0.4, "end": 0.75},
    {"char": "world", "start": 0.75, "end": 1.65},
    {"char": "again", "start": 1.65, "end": 2.25}
  ]
}
```

`edit_st_inst` can be used with explicit source/target transcripts, or as the only transcript source. It supports text edit tags (`<del>`, `<ins>`, `<sub targ="...">`) plus arbitrary business tags such as `<happy>...</happy>` or `<rate, factor=1.3>...</rate>`. Business tag contents are present on both source and target sides but are excluded from preserved-mode WDTW. Self-closing point tags such as `<pause act="ins"/>` do not cover tokens, so they do not affect WDTW. Nested or malformed tags are rejected.

Use `preservation_st_inst` to define masks for the two new metrics without
changing legacy WDTW:

```json
{
  "source_audio": "/abs/path/source.wav",
  "target_audio": "/abs/path/target.wav",
  "source_transcript": "keep pitch rate",
  "target_transcript": "keep pitch rate",
  "preservation_st_inst": "keep <pitch>pitch</pitch> <rate>rate</rate>"
}
```

WDTW-Dur retains words inside `pitch` tags and excludes all other paired edit
tags. WDTW-F0 excludes every paired edit tag, including `pitch`. Text edit tags
are excluded from both, while self-closing pause tags exclude no words. If
`preservation_st_inst` is omitted, it falls back to `edit_st_inst`, then to
transcript LCS selection. These two metrics always use preserved-word selection;
the request `mode` continues to control only legacy WDTW.

Response:

```json
{
  "wdtw": 0.0,
  "distance": 0.0,
  "normalizer": 1.8,
  "mode": "preserved",
  "status": "ok",
  "num_source_segments": 2,
  "num_target_segments": 2,
  "num_preserved_pairs": 2,
  "preserved_pairs": [[0, 0], [1, 2]],
  "wdtw_dur": 0.0,
  "wdtw_dur_result": {
    "wdtw_dur": 0.0,
    "distance": 0.0,
    "normalizer": 1.8,
    "status": "ok"
  },
  "wdtw_f0": 0.42,
  "wdtw_f0_result": {
    "wdtw_f0": 0.42,
    "distance": 0.84,
    "normalizer": 2.0,
    "status": "ok",
    "unit": "semitones",
    "eligible_word_count": 2,
    "valid_word_count": 2,
    "skipped_word_count": 0
  }
}
```

When `mode="preserved"` and the source/target texts share no unchanged words, `wdtw` is `null` and `status` is `no_segments`.

### `POST /compute_wdtw_batch`

Compute WDTW for a list of requests:

```json
{
  "items": [
    {
      "source_transcript": "hello world",
      "target_transcript": "hello world",
      "source_alignments": [{"char": "hello", "start": 0.0, "end": 0.4}],
      "target_alignments": [{"char": "hello", "start": 0.0, "end": 0.5}]
    }
  ],
  "include_aggregate": true
}
```

When `include_aggregate=true`, the response includes sequence-level WDTW:
`sum(distance) / sum(normalizer)` over valid items. Items with
`status="no_segments"` or `normalizer=0` are skipped and counted.

### `POST /aggregate_wdtw_results`

Aggregate previously computed full WDTW result dictionaries:

```json
{
  "results": [
    {"wdtw": 0.1, "distance": 0.2, "normalizer": 2.0, "mode": "preserved", "status": "ok"}
  ]
}
```

## Failure behavior

Requests containing audio require the configured F0 dependency when preserved
words exist. Dependency failures fail the HTTP request. Alignment-only requests
compute timing metrics; F0 is unavailable because audio was not provided.
An edit with no preserved words has no F0 preservation score.
