# WER/CER Evaluation Service

Compute text error rate between reference and hypothesis texts. The default
`auto` mode uses mixed-language units: CJK characters are scored as individual
units and non-CJK runs are scored as whitespace-delimited words.


## Environment


**Environment**: `envs/pipe-cpu`
**Python**: 3.10
**Key dependencies**: jiwer, numpy

## Start

```bash
source envs/pipe-cpu/.venv/bin/activate
cd tools/wer_calc
python api_service.py --port 8011
```

## Endpoints

### `POST /compute_wer`

Compute error rate for a single pair.

**Request body:**
```json
{
  "reference": "ground truth text",
  "hypothesis": "predicted text",
  "language": "auto"
}
```

- `language`: `"auto"` or `"mixed"` (default) uses mixed-language units. Use
  `"zh"` or `"chinese"` to force CER, any other value for WER.

**Response:**
```json
{
  "error_rate": 0.333,
  "metric": "wer",
  "substitutions": 1,
  "deletions": 0,
  "insertions": 0,
  "hits": 2,
  "ref_length": 3,
  "hyp_length": 3,
  "reference_normalized": "...",
  "hypothesis_normalized": "..."
}
```

### `POST /compute_wer_batch`

Compute error rates for multiple pairs.

**Request body:**
```json
{
  "pairs": [
    {"reference": "...", "hypothesis": "...", "language": "auto"},
    {"reference": "...", "hypothesis": "...", "language": "auto"}
  ],
  "include_aggregate": true
}
```

**Response:**
```json
{
  "results": [...],
  "aggregate": {
    "error_rate": 0.125,
    "substitutions": 1,
    "deletions": 0,
    "insertions": 0,
    "hits": 7,
    "ref_length": 8,
    "hyp_length": 8,
    "metric": "wer",
    "count": 2,
    "valid_count": 2,
    "skipped_count": 0
  }
}
```

`aggregate.error_rate` is sequence-level: total errors divided by total
reference tokens, equivalent to concatenating the batch before scoring. It is
not the arithmetic mean of per-utterance WER/CER.

### `POST /aggregate_wer_results`

Aggregate previously computed WER/CER result dictionaries:

```json
{
  "results": [
    {"substitutions": 1, "deletions": 0, "insertions": 0, "hits": 2, "ref_length": 3, "hyp_length": 3, "metric": "wer"}
  ]
}
```

### `GET /health`

Returns `{"status": "healthy"}`.

## Python Function Interface

```python
from wer_calc import compute_wer, compute_cer, compute_error_rate

# Mixed-language units (CJK chars + non-CJK words)
result = compute_error_rate("reference", "hypothesis")
print(result.error_rate, result.metric)

# Explicit WER / CER
result = compute_wer("the cat sat", "the bat sat")
result = compute_cer("你好世界", "你好时间")
```
