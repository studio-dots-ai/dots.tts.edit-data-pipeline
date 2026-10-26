# Aligner Gateway Service

Unified HTTP gateway for alignment backends.

## Overview

This service proxies alignment requests to the Qwen3 forced aligner in `tools/qwen3-asr/`.
Requests use ISO 639-3 language codes such as `eng` and `zho`; the gateway maps
codes to Qwen3 language names and converts response languages back to codes.
`backend="auto"` selects Qwen3 and infers English or Chinese when the transcript
permits it. Other languages require an explicit language hint.

## Environment

- **uv-managed environment**: `envs/pipe-cpu`
- **Python**: 3.10
- **Port**: 8029

## Start

```bash
envs/pipe-cpu/.venv/bin/python tools/aligner-service/api_service.py \
    --port 8029 \
    --qwen3-url http://localhost:8028
```

## Endpoints

### `GET /health`

Returns shallow gateway health and configured backend URLs. This endpoint does
not call backend services unless `deep=true` is passed.

Use `GET /health?deep=true` to probe every configured backend's `/health`.
Deep health returns HTTP 503 if any configured backend is unhealthy. The gateway
also returns HTTP 503 when no backend URL is configured, because it has no valid
route target.

### `GET /info`

Returns gateway metadata and backend URLs.

### `POST /align`

Routes a single alignment request.

Default request example:

```json
{
  "audio": {"data": "<base64>", "format": "wav"},
  "text": "Hello world.",
  "language": "eng"
}
```

Qwen3 request through the gateway:

```json
{
  "backend": "qwen3",
  "audio": {"data": "<base64>", "format": "wav"},
  "text": "开发边界将作为城市发展的刚性约定",
  "language": "zho"
}
```

#### Request Parameters

| Parameter | Description |
|---|---|
| `backend` | `auto` (default) or `qwen3`. |
| `audio` | Required base64 object `{data, format}` or path/URL string. |
| `text` | Required transcript. |
| `language` | Required for explicit `qwen3`; `auto` can infer English or Chinese. |
| `word_level` | Request word-level alignment. Chinese remains character-level. |
| `queries` | Tagged region queries forwarded unchanged. |

The backend rejects enabled `separate_punctuation`, `debug`, or `debug_dir` options.

Common gateway language codes:

| Code | Language |
|---|---|
| `eng` | English |
| `zho` | Chinese |
| `yue` | Cantonese |
| `jpn` | Japanese |
| `kor` | Korean |
| `fra` | French |
| `deu` | German |
| `spa` | Spanish |
| `por` | Portuguese |
| `ita` | Italian |
| `rus` | Russian |

For convenience, the gateway also accepts `en` as an alias for `eng` and `zh` as an alias for `zho`; responses still use ISO 639-3 codes.

The gateway also recognizes a wider common set for Qwen3 routing, including `ara`, `ces`, `dan`, `ell`, `fas`, `fil`, `fin`, `hin`, `hun`, `ind`, `mkd`, `msa`, `nld`, `pol`, `ron`, `swe`, `tha`, `tur`, and `vie`.

#### Response Fields

| Field | Meaning |
|---|---|
| `status` | Backend success status. |
| `backend` | `qwen3`. |
| `language` | ISO 639-3 language code. |
| `level` | `token`, `word`, or `char`, as returned by Qwen3. |
| `num_chars` | Number of returned alignment entries. |
| `alignments` | Array of aligned entries. |
| `answers` | Query results when queries were supplied. |
| `query_tokenization_matches_default` | Whether query-boundary tokenization matched default tokenization. |

`alignments[]` fields:

| Field | Meaning |
|---|---|
| `char` | Historical field name for the aligned unit. Depending on backend and `level`, this can represent a character, word, or other token-like unit. |
| `start` | Start timestamp in seconds. |
| `end` | End timestamp in seconds. |
| `is_punctuation` | Optional punctuation marker. When punctuation separation is supported and enabled, inserted punctuation entries use `true` and aligned text entries use `false`. Otherwise it is typically `null` or absent. |

Audio exceeding 180 seconds is rejected before forwarding. Query payloads and
response-level query metadata are preserved.

### `POST /align_batch`

Batch alignment.

- For `qwen3`, forwards to the backend batch endpoint after mapping gateway language codes to Qwen3 language names.
- For `auto`, aligns each item individually after language inference.
- Gateway batch responses normalize the top-level fields in the same way as `POST /align`.

## CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--port` | 8029 | Service port |
| `--host` | `0.0.0.0` | Bind host |

| `--qwen3-url` | `http://localhost:8028` | Qwen3 backend URL |
