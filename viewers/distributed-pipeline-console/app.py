"""Console for preparing runs and monitoring externally hosted workers."""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from viewers.security import configure_access
from speech_edit.runtime.paths import get_artifact_paths
from speech_edit.runtime.config import load_run_config
from speech_edit.runtime.queue_store import QueueStore
from speech_edit.runtime.submit_run import prepare_run
from runtime_discovery import discover_run, iter_runs
from runtime_store import RuntimeStore


HEAVY_PARQUET_COLUMNS = {"audios", "speaker_prompt"}

def _run_path(run: Any) -> Path:
    run_path = Path(run.run_dir if getattr(run, "run_dir", None) else run.path).expanduser().resolve()
    if not run_path.is_dir() and getattr(run, "path", None):
        run_path = Path(run.path).expanduser().resolve()
    return run_path

def _resolve_run_parquet_path(run: Any, rel_path: str) -> Path:
    requested = Path(rel_path.strip()).expanduser()
    if not str(requested):
        raise FileNotFoundError("Parquet file path is required")
    base = _run_path(run).resolve()
    path = requested.resolve() if requested.is_absolute() else (base / requested).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"Parquet file escapes run directory: {rel_path}") from exc
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Not a parquet file: {rel_path}")
    if not path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {rel_path}")
    return path

def _display_run_path(run: Any, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_run_path(run).resolve()))
    except (OSError, ValueError):
        return str(path)

def _safe_json_loads(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[\"-0123456789tnf":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        return value

def _json_preview(value: Any) -> Any:
    parsed = _safe_json_loads(value)
    if isinstance(parsed, dict):
        return {str(key): _json_preview(item) for key, item in parsed.items()}
    if isinstance(parsed, list):
        return [_json_preview(item) for item in parsed]
    return parsed

def _audio_map(row: dict[str, Any]) -> dict[str, str]:
    audios = _safe_json_loads(row.get("audios"))
    if not isinstance(audios, dict):
        return {}
    return {
        str(key): value
        for key, value in audios.items()
        if isinstance(value, str) and value.strip()
    }

def _audio_media_type(data: bytes) -> str:
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3"):
        return "audio/mpeg"
    return "application/octet-stream"

def _preview_text(row: dict[str, Any]) -> str:
    for field in ("plain_text", "text", "transcript"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for field in ("pair_metadata", "segments", "instructions"):
        value = _safe_json_loads(row.get(field))
        if isinstance(value, dict):
            for key in ("plain_text", "text", "source_text", "target_text", "label", "st"):
                item = value.get(key)
                if isinstance(item, str) and item.strip():
                    return item.strip()
            for item in value.values():
                if isinstance(item, dict):
                    nested = item.get("label") or item.get("text")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
    return ""

def _preview_instruction(row: dict[str, Any]) -> str:
    instructions = _safe_json_loads(row.get("instructions"))
    if not isinstance(instructions, dict) or not instructions:
        return ""
    st = instructions.get("st")
    if isinstance(st, str) and st.strip():
        return st.strip()
    return json.dumps(instructions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

def _preview_sample(
    row: dict[str, Any],
    *,
    run: Any,
    path: Path,
    row_index: int,
) -> dict[str, Any]:
    metadata = {
        key: _json_preview(value)
        for key, value in row.items()
        if key not in HEAVY_PARQUET_COLUMNS
    }
    audio_keys = sorted(_audio_map(row))
    return {
        "file": _display_run_path(run, path),
        "run_id": run.run_id,
        "row_index": row_index,
        "audio_id": row.get("audio_id") or f"row-{row_index}",
        "from_dataset": row.get("from_dataset"),
        "duration": _json_preview(row.get("duration")),
        "text": _preview_text(row),
        "instruction": _preview_instruction(row),
        "audio_keys": audio_keys,
        "metadata": metadata,
    }

def _read_parquet_sample(run: Any, rel_path: str, *, row_index: int) -> dict[str, Any]:
    path = _resolve_run_parquet_path(run, rel_path)
    parquet_file = pq.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows)
    if row_index < 0 or row_index >= row_count:
        raise IndexError(f"row_index out of range: {row_index}")
    rows = pq.read_table(path).slice(row_index, 1).to_pylist()
    if not rows:
        raise IndexError(f"row_index out of range: {row_index}")
    return _preview_sample(rows[0], run=run, path=path, row_index=row_index)

def _preview_parquet_samples(
    run: Any,
    rel_path: str,
    *,
    mode: str = "top",
    limit: int = 1,
) -> dict[str, Any]:
    path = _resolve_run_parquet_path(run, rel_path)
    parquet_file = pq.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows)
    limit = max(1, min(50, int(limit)))
    if mode == "random":
        indices = sorted(random.sample(range(row_count), min(limit, row_count))) if row_count else []
    elif mode == "top":
        indices = list(range(min(limit, row_count)))
    else:
        raise ValueError("mode must be 'top' or 'random'")
    return {
        "file": _display_run_path(run, path),
        "mode": mode,
        "limit": limit,
        "row_count": row_count,
        "samples": [
            _read_parquet_sample(run, rel_path, row_index=row_index)
            for row_index in indices
        ],
    }

def _read_sample_audio(
    run: Any,
    rel_path: str,
    *,
    row_index: int,
    audio_key: str,
) -> tuple[bytes, str]:
    path = _resolve_run_parquet_path(run, rel_path)
    parquet_file = pq.ParquetFile(path)
    row_count = int(parquet_file.metadata.num_rows)
    if row_index < 0 or row_index >= row_count:
        raise IndexError(f"row_index out of range: {row_index}")
    if "audios" not in parquet_file.schema_arrow.names:
        raise KeyError("parquet file has no audios column")
    rows = pq.read_table(path, columns=["audios"]).slice(row_index, 1).to_pylist()
    if not rows:
        raise IndexError(f"row_index out of range: {row_index}")
    audio_value = _audio_map(rows[0]).get(audio_key)
    if not audio_value:
        raise KeyError(f"audio key not found: {audio_key}")
    try:
        data = base64.b64decode(audio_value, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"audio value is not valid base64: {audio_key}") from exc
    return data, _audio_media_type(data)

def _duration_value_to_sec(value: Any) -> float:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError):
            try:
                numeric = float(value)
            except ValueError:
                return 0.0
            return numeric if math.isfinite(numeric) else 0.0
    if isinstance(value, dict):
        return sum(_duration_value_to_sec(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_duration_value_to_sec(item) for item in value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    return 0.0

def _resolve_run(app: FastAPI, root: Path, date: str, name: str):
    try:
        return discover_run(root, date, name)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(404, str(exc)) from exc


def create_app(data_root: str | Path | None = None, *, host: str = "127.0.0.1") -> FastAPI:
    root = Path(data_root or get_artifact_paths(PROJECT_ROOT).worktree / "distributed_pipeline_runs").resolve()
    app = FastAPI(title="Distributed Pipeline Console")
    configure_access(app, host)
    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def index():
        return FileResponse(static / "index.html")

    @app.get("/sample.html")
    def sample_page():
        return FileResponse(static / "sample.html")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/configs")
    def configs():
        return {"configs": [{"name": str(p.relative_to(PROJECT_ROOT / "configs/pipeline")), "path": str(p)}
                            for p in sorted((PROJECT_ROOT / "configs/pipeline").glob("*/*.yaml"))]}

    @app.get("/api/runs")
    def runs():
        result = []
        for run in iter_runs(root):
            try:
                snapshot = RuntimeStore().snapshot(run)
                result.append({**run.to_dict(), "status": snapshot["status"]})
            except Exception:
                result.append({**run.to_dict(), "status": "unavailable"})
        return {"runs": result}

    @app.post("/api/runs/submit")
    def submit(payload: dict[str, Any]):
        config = str(payload.get("run_config") or "")
        if not config:
            raise HTTPException(400, "run_config is required")
        try:
            # Runs created here stay in the console's discovery root.
            from speech_edit.runtime.submit_run import validate_run_id
            import uuid
            run_id = validate_run_id(str(payload.get("run_id") or uuid.uuid4().hex[:16]))
            pipeline = str(load_run_config(config)["pipeline"])
            if Path(pipeline).name != pipeline or pipeline in {".", ".."}:
                raise ValueError("Invalid pipeline name")
            run_dir = root / pipeline / run_id
            path = prepare_run(run_config_path=config, run_id=run_id, run_dir=str(run_dir))
            return {"status": "ready", "config_path": str(path), "run_id": run_id}
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get("/api/runs/{date}/{name}/dashboard")
    def dashboard(date: str, name: str):
        run = _resolve_run(app, root, date, name)
        try:
            return RuntimeStore().snapshot(run)
        except Exception as exc:
            raise HTTPException(503, "Runtime database unavailable") from exc

    @app.post("/api/runs/{date}/{name}/{action}")
    def control(date: str, name: str, action: str):
        if action not in {"stop", "resume"}:
            raise HTTPException(404, "Unknown action")
        run = _resolve_run(app, root, date, name)
        store = QueueStore(run_id=run.store_run_id, ensure_schema=False)
        try:
            state = "stopped" if action == "stop" else "running"
            if action == "resume" and store.get_run_state()["desired_state"] != "stopped":
                raise HTTPException(409, "Only a stopped run can be resumed")
            store.set_run_state(desired_state=state, updated_by="console")
            return {"desired_state": state}
        finally:
            store.close()

    @app.get("/api/runs/{date}/{name}/parquet-preview")
    def parquet_preview(
        date: str,
        name: str,
        file: str = Query(..., min_length=1),
        mode: str = Query("top", pattern="^(top|random)$"),
        limit: int = Query(1, ge=1, le=50),
    ) -> dict[str, Any]:
        try:
            return _preview_parquet_samples(
                _resolve_run(app, root, date, name),
                file,
                mode=mode,
                limit=limit,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{date}/{name}/parquet-row")
    def parquet_row(
        date: str,
        name: str,
        file: str = Query(..., min_length=1),
        row_index: int = Query(..., ge=0),
    ) -> dict[str, Any]:
        try:
            return _read_parquet_sample(
                _resolve_run(app, root, date, name),
                file,
                row_index=row_index,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, OSError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/runs/{date}/{name}/parquet-audio")
    def parquet_audio(
        date: str,
        name: str,
        file: str = Query(..., min_length=1),
        row_index: int = Query(..., ge=0),
        key: str = Query(..., min_length=1),
    ) -> Response:
        try:
            data, media_type = _read_sample_audio(
                _resolve_run(app, root, date, name),
                file,
                row_index=row_index,
                audio_key=key,
            )
            return Response(content=data, media_type=media_type)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (IndexError, KeyError, ValueError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main():
    parser = argparse.ArgumentParser(description="Monitor pipeline runs")
    parser.add_argument("--data-root")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20110)
    args = parser.parse_args()
    uvicorn.run(create_app(args.data_root, host=args.host), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
