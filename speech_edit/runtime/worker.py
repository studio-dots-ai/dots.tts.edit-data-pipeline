"""Unified distributed-pipeline worker."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from collections.abc import Mapping
from typing import Any, Optional, Sequence

from speech_edit.runtime.services import ResolvedServiceRoots

from .artifacts import ArtifactRef, ArtifactStore
from .config import load_run_config, queue_max_sizes
from .models import AdapterContext
from .parquet_output import ParquetOutputWriter
from .queue_store import QueueFull, QueueStore
from .registry import get_adapter


DEFAULT_QUEUE_STORE_POOL = {
    "enabled": True,
    "min_size": 2,
    "max_size": 2,
    "timeout_sec": 120.0,
}


def _load_payload(item: Any, artifact_store: ArtifactStore) -> Any:
    if item.payload_ref:
        return artifact_store.read_pickle(ArtifactRef.from_json(item.payload_ref))
    if item.payload_json:
        return json.loads(item.payload_json)
    return None


def _should_stop_before_claim(queue_store: QueueStore) -> bool:
    if queue_store.get_run_state()["desired_state"] != "running":
        return True
    if queue_store.completion_stats().get("run_complete"):
        return True
    return False


def _log_worker_event(event: str, *, stream: Any = None, **payload: Any) -> None:
    record = {"event": event, "time": time.time(), **payload}
    print(
        json.dumps(record, ensure_ascii=False, sort_keys=True, default=str),
        file=stream or sys.stdout,
        flush=True,
    )


def _queue_store_pool_config(
    run_config: Mapping[str, Any],
    stage_name: str,
) -> dict[str, Any]:
    stages = run_config.get("stages") if isinstance(run_config.get("stages"), Mapping) else {}
    stage_raw = stages.get(stage_name) if isinstance(stages, Mapping) else None
    stage = stage_raw if isinstance(stage_raw, Mapping) else {}
    queue_store = stage.get("queue_store") if isinstance(stage.get("queue_store"), Mapping) else {}
    pool = queue_store.get("pool") if isinstance(queue_store.get("pool"), Mapping) else {}
    merged = {**DEFAULT_QUEUE_STORE_POOL, **dict(pool or {})}
    return {
        "enabled": bool(merged.get("enabled", True)),
        "min_size": max(0, int(merged.get("min_size", DEFAULT_QUEUE_STORE_POOL["min_size"]))),
        "max_size": max(1, int(merged.get("max_size", DEFAULT_QUEUE_STORE_POOL["max_size"]))),
        "timeout_sec": float(merged.get("timeout_sec", DEFAULT_QUEUE_STORE_POOL["timeout_sec"])),
    }


def _artifact_cleanup_enabled(run_config: Mapping[str, Any]) -> bool:
    runtime = run_config.get("runtime") if isinstance(run_config.get("runtime"), Mapping) else {}
    cleanup = runtime.get("artifact_cleanup") if isinstance(runtime, Mapping) else {}
    if not isinstance(cleanup, Mapping):
        return True
    value = cleanup.get("enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _cleanup_artifact_refs(
    artifact_store: ArtifactStore,
    refs: Sequence[str],
    *,
    stage_name: str,
    worker_id: str,
    reason: str,
) -> None:
    if not refs:
        return
    result = artifact_store.delete_refs(
        list(refs),
        prune_empty_parents=False,
    )
    if result.deleted_count or result.bytes_freed:
        _log_worker_event(
            "artifact_cleanup",
            stage=stage_name,
            worker_id=worker_id,
            reason=reason,
            deleted_count=result.deleted_count,
            bytes_freed=result.bytes_freed,
            failure_count=len(result.failures),
        )
    for failure in result.failures:
        _log_worker_event(
            "artifact_cleanup_warning",
            stream=sys.stderr,
            stage=stage_name,
            worker_id=worker_id,
            reason=reason,
            path=failure.path,
            error=failure.error,
        )


def _flush_parquet_output(
    parquet_output: Optional[ParquetOutputWriter],
    *,
    force: bool,
    stage_name: str,
    worker_id: str,
    reason: str,
) -> Optional[str]:
    if parquet_output is None:
        return None
    output_path = parquet_output.flush_due(force=force)
    if output_path is not None:
        info = dict(parquet_output.last_flush_info or {})
        info.setdefault("output_path", output_path)
        _log_worker_event(
            "parquet_flush_success",
            stage=stage_name,
            worker_id=worker_id,
            reason=reason,
            **info,
        )
    return output_path


def _write_stage_result(
    *,
    pipeline: str,
    stage_name: str,
    item: Any,
    result: Any,
    queue_store: QueueStore,
    artifact_store: ArtifactStore,
    parquet_output: Optional[ParquetOutputWriter],
    worker_id: str,
    queue_limits: dict[str, Optional[int]],
    artifact_cleanup_enabled: bool,
) -> None:
    if result.parquet_rows is not None:
        if parquet_output is None:
            raise RuntimeError(f"stage {stage_name} returned parquet rows but parquet output is disabled")
        parquet_output.add_item_rows(
            queue_item=item,
            rows=result.parquet_rows,
        )
        _flush_parquet_output(
            parquet_output,
            force=False,
            stage_name=stage_name,
            worker_id=worker_id,
            reason="threshold",
        )
        return

    refs = [
        json.dumps(
            artifact_store.write_pickle(
                f"{pipeline}_{stage_name}_outputs",
                f"{output.queue_name}:{output.item_key}:{uuid.uuid4().hex}",
                output.payload,
            ).to_json(),
            ensure_ascii=False,
            sort_keys=True,
        )
        for output in result.outputs
    ]
    try:
        ack_result = queue_store.ack_and_enqueue(
            item.id,
            worker_id=worker_id,
            outputs=result.outputs,
            payload_refs=refs,
            queue_max_sizes=queue_limits,
            clear_input_payload_ref=artifact_cleanup_enabled,
        )
    except Exception:
        if artifact_cleanup_enabled:
            _cleanup_artifact_refs(
                artifact_store,
                refs,
                stage_name=stage_name,
                worker_id=worker_id,
                reason="ack_failed",
            )
        raise
    if artifact_cleanup_enabled:
        _cleanup_artifact_refs(
            artifact_store,
            getattr(ack_result, "cleanup_payload_refs", []),
            stage_name=stage_name,
            worker_id=worker_id,
            reason="ack_success",
        )


def run_worker(
    *,
    run_config_path: str,
    pipeline: str,
    stage_name: str,
    worker_id: str,
    once: bool = False,
    idle_sleep_sec: float = 5.0,
    services: ResolvedServiceRoots,
) -> int:
    run_config = load_run_config(run_config_path)
    artifact_store = ArtifactStore(str(run_config["artifact_root"]))
    artifact_cleanup_enabled = _artifact_cleanup_enabled(run_config)
    adapter = get_adapter(pipeline, run_config)
    stage_specs = adapter.stage_specs(run_config)
    if stage_name not in stage_specs:
        raise ValueError(f"stage {stage_name!r} is not declared by pipeline {pipeline!r}")
    stage = stage_specs[stage_name]
    if stage.parquet_output.enabled and stage.output_queue is not None:
        raise ValueError(f"stage {stage_name!r} cannot enable parquet output and declare an output queue")
    num_clients_per_worker = max(1, int(stage.num_clients_per_worker))
    if stage.parquet_output.enabled and num_clients_per_worker > 1:
        raise ValueError(
            f"stage {stage_name!r} cannot enable parquet output with "
            f"num_clients_per_worker={num_clients_per_worker}; parquet output stages require 1 client"
        )
    pool_config = _queue_store_pool_config(run_config, stage_name)
    queue_store = QueueStore(
        run_id=str(run_config["run_id"]),
        use_pool=pool_config["enabled"],
        pool_min_size=pool_config["min_size"],
        pool_max_size=pool_config["max_size"],
        pool_timeout=pool_config["timeout_sec"],
        ensure_schema=False,
    )
    queue_limits = queue_max_sizes(run_config, stage_specs)
    ctx = AdapterContext(
        run_config=run_config,
        worker_id=worker_id,
        stage=stage,
        services=services,
    )
    parquet_output = (
        ParquetOutputWriter(
            queue_store=queue_store,
            artifact_store=artifact_store,
            config=stage.parquet_output,
            writer_id=worker_id,
            cleanup_artifacts=artifact_cleanup_enabled,
        )
        if stage.parquet_output.enabled
        else None
    )

    heartbeat_lock = threading.Lock()
    heartbeat_stop = threading.Event()
    latest_heartbeat: dict[str, Any] = {"status": "starting", "metadata": None}

    def heartbeat(status: str, metadata: Optional[dict[str, Any]] = None) -> None:
        with heartbeat_lock:
            latest_heartbeat.update(status=status, metadata=metadata)
            write_heartbeat()

    def write_heartbeat() -> None:
        queue_store.heartbeat(
            worker_id=worker_id,
            pipeline=pipeline,
            stage=stage_name,
            pid=os.getpid(),
            status=latest_heartbeat["status"],
            metadata=latest_heartbeat["metadata"],
        )

    def refresh_heartbeat() -> None:
        while not heartbeat_stop.wait(30):
            try:
                with heartbeat_lock:
                    write_heartbeat()
            except Exception as exc:
                _log_worker_event("heartbeat_error", stream=sys.stderr, worker_id=worker_id, error=str(exc))

    heartbeat_thread = threading.Thread(target=refresh_heartbeat, daemon=True)
    processed = 0
    heartbeat("starting")
    heartbeat_thread.start()
    _log_worker_event(
        "worker_starting",
        pipeline=pipeline,
        stage=stage_name,
        worker_id=worker_id,
        pid=os.getpid(),
        run_config_path=run_config_path,
        input_queue=stage.input_queue,
        output_queue=stage.output_queue,
        lease_seconds=stage.lease_seconds,
        num_clients_per_worker=num_clients_per_worker,
        parquet_output=stage.parquet_output.__dict__,
        queue_store_pool=pool_config,
        artifact_cleanup_enabled=artifact_cleanup_enabled,
    )
    try:
        if not stage.input_queue:
            raise RuntimeError(
                "source stages without input queues are not supported; "
                "seed a source queue and declare it as input_queue"
            )

        processed_lock = threading.Lock()
        stop_event = threading.Event()

        def processed_count() -> int:
            with processed_lock:
                return processed

        def increment_processed() -> int:
            nonlocal processed
            with processed_lock:
                processed += 1
                return processed

        def client_heartbeat(
            status: str,
            metadata: Optional[dict[str, Any]] = None,
            *,
            client_index: int,
        ) -> None:
            heartbeat(
                status,
                {
                    "processed": processed_count(),
                    "client_index": client_index,
                    "num_clients_per_worker": num_clients_per_worker,
                    **dict(metadata or {}),
                },
            )

        def client_loop(client_index: int) -> None:
            while not stop_event.is_set():
                client_heartbeat("idle", client_index=client_index)
                if _should_stop_before_claim(queue_store):
                    stop_event.set()
                    break
                if (
                    stage.output_queue
                    and stage.output_queue in queue_limits
                    and not queue_store.has_queue_capacity(
                        stage.output_queue,
                        requested=1,
                        max_size=queue_limits[stage.output_queue],
                    )
                ):
                    client_heartbeat(
                        "backpressure",
                        {"output_queue": stage.output_queue},
                        client_index=client_index,
                    )
                    _flush_parquet_output(
                        parquet_output,
                        force=False,
                        stage_name=stage_name,
                        worker_id=worker_id,
                        reason="backpressure",
                    )
                    if once:
                        break
                    time.sleep(idle_sleep_sec)
                    continue
                item = queue_store.claim(stage.input_queue, worker_id, stage.lease_seconds)
                if item is None:
                    _flush_parquet_output(
                        parquet_output,
                        force=False,
                        stage_name=stage_name,
                        worker_id=worker_id,
                        reason="idle",
                    )
                    if once:
                        break
                    time.sleep(idle_sleep_sec)
                    continue

                try:
                    client_heartbeat(
                        "working",
                        {"item_key": item.item_key, "attempts": item.attempts},
                        client_index=client_index,
                    )
                    payload = _load_payload(item, artifact_store)
                    result = adapter.process(stage_name, payload, ctx)
                    _write_stage_result(
                        pipeline=pipeline, stage_name=stage_name, item=item, result=result,
                        queue_store=queue_store, artifact_store=artifact_store,
                        parquet_output=parquet_output, worker_id=worker_id,
                        queue_limits=queue_limits, artifact_cleanup_enabled=artifact_cleanup_enabled,
                    )
                    increment_processed()
                except QueueFull as exc:
                    queue_store.release_item(
                        item.id,
                        worker_id=worker_id,
                        reason=str(exc),
                    )
                    if once:
                        break
                except Exception as exc:
                    _log_worker_event(
                        "worker_item_error",
                        stream=sys.stderr,
                        pipeline=pipeline,
                        stage=stage_name,
                        worker_id=worker_id,
                        item_id=item.id,
                        item_key=item.item_key,
                        client_index=client_index,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                    queue_store.fail_item(
                        item.id,
                        worker_id=worker_id,
                        error=f"{exc}\n{traceback.format_exc()}",
                    )
                    if once:
                        raise
                if once:
                    break

        if num_clients_per_worker == 1:
            client_loop(0)
        else:
            with ThreadPoolExecutor(max_workers=num_clients_per_worker) as pool:
                futures = [
                    pool.submit(client_loop, client_index)
                    for client_index in range(num_clients_per_worker)
                ]
                done, pending = wait(futures, return_when=FIRST_EXCEPTION)
                first_exception: BaseException | None = None
                for future in done:
                    first_exception = future.exception()
                    if first_exception is not None:
                        break
                if first_exception is not None:
                    stop_event.set()
                    for future in pending:
                        future.cancel()
                    wait(pending)
                    raise first_exception
                for future in pending:
                    future.result()
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
        final_metadata: dict[str, Any] = {
            "processed": processed,
            "num_clients_per_worker": num_clients_per_worker,
        }
        try:
            _flush_parquet_output(
                parquet_output,
                force=True,
                stage_name=stage_name,
                worker_id=worker_id,
                reason="worker_stop",
            )
        except Exception as exc:
            final_metadata["final_flush_error"] = f"{exc}\n{traceback.format_exc()}"
            _log_worker_event(
                "parquet_flush_error",
                stream=sys.stderr,
                pipeline=pipeline,
                stage=stage_name,
                worker_id=worker_id,
                reason="worker_stop",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
        heartbeat("stopped", final_metadata)
        _log_worker_event(
            "worker_stopped",
            pipeline=pipeline,
            stage=stage_name,
            worker_id=worker_id,
            **final_metadata,
        )
        close_store = getattr(queue_store, "close", None)
        if callable(close_store):
            close_store()
        if "final_flush_error" in final_metadata:
            raise RuntimeError(final_metadata["final_flush_error"])
    return 0
