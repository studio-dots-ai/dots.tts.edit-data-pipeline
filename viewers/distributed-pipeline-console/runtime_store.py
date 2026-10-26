"""Read-only monitoring of queue, worker and output state."""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any

import psycopg
from psycopg.rows import dict_row
from speech_edit.runtime.config import stage_specs_from_config, queue_max_sizes
from runtime_discovery import RuntimeRunInfo

ACTIVE_WORKER_STATUSES = {"starting", "idle", "working", "backpressure"}


def _queue_max_sizes(config):
    return queue_max_sizes(config, stage_specs_from_config(config))


def _json_loads(value: Any) -> Any:
    if not value:
        return None
    try:
        return json.loads(str(value))
    except json.JSONDecodeError:
        return None

def _truncate(value: Any, limit: int = 180) -> Any:
    if value is None or isinstance(value, (int, float, bool)):
        return value
    text = str(value)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."

def _safe_metadata(value: Any) -> dict[str, Any]:
    data = _json_loads(value)
    if not isinstance(data, dict):
        return {}
    blocked_fragments = ("payload", "audio", "samples", "rows", "data_blob")
    safe: dict[str, Any] = {}
    for key, raw_value in sorted(data.items()):
        lowered = str(key).lower()
        if any(fragment in lowered for fragment in blocked_fragments):
            continue
        if isinstance(raw_value, dict):
            safe[str(key)] = f"object({len(raw_value)})"
        elif isinstance(raw_value, list):
            safe[str(key)] = f"list({len(raw_value)})"
        else:
            safe[str(key)] = _truncate(raw_value)
    return safe

def _queue_graph(run_config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_queues = run_config.get("queues", {})
    declared_queue_names = [
        str(queue_name)
        for queue_name in (raw_queues or {})
        if queue_name is not None
    ] if isinstance(raw_queues, dict) else []
    first_seen: dict[str, int] = {}
    producers: dict[str, list[str]] = defaultdict(list)
    consumers: dict[str, list[str]] = defaultdict(list)
    stage_inputs: dict[str, list[str]] = defaultdict(list)
    stage_outputs: dict[str, list[str]] = defaultdict(list)

    def remember(queue_name: Any) -> str | None:
        if not queue_name:
            return None
        value = str(queue_name)
        first_seen.setdefault(value, len(first_seen))
        return value

    def binding_producers(binding: dict[str, Any]) -> list[str]:
        result: list[str] = []
        raw_producers = binding.get("producers", [])
        if raw_producers is None:
            raw_producers = []
        if isinstance(raw_producers, str):
            raw_producers = [raw_producers]
        if isinstance(raw_producers, list):
            result.extend(str(item) for item in raw_producers if item)
        return list(dict.fromkeys(result))

    def binding_consumers(binding: dict[str, Any]) -> list[str]:
        raw_consumers = binding.get("consumers", [])
        if raw_consumers is None:
            raw_consumers = []
        if isinstance(raw_consumers, str):
            raw_consumers = [raw_consumers]
        if not isinstance(raw_consumers, list):
            return []
        return [str(item) for item in raw_consumers if item]

    for queue_name in declared_queue_names:
        remember(queue_name)

    topology = run_config.get("topology")
    bindings = topology.get("bindings") if isinstance(topology, dict) else None
    if isinstance(bindings, dict) and bindings:
        for raw_queue_name, raw_binding in bindings.items():
            queue_name = remember(raw_queue_name)
            if not queue_name or not isinstance(raw_binding, dict):
                continue
            for producer_name in binding_producers(raw_binding):
                producers[queue_name].append(producer_name)
                stage_outputs[producer_name].append(queue_name)
            for consumer_name in binding_consumers(raw_binding):
                consumers[queue_name].append(consumer_name)
                stage_inputs[consumer_name].append(queue_name)
    edges: set[tuple[str, str]] = set()
    for stage_name, input_queues in stage_inputs.items():
        for upstream in input_queues:
            for downstream in stage_outputs.get(stage_name, []):
                if upstream != downstream:
                    edges.add((upstream, downstream))

    queue_names = list(first_seen)
    outgoing: dict[str, set[str]] = {queue_name: set() for queue_name in queue_names}
    indegree: dict[str, int] = {queue_name: 0 for queue_name in queue_names}
    for upstream, downstream in sorted(edges):
        outgoing.setdefault(upstream, set()).add(downstream)
        indegree.setdefault(downstream, 0)
        indegree[downstream] += 1
    ready = sorted(
        [queue_name for queue_name, count in indegree.items() if count == 0],
        key=lambda queue_name: (first_seen.get(queue_name, 0), queue_name),
    )
    ordered: list[str] = []
    while ready:
        queue_name = ready.pop(0)
        if queue_name in ordered:
            continue
        ordered.append(queue_name)
        for downstream in sorted(outgoing.get(queue_name, ())):
            indegree[downstream] -= 1
            if indegree[downstream] == 0:
                ready.append(downstream)
        ready.sort(key=lambda item: (first_seen.get(item, 0), item))
    for queue_name in sorted(queue_names, key=lambda item: (first_seen.get(item, 0), item)):
        if queue_name not in ordered:
            ordered.append(queue_name)

    graph: dict[str, dict[str, Any]] = {}
    for index, queue_name in enumerate(ordered):
        producer_names = producers.get(queue_name, [])
        consumer_names = consumers.get(queue_name, [])
        if not producer_names and consumer_names:
            role = "source"
        elif producer_names and consumer_names:
            role = "intermediate"
        elif producer_names:
            role = "terminal"
        else:
            role = "declared"
        graph[queue_name] = {
            "flow_order": index,
            "queue_role": role,
            "producer_stages": producer_names,
            "consumer_stages": consumer_names,
        }
    return graph

def _ordered_queues(run_config: dict[str, Any], queues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flow = _queue_graph(run_config)
    max_sizes = _queue_max_sizes(run_config)
    fallback = len(flow) + 1000
    by_name = {str(queue.get("queue_name") or ""): dict(queue) for queue in queues}
    for queue_name in set(flow) | set(max_sizes):
        by_name.setdefault(
            queue_name,
            {
                "queue_name": queue_name,
                "counts": {},
                "total": 0,
                "pending": 0,
                "ready": 0,
                "leased": 0,
            },
        )
    decorated: list[dict[str, Any]] = []
    for queue in by_name.values():
        queue_name = str(queue.get("queue_name") or "")
        queue_meta = flow.get(queue_name, {})
        decorated.append(
            {
                **queue,
                "flow_order": queue_meta.get("flow_order", fallback),
                "queue_role": queue_meta.get("queue_role", "unmatched"),
                "producer_stages": queue_meta.get("producer_stages", []),
                "consumer_stages": queue_meta.get("consumer_stages", []),
                "max_size": max_sizes.get(queue_name),
            }
        )
    return sorted(decorated, key=lambda item: (int(item["flow_order"]), str(item["queue_name"])))


def _queue_stats(conn, *, run_id):
    rows = conn.execute(
        "SELECT queue_name, status, COUNT(*) AS n FROM queue_items "
        "WHERE run_id=%s GROUP BY queue_name, status", (run_id,),
    ).fetchall()
    queues = {}
    for row in rows:
        item = queues.setdefault(row["queue_name"], {
            "queue_name": row["queue_name"], "counts": {}, "total": 0,
            "pending": 0, "ready": 0, "leased": 0,
        })
        status, count = row["status"], int(row["n"])
        item["counts"][status] = count
        item["total"] += count
        if status in {"ready", "leased"}:
            item[status] = count
            item["pending"] += count
    return list(queues.values())


def _parquet_parts(conn, *, run_id):
    summary = conn.execute("""
        SELECT COUNT(*) AS total_parts, COALESCE(SUM(row_count), 0) AS total_rows,
               COALESCE(SUM(byte_count), 0) AS total_bytes,
               MIN(created_at) AS first_created_at, MAX(created_at) AS latest_created_at,
               CASE WHEN COUNT(duration_sec)=COUNT(*) THEN SUM(duration_sec) END AS total_duration_sec
        FROM parquet_parts WHERE run_id=%s
    """, (run_id,)).fetchone()
    parts = conn.execute("""
        SELECT output_path, row_count, byte_count FROM parquet_parts
        WHERE run_id=%s ORDER BY created_at DESC, output_path DESC LIMIT 50
    """, (run_id,)).fetchall()
    return {**summary, "parts": parts}


def _completion_stats(queues: list[dict[str, Any]]) -> dict[str, int]:
    stats: dict[str, int] = defaultdict(int)
    for queue in queues:
        for status, count in queue["counts"].items():
            stats[f"queue_{status}"] += int(count)
        stats["queue_total"] += int(queue["total"])
        stats["queue_pending"] += int(queue["pending"])
    stats["run_complete"] = int(
        stats.get("queue_pending", 0) == 0
        and stats.get("queue_dead", 0) == 0
        and stats.get("queue_total", 0) > 0
    )
    return dict(sorted(stats.items()))

class RuntimeStore:
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or os.environ.get("DIST_PIPELINE_DATABASE_URL")

    def snapshot(self, run: RuntimeRunInfo, *, now: float | None = None) -> dict[str, Any]:
        now = time.time() if now is None else now
        if not self.database_url:
            raise RuntimeError("DIST_PIPELINE_DATABASE_URL is required")
        with psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=10) as conn:
            conn.execute("SET TRANSACTION READ ONLY")
            queues = _ordered_queues(run.run_config, _queue_stats(conn, run_id=run.store_run_id))
            rows = conn.execute(
                "SELECT worker_id, stage, pid, status, first_hbeat_at, updated_at, metadata_json "
                "FROM worker_heartbeats WHERE run_id=%s ORDER BY stage, worker_id",
                (run.store_run_id,),
            ).fetchall()
            state = conn.execute("SELECT desired_state FROM runtime_run_state WHERE run_id=%s", (run.store_run_id,)).fetchone()
            parts = _parquet_parts(conn, run_id=run.store_run_id)
        workers = []
        for row in rows:
            age = max(0, now - row["updated_at"])
            active = row["status"] in ACTIVE_WORKER_STATUSES
            workers.append({
                **{k: v for k, v in row.items() if k != "metadata_json"},
                "metadata": _safe_metadata(row["metadata_json"]),
                "age_sec": age, "active": active and age <= 300,
                "stale": active and age > 300,
            })
        completion = _completion_stats(queues)
        desired = state["desired_state"] if state else "preparing"
        active = sum(w["active"] for w in workers)
        if completion.get("run_complete"):
            status = "completed"
        elif completion.get("queue_dead", 0) or desired == "failed":
            status = "failed"
        elif desired == "stopped":
            status = "stopping" if active else "stopped"
        elif desired == "preparing":
            status = "preparing"
        else:
            status = "running" if active else "waiting_for_workers"
        stages = []
        for name, spec in stage_specs_from_config(run.run_config).items():
            stage_workers = [w for w in workers if w["stage"] == name]
            stages.append({
                "stage": name, "input_queue": spec.input_queue,
                "output_queues": [queue for queue, binding in run.run_config.get("topology", {}).get("bindings", {}).items()
                                  if name in binding.get("producers", [])],
                "writes_parquet": spec.parquet_output.enabled,
                "max_workers": spec.max_workers,
                "active_workers": sum(w["active"] for w in stage_workers),
                "stale_workers": sum(w["stale"] for w in stage_workers),
            })
        return {"run": run.to_dict(), "status": status, "desired_state": desired,
                "queues": queues, "workers": workers, "stages": stages,
                "completion": completion, "parquet": parts}
