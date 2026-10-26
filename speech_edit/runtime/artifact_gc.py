"""Garbage collect consumed distributed-pipeline queue artifacts."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

from .artifacts import ArtifactStore
from .config import load_run_config
from .queue_store import QueueStore


def _summary_totals(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "item_count": sum(int(row.get("item_count", 0)) for row in rows),
        "size_bytes": sum(int(row.get("size_bytes", 0)) for row in rows),
    }


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def run_gc(
    *,
    run_config_path: str,
    statuses: Sequence[str],
    execute: bool,
    batch_size: int,
) -> dict[str, Any]:
    run_config = load_run_config(run_config_path)
    database_url = os.environ.get("DIST_PIPELINE_DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "DIST_PIPELINE_DATABASE_URL or run_dir/postgres/dist_pipeline_database_url is required"
        )
    store = QueueStore(
        run_id=str(run_config["run_id"]),
        database_url=database_url,
        ensure_schema=False,
    )
    status_list = [str(status) for status in statuses]
    summary = store.artifact_payload_ref_summary(statuses=status_list)
    payload: dict[str, Any] = {
        "run_id": str(run_config["run_id"]),
        "artifact_root": str(run_config["artifact_root"]),
        "statuses": status_list,
        "dry_run": not execute,
        "summary": summary,
        "totals": _summary_totals(summary),
    }
    if not execute:
        return payload

    artifact_store = ArtifactStore(str(run_config["artifact_root"]))
    deleted_count = 0
    bytes_freed = 0
    cleared_count = 0
    failure_count = 0
    while True:
        batch = store.artifact_payload_ref_batch(statuses=status_list, limit=batch_size)
        if not batch:
            break
        clear_ids: list[int] = []
        for item in batch:
            result = artifact_store.delete_ref(
                item["payload_ref"],
                prune_empty_parents=True,
            )
            if result.failures:
                failure_count += len(result.failures)
                continue
            deleted_count += result.deleted_count
            bytes_freed += result.bytes_freed
            clear_ids.append(int(item["id"]))
        cleared_count += store.clear_artifact_payload_refs(
            item_ids=clear_ids,
            statuses=status_list,
        )
        if not clear_ids and failure_count:
            break

    payload["deleted_count"] = deleted_count
    payload["bytes_freed"] = bytes_freed
    payload["cleared_count"] = cleared_count
    payload["failure_count"] = failure_count
    payload["remaining_summary"] = store.artifact_payload_ref_summary(statuses=status_list)
    payload["remaining_totals"] = _summary_totals(payload["remaining_summary"])
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", required=True)
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help="Queue status to clean. May be provided multiple times. Defaults to done.",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true", help="Only summarize DB refs.")
    parser.add_argument("--execute", action="store_true", help="Delete files and clear DB refs.")
    args = parser.parse_args(argv)
    if args.dry_run and args.execute:
        parser.error("--dry-run and --execute are mutually exclusive")
    payload = run_gc(
        run_config_path=args.run_config,
        statuses=args.status or ["done"],
        execute=bool(args.execute),
        batch_size=max(1, int(args.batch_size)),
    )
    _print_json(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
