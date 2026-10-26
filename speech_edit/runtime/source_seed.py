"""Seed source parquet queues for distributed-pipeline runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pyarrow.parquet as pq

from speech_edit.data.sources import resolve_parquet_sources

from .config import DEFAULT_SOURCE_QUEUE_NAME, source_config
from .queue_store import QueueStore
from .paths import get_artifact_paths, resolve_output_path

ProgressFn = Callable[[str], None]


def _default_progress(message: str) -> None:
    print(message, flush=True)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _source_entries(sources: Sequence[Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(sources):
        if isinstance(raw, Mapping):
            if "path" not in raw:
                raise ValueError(f"source entry {index} must contain path")
            unknown_fields = set(raw) - {"path"}
            if unknown_fields:
                raise ValueError(f"source entry {index} has unsupported fields: {sorted(unknown_fields)}")
            entry = {
                "path": str(raw["path"]),
            }
        else:
            entry = {"path": str(raw)}
        entries.append(entry)
    if not entries:
        raise ValueError("sources must contain at least one source")
    return entries


def _source_specs_for_cache(sources: Sequence[Any]) -> list[dict[str, Any]]:
    return _source_entries(sources)


def _source_cache_key(
    *,
    sources: Sequence[Any],
    rows_per_item: int,
) -> str:
    specs = _source_specs_for_cache(sources)
    hasher = hashlib.sha256()
    hasher.update(json.dumps(specs, sort_keys=True).encode("utf-8"))
    hasher.update(str(rows_per_item).encode("utf-8"))
    for source in specs:
        path = Path(str(source["path"])).expanduser()
        if not path.exists():
            continue
        stat = path.stat()
        hasher.update(str(path.resolve()).encode("utf-8"))
        hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        hasher.update(str(stat.st_size).encode("utf-8"))
    return hasher.hexdigest()[:24]


def _default_source_index_cache_path(
    *,
    run_config: dict,
    sources: Sequence[Any],
    rows_per_item: int,
) -> Path:
    paths = get_artifact_paths(
        run_config.get("project_root"), artifact_dir=run_config.get("artifact_dir")
    )
    cache_key = _source_cache_key(sources=sources, rows_per_item=rows_per_item)
    return paths.resolve(f"source_indexes/{cache_key}.jsonl", scope="shared")


def _read_source_index(
    cache_path: Path,
    *,
    sources: Sequence[Any],
    rows_per_item: int,
) -> list[dict[str, Any]] | None:
    if not cache_path.is_file():
        return None
    with cache_path.open(encoding="utf-8") as handle:
        first = handle.readline()
        if not first:
            return None
        meta = json.loads(first)
        if meta.get("type") != "source_index_meta":
            return None
        if meta.get("sources") != _source_specs_for_cache(sources):
            return None
        if int(meta.get("rows_per_item", -1)) != int(rows_per_item):
            return None
        items = [json.loads(line) for line in handle if line.strip()]
    return items


def _write_source_index(
    cache_path: Path,
    *,
    sources: Sequence[Any],
    rows_per_item: int,
    items: Sequence[dict[str, Any]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.{uuid.uuid4().hex}.tmp")
    handle = tmp_path.open("x", encoding="utf-8")
    try:
        with handle:
            handle.write(
                json.dumps(
                    {
                        "type": "source_index_meta",
                        "version": 1,
                        "sources": _source_specs_for_cache(sources),
                        "rows_per_item": int(rows_per_item),
                        "item_count": len(items),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        tmp_path.replace(cache_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _row_range_items(
    *,
    parquet_path: Path,
    source_file_index: int,
    source_entry_index: int,
    num_rows: int | None,
    rows_per_item: int,
) -> list[dict[str, Any]]:
    resolved = parquet_path.resolve()
    payload_base: dict[str, Any] = {
        "parquet_path": str(resolved),
        "source_file_index": source_file_index,
        "source_entry_index": source_entry_index,
    }
    if rows_per_item <= 0:
        return [
            {
                "item_key": str(resolved),
                "payload": dict(payload_base),
            }
        ]
    if num_rows is None:
        num_rows = pq.ParquetFile(resolved).metadata.num_rows
    items = []
    for row_start in range(0, num_rows, rows_per_item):
        row_stop = min(num_rows, row_start + rows_per_item)
        items.append(
            {
                "item_key": f"{resolved}:{row_start}:{row_stop}",
                "payload": {**payload_base, "row_start": row_start, "row_stop": row_stop},
            }
        )
    return items


def build_source_index(
    *,
    sources: Sequence[Any],
    rows_per_item: int,
    progress: ProgressFn | None = None,
    progress_interval: int = 500,
    metadata_workers: int = 32,
) -> list[dict[str, Any]]:
    progress = progress or _default_progress
    started = time.time()
    source_entries = _source_entries(sources)
    resolved_sources: list[tuple[Path, int]] = []
    for source_entry_index, entry in enumerate(source_entries):
        for parquet_path in resolve_parquet_sources([Path(str(entry["path"]))]):
            resolved_sources.append(
                (parquet_path, source_entry_index)
            )
    progress(
        "source_index_resolved "
        f"files={len(resolved_sources)} rows_per_item={rows_per_item} "
        f"elapsed_sec={time.time() - started:.1f}"
    )
    if rows_per_item <= 0:
        items: list[dict[str, Any]] = []
        for source_file_index, (parquet_path, source_entry_index) in enumerate(
            resolved_sources
        ):
            items.extend(
                _row_range_items(
                    parquet_path=parquet_path,
                    source_file_index=source_file_index,
                    source_entry_index=source_entry_index,
                    num_rows=None,
                    rows_per_item=rows_per_item,
                )
            )
        return items

    items_by_index: dict[int, list[dict[str, Any]]] = {}

    def load_one(
        index_and_source: tuple[int, tuple[Path, int]]
    ) -> tuple[int, list[dict[str, Any]]]:
        source_file_index, (parquet_path, source_entry_index) = index_and_source
        num_rows = pq.ParquetFile(parquet_path.resolve()).metadata.num_rows
        return (
            source_file_index,
            _row_range_items(
                parquet_path=parquet_path,
                source_file_index=source_file_index,
                source_entry_index=source_entry_index,
                num_rows=num_rows,
                rows_per_item=rows_per_item,
            ),
        )

    completed = 0
    with ThreadPoolExecutor(max_workers=max(1, int(metadata_workers))) as executor:
        futures = [
            executor.submit(load_one, (source_file_index, source_entry))
            for source_file_index, source_entry in enumerate(resolved_sources)
        ]
        for future in as_completed(futures):
            source_file_index, source_items = future.result()
            items_by_index[source_file_index] = source_items
            completed += 1
            if completed == 1 or completed % progress_interval == 0 or completed == len(futures):
                item_count = sum(len(value) for value in items_by_index.values())
                progress(
                    "source_index_progress "
                    f"files={completed}/{len(futures)} items={item_count} "
                    f"elapsed_sec={time.time() - started:.1f}"
                )
    items = []
    for source_file_index in range(len(resolved_sources)):
        items.extend(items_by_index[source_file_index])
    return items


def _limit_items_per_input(
    source_items: Sequence[dict[str, Any]],
    *,
    max_items_per_input: int | None,
    progress: ProgressFn,
) -> list[dict[str, Any]]:
    if max_items_per_input is None:
        return list(source_items)
    limit = max(0, int(max_items_per_input))
    counts_by_input: dict[int, int] = {}
    limited: list[dict[str, Any]] = []
    for item in source_items:
        payload = item.get("payload")
        if not isinstance(payload, Mapping) or "source_entry_index" not in payload:
            raise ValueError("source item payload missing source_entry_index")
        source_entry_index = int(payload["source_entry_index"])
        count = counts_by_input.get(source_entry_index, 0)
        if count >= limit:
            continue
        counts_by_input[source_entry_index] = count + 1
        limited.append(item)
    if len(limited) < len(source_items):
        progress(
            "source_seed_limited_per_input "
            f"items={len(limited)}/{len(source_items)} "
            f"max_items_per_input={limit}"
        )
    return limited


def _shuffle_source_items(
    source_items: Sequence[dict[str, Any]],
    *,
    shuffle_items: bool,
    shuffle_seed: int,
    progress: ProgressFn,
) -> list[dict[str, Any]]:
    items = list(source_items)
    if not shuffle_items:
        return items
    rng = random.Random(int(shuffle_seed))
    rng.shuffle(items)
    progress(f"source_seed_shuffled items={len(items)} shuffle_seed={int(shuffle_seed)}")
    return items


def seed_parquet_queue(
    *,
    run_id: str | None = None,
    queue_name: str,
    sources: Sequence[Any],
    rows_per_item: int = 0,
    queue_max_size: int | None = None,
    source_index_cache: str | None = None,
    artifact_dir: str | None = None,
    project_root: str | None = None,
    progress: ProgressFn | None = None,
    progress_interval: int = 500,
    metadata_workers: int = 32,
    insert_batch_size: int = 1000,
    max_items_per_input: int | None = None,
    shuffle_items: bool = False,
    shuffle_seed: int = 0,
) -> int:
    progress = progress or _default_progress
    started = time.time()
    inserted = 0
    source_items = None
    cache_path = (
        resolve_output_path(Path(source_index_cache).expanduser(), project_root=project_root,
                            artifact_dir=artifact_dir, scope="shared")
        if source_index_cache else None
    )
    store = QueueStore(run_id=run_id)
    if cache_path is not None:
        source_items = _read_source_index(
            cache_path,
            sources=sources,
            rows_per_item=rows_per_item,
        )
        if source_items is not None:
            progress(f"source_index_cache_hit path={cache_path} items={len(source_items)}")
    if source_items is None:
        source_items = build_source_index(
            sources=sources,
            rows_per_item=rows_per_item,
            progress=progress,
            progress_interval=progress_interval,
            metadata_workers=metadata_workers,
        )
        if cache_path is not None:
            _write_source_index(
                cache_path,
                sources=sources,
                rows_per_item=rows_per_item,
                items=source_items,
            )
            progress(f"source_index_cache_written path={cache_path} items={len(source_items)}")
    source_items = _shuffle_source_items(
        source_items,
        shuffle_items=shuffle_items,
        shuffle_seed=shuffle_seed,
        progress=progress,
    )
    source_items = _limit_items_per_input(
        source_items,
        max_items_per_input=max_items_per_input,
        progress=progress,
    )
    total = len(source_items)
    batch_size = max(1, int(insert_batch_size))
    next_progress = 1
    for offset in range(0, total, batch_size):
        batch = source_items[offset : offset + batch_size]
        inserted += store.enqueue_json_many(
            queue_name,
            batch,
            max_attempts=1,
            queue_max_size=queue_max_size,
        )
        idx = min(total, offset + len(batch))
        if idx >= next_progress or idx == total:
            progress(
                "source_seed_progress "
                f"items={idx}/{total} inserted={inserted} "
                f"elapsed_sec={time.time() - started:.1f}"
            )
            while next_progress <= idx:
                next_progress += max(1, int(progress_interval))
    return inserted


def seed_source_from_run_config(*, run_config: dict) -> int:
    source = source_config(run_config)
    raw_inputs = source.get("inputs")
    if not raw_inputs:
        raise ValueError("source.inputs must explicitly select input Parquet files, directories, or manifests")
    if isinstance(raw_inputs, (list, tuple)):
        sources = list(raw_inputs)
    elif isinstance(raw_inputs, Mapping):
        sources = [raw_inputs]
    else:
        sources = [str(raw_inputs)]
    queue_limit = source.get("queue_max_size")
    rows_per_item = int(source.get("rows_per_item", 256))
    source_index_cache = source.get("source_index_cache")
    if not source_index_cache:
        source_index_cache = str(
            _default_source_index_cache_path(
                run_config=run_config,
                sources=sources,
                rows_per_item=rows_per_item,
            )
        )
    return seed_parquet_queue(
        run_id=str(run_config["run_id"]),
        queue_name=str(source["queue_name"]),
        sources=sources,
        rows_per_item=rows_per_item,
        queue_max_size=int(queue_limit) if queue_limit is not None else None,
        source_index_cache=str(source_index_cache),
        artifact_dir=run_config.get("artifact_dir"),
        project_root=run_config.get("project_root"),
        progress_interval=int(source.get("progress_interval", 500)),
        metadata_workers=int(source.get("metadata_workers", 32)),
        insert_batch_size=int(source.get("insert_batch_size", 1000)),
        max_items_per_input=(
            int(source["max_items_per_input"])
            if source.get("max_items_per_input") is not None
            else None
        ),
        shuffle_items=_parse_bool(source.get("shuffle_items", False)),
        shuffle_seed=int(source.get("shuffle_seed", 0) or 0),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a distributed-pipeline source parquet queue")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--queue-name", default=DEFAULT_SOURCE_QUEUE_NAME)
    parser.add_argument("--rows-per-item", type=int, default=0)
    parser.add_argument("--queue-max-size", type=int, default=None)
    parser.add_argument("--source-index-cache", default=None)
    parser.add_argument("--metadata-workers", type=int, default=32)
    parser.add_argument("--insert-batch-size", type=int, default=1000)
    parser.add_argument("--progress-interval", type=int, default=500)
    parser.add_argument("--max-items-per-input", type=int, default=None)
    parser.add_argument("--shuffle-items", action="store_true")
    parser.add_argument("--shuffle-seed", type=int, default=0)
    parser.add_argument("sources", nargs="+")
    args = parser.parse_args(argv)
    inserted = seed_parquet_queue(
        run_id=args.run_id,
        queue_name=args.queue_name,
        sources=args.sources,
        rows_per_item=args.rows_per_item,
        queue_max_size=args.queue_max_size,
        source_index_cache=args.source_index_cache,
        metadata_workers=args.metadata_workers,
        insert_batch_size=args.insert_batch_size,
        progress_interval=args.progress_interval,
        max_items_per_input=args.max_items_per_input,
        shuffle_items=args.shuffle_items,
        shuffle_seed=args.shuffle_seed,
    )
    print(f"inserted={inserted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
