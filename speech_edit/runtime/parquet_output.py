"""Worker-local parquet batching for terminal stages."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .artifacts import ArtifactStore
from .models import ClaimedItem, ParquetOutputConfig
from .queue_store import QueueStore

logger = logging.getLogger(__name__)


class ParquetOutputWriter:
    """Accumulate rows in memory and ack queue items after a successful write."""

    def __init__(
        self,
        *,
        queue_store: QueueStore,
        artifact_store: Optional[ArtifactStore] = None,
        config: ParquetOutputConfig,
        writer_id: str,
        cleanup_artifacts: bool = False,
    ):
        if not config.output_dir:
            raise ValueError("ParquetOutputConfig.output_dir is required when parquet output is enabled")
        self.queue_store = queue_store
        self.artifact_store = artifact_store
        self.cleanup_artifacts = bool(cleanup_artifacts and artifact_store is not None)
        self.config = config
        self.writer_id = writer_id
        self.output_dir = Path(config.output_dir)
        self.worker_output_dir = self.output_dir / f"worker_{writer_id}"
        self.worker_output_dir.mkdir(parents=True, exist_ok=True)
        self._items: List[ClaimedItem] = []
        self._rows: List[Dict[str, Any]] = []
        self._byte_count = 0
        self._batch_started_monotonic: Optional[float] = None
        self.last_flush_info: Optional[Dict[str, Any]] = None

    def add_item_rows(
        self,
        *,
        queue_item: ClaimedItem,
        rows: List[Dict[str, Any]],
    ) -> None:
        if not rows:
            ack_result = self.queue_store.ack_and_enqueue(
                queue_item.id,
                worker_id=self.writer_id,
                outputs=[],
                payload_refs=[],
                clear_input_payload_ref=self.cleanup_artifacts,
            )
            self._cleanup_ack_result(ack_result, reason="empty_rows")
            return
        if self._batch_started_monotonic is None:
            self._batch_started_monotonic = time.monotonic()
        self._items.append(queue_item)
        self._rows.extend(rows)
        self._byte_count += _estimate_rows_bytes(rows)

    def flush_due(self, *, force: bool = False) -> Optional[str]:
        self.last_flush_info = None
        if not self._rows:
            return None
        if not force and not self._is_due():
            return None

        items = list(self._items)
        rows = list(self._rows)
        output_path = self._next_part_path()
        flush_started_monotonic = time.monotonic()
        batch_age_sec = (
            flush_started_monotonic - self._batch_started_monotonic
            if self._batch_started_monotonic is not None
            else None
        )
        invalid_items = [
            item
            for item in items
            if not self.queue_store.item_is_leased_by(item.id, worker_id=self.writer_id)
        ]
        if invalid_items:
            reason = f"parquet output lease expired for {len(invalid_items)} queue item(s)"
            self._release_items(items, reason=reason)
            self._clear()
            raise RuntimeError(reason)
        try:
            self._write_part(rows, output_path)
        except Exception as exc:
            self._release_items(items, reason=str(exc))
            self._clear()
            raise

        ack_error: Optional[Exception] = None
        cleanup_refs: List[str] = []
        for item in items:
            try:
                ack_result = self.queue_store.ack_and_enqueue(
                    item.id,
                    worker_id=self.writer_id,
                    outputs=[],
                    payload_refs=[],
                    clear_input_payload_ref=self.cleanup_artifacts,
                )
                cleanup_refs.extend(getattr(ack_result, "cleanup_payload_refs", []))
            except Exception as exc:  # noqa: BLE001 - preserve the first ack failure after writing.
                if ack_error is None:
                    ack_error = exc
        self._clear()
        self._cleanup_refs(cleanup_refs, reason="parquet_flush_ack")
        if ack_error is not None:
            raise ack_error
        duration_sec = _sum_rows_duration(rows)
        byte_count = int(output_path.stat().st_size)
        self.queue_store.record_parquet_part(
            output_path=str(output_path),
            writer_id=self.writer_id,
            row_count=len(rows),
            byte_count=byte_count,
            duration_sec=duration_sec,
        )
        self.last_flush_info = {
            "output_path": str(output_path),
            "row_count": len(rows),
            "item_count": len(items),
            "byte_count": byte_count,
            "duration_sec": duration_sec,
            "flush_elapsed_sec": time.monotonic() - flush_started_monotonic,
            "batch_age_sec": batch_age_sec,
            "force": force,
        }
        return str(output_path)

    def _is_due(self) -> bool:
        if len(self._rows) >= max(1, self.config.flush_rows):
            return True
        if self._byte_count >= max(1, self.config.flush_bytes_mb) * 1024 * 1024:
            return True
        if self._batch_started_monotonic is None:
            return False
        return (time.monotonic() - self._batch_started_monotonic) >= self.config.flush_interval_sec

    def _release_items(self, items: List[ClaimedItem], *, reason: str) -> None:
        for item in items:
            self.queue_store.release_item(
                item.id,
                worker_id=self.writer_id,
                reason=reason,
            )

    def _clear(self) -> None:
        self._items.clear()
        self._rows.clear()
        self._byte_count = 0
        self._batch_started_monotonic = None

    def _cleanup_ack_result(self, ack_result: Any, *, reason: str) -> None:
        self._cleanup_refs(getattr(ack_result, "cleanup_payload_refs", []), reason=reason)

    def _cleanup_refs(self, refs: List[str], *, reason: str) -> None:
        if not self.cleanup_artifacts or self.artifact_store is None or not refs:
            return
        result = self.artifact_store.delete_refs(
            refs,
            prune_empty_parents=False,
        )
        for failure in result.failures:
            logger.warning(
                "artifact cleanup failed after parquet ack",
                extra={
                    "path": failure.path,
                    "error": failure.error,
                    "writer_id": self.writer_id,
                    "reason": reason,
                },
            )

    def _write_part(self, rows: List[Dict[str, Any]], final_path: Path) -> Path:
        if not rows:
            raise ValueError("cannot write empty parquet part")
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(f".{final_path.name}.{self.writer_id}.inprogress")
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, tmp_path)
            os.replace(tmp_path, final_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        return final_path

    def _next_part_path(self) -> Path:
        part_idx = self._next_part_index()
        return self.worker_output_dir / f"part-{part_idx:06d}.parquet"

    def _next_part_index(self) -> int:
        max_idx = 0
        for path in self.worker_output_dir.glob("part-*.parquet"):
            try:
                max_idx = max(max_idx, int(path.stem.split("-")[-1]))
            except ValueError:
                continue
        return max_idx + 1


def _estimate_rows_bytes(rows: List[Dict[str, Any]]) -> int:
    try:
        return len(json.dumps(rows, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(rows)


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
        total = 0.0
        for item in value.values():
            total += _duration_value_to_sec(item)
        return total
    if isinstance(value, (list, tuple)):
        return sum(_duration_value_to_sec(item) for item in value)
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else 0.0
    return 0.0


def _sum_rows_duration(rows: List[Dict[str, Any]]) -> Optional[float]:
    total = 0.0
    found = False
    for row in rows:
        for key in ("duration_sec", "duration", "target_duration", "generated_duration"):
            if key in row and row[key] is not None:
                total += _duration_value_to_sec(row[key])
                found = True
                break
        if {"source_duration_sec", "target_duration_sec"}.issubset(row):
            total += _duration_value_to_sec(row["source_duration_sec"])
            total += _duration_value_to_sec(row["target_duration_sec"])
            found = True
    return total if found else None
