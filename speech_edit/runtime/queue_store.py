"""PostgreSQL-backed durable queues and runtime state."""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, TYPE_CHECKING

import psycopg
from psycopg.rows import dict_row

from .models import ClaimedItem, OutputItem

if TYPE_CHECKING:
    from psycopg_pool import ConnectionPool
else:
    try:
        from psycopg_pool import ConnectionPool
    except ImportError:
        ConnectionPool = None


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS queue_items (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    queue_name TEXT NOT NULL,
    item_key TEXT NOT NULL,
    payload_ref TEXT,
    payload_json TEXT,
    status TEXT NOT NULL DEFAULT 'ready',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 1,
    lease_owner TEXT,
    lease_expires_at DOUBLE PRECISION,
    error TEXT,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    UNIQUE(run_id, queue_name, item_key)
);
CREATE INDEX IF NOT EXISTS idx_queue_claim
    ON queue_items(run_id, queue_name, status, created_at, id);
CREATE INDEX IF NOT EXISTS idx_queue_lease
    ON queue_items(run_id, status, lease_expires_at);

CREATE TABLE IF NOT EXISTS parquet_parts (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    output_path TEXT NOT NULL,
    writer_id TEXT NOT NULL,
    row_count INTEGER NOT NULL,
    byte_count INTEGER NOT NULL,
    duration_sec DOUBLE PRECISION,
    created_at DOUBLE PRECISION NOT NULL,
    UNIQUE(run_id, output_path)
);
CREATE INDEX IF NOT EXISTS idx_parquet_parts_run_created
    ON parquet_parts(run_id, created_at DESC, id DESC);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    run_id TEXT NOT NULL,
    worker_id TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    stage TEXT NOT NULL,
    pid INTEGER,
    status TEXT NOT NULL,
    first_hbeat_at DOUBLE PRECISION,
    updated_at DOUBLE PRECISION NOT NULL,
    metadata_json TEXT,
    PRIMARY KEY(run_id, worker_id)
);

CREATE TABLE IF NOT EXISTS runtime_run_state (
    run_id TEXT PRIMARY KEY,
    desired_state TEXT NOT NULL DEFAULT 'running',
    updated_at DOUBLE PRECISION NOT NULL,
    updated_by TEXT
);
"""


class QueueFull(RuntimeError):
    """Raised when enqueueing would exceed a logical queue capacity."""

    def __init__(self, queue_name: str, max_size: int, current_size: int, requested: int):
        super().__init__(
            f"queue {queue_name!r} is full: current={current_size}, "
            f"requested={requested}, max_size={max_size}"
        )
        self.queue_name = queue_name
        self.max_size = max_size
        self.current_size = current_size
        self.requested = requested


@dataclass(frozen=True)
class QueueAckResult:
    """Artifact refs made obsolete by a successful queue ack transaction."""

    input_payload_ref: Optional[str] = None
    discarded_payload_refs: List[str] = field(default_factory=list)

    @property
    def cleanup_payload_refs(self) -> List[str]:
        refs: List[str] = []
        if self.input_payload_ref:
            refs.append(self.input_payload_ref)
        refs.extend(self.discarded_payload_refs)
        return refs


class QueueStore:
    """One logical run in the shared PostgreSQL distributed-pipeline store."""

    def __init__(
        self,
        *,
        run_id: str | None = None,
        database_url: str | None = None,
        connection_pool: "ConnectionPool | None" = None,
        use_pool: bool = False,
        pool_min_size: int = 1,
        pool_max_size: int = 4,
        pool_timeout: float = 120.0,
        ensure_schema: bool = True,
    ):
        self.run_id = run_id or os.environ.get("DIST_PIPELINE_RUN_ID") or "default"
        self.database_url = (
            database_url
            or os.environ.get("DIST_PIPELINE_DATABASE_URL")
        )
        if not self.database_url:
            raise RuntimeError("DIST_PIPELINE_DATABASE_URL is required for QueueStore")
        self._pool = connection_pool
        self._owns_pool = False
        if self._pool is None and use_pool:
            if ConnectionPool is None:
                raise RuntimeError(
                    "psycopg-pool is required when QueueStore(use_pool=True)"
                )
            self._pool = ConnectionPool(
                conninfo=self.database_url,
                min_size=max(0, int(pool_min_size)),
                max_size=max(1, int(pool_max_size)),
                timeout=float(pool_timeout),
                kwargs={"row_factory": dict_row, "autocommit": False},
                open=True,
            )
            self._owns_pool = True
        if ensure_schema:
            self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[psycopg.Connection[Any]]:
        if self._pool is not None:
            with self._pool.connection() as conn:
                yield conn
            return
        conn = psycopg.connect(self.database_url, row_factory=dict_row, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[psycopg.Connection[Any]]:
        with self._connect() as conn:
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    def close(self) -> None:
        if self._owns_pool and self._pool is not None:
            self._pool.close()

    def _init_db(self) -> None:
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)

    @staticmethod
    def _now() -> float:
        return time.time()

    @staticmethod
    def _json_dumps(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, memoryview):
            value = value.tobytes()
        if isinstance(value, (bytes, bytearray)):
            return bytes(value).decode("utf-8")
        return str(value)

    def _pending_queue_size(self, conn: psycopg.Connection[Any], queue_name: str) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) AS n
                FROM queue_items
                WHERE run_id=%s AND queue_name=%s AND status IN ('ready', 'leased')
                """,
                (self.run_id, queue_name),
            )
            row = cur.fetchone()
        return int(row["n"]) if row is not None else 0

    def _check_queue_capacity(
        self,
        conn: psycopg.Connection[Any],
        *,
        queue_name: str,
        requested: int,
        max_size: Optional[int],
    ) -> None:
        if max_size is None or max_size <= 0 or requested <= 0:
            return
        current = self._pending_queue_size(conn, queue_name)
        if current + requested > int(max_size):
            raise QueueFull(queue_name, int(max_size), current, requested)

    def _check_output_capacity(
        self,
        conn: psycopg.Connection[Any],
        outputs: Sequence[OutputItem],
        queue_max_sizes: Optional[Mapping[str, Optional[int]]],
    ) -> None:
        if not queue_max_sizes:
            return
        requested_by_queue: Dict[str, int] = {}
        seen: set[tuple[str, str]] = set()
        with conn.cursor() as cur:
            for output in outputs:
                max_size = queue_max_sizes.get(output.queue_name)
                if max_size is None or max_size <= 0:
                    continue
                dedupe_key = (output.queue_name, output.item_key)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                cur.execute(
                    """
                    SELECT 1
                    FROM queue_items
                    WHERE run_id=%s AND queue_name=%s AND item_key=%s
                    LIMIT 1
                    """,
                    (self.run_id, output.queue_name, output.item_key),
                )
                if cur.fetchone() is not None:
                    continue
                requested_by_queue[output.queue_name] = requested_by_queue.get(output.queue_name, 0) + 1
        for queue_name, requested in requested_by_queue.items():
            self._check_queue_capacity(
                conn,
                queue_name=queue_name,
                requested=requested,
                max_size=queue_max_sizes.get(queue_name),
            )

    def enqueue(
        self,
        queue_name: str,
        item_key: str,
        *,
        payload_ref: Optional[str] = None,
        payload_json: Optional[Dict[str, Any]] = None,
        max_attempts: int = 1,
        queue_max_size: Optional[int] = None,
    ) -> bool:
        now = self._now()
        payload_json_text = self._json_dumps(payload_json) if payload_json is not None else None
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM queue_items
                    WHERE run_id=%s AND queue_name=%s AND item_key=%s
                    LIMIT 1
                    """,
                    (self.run_id, queue_name, item_key),
                )
                if cur.fetchone() is not None:
                    return False
                self._check_queue_capacity(
                    conn,
                    queue_name=queue_name,
                    requested=1,
                    max_size=queue_max_size,
                )
                cur.execute(
                    """
                    INSERT INTO queue_items (
                        run_id, queue_name, item_key, payload_ref, payload_json,
                        max_attempts, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, queue_name, item_key) DO NOTHING
                    """,
                    (
                        self.run_id,
                        queue_name,
                        item_key,
                        payload_ref,
                        payload_json_text,
                        max_attempts,
                        now,
                        now,
                    ),
                )
                return cur.rowcount > 0

    def enqueue_json_many(
        self,
        queue_name: str,
        items: Sequence[Mapping[str, Any]],
        *,
        max_attempts: int = 1,
        queue_max_size: Optional[int] = None,
    ) -> int:
        if not items:
            return 0
        now = self._now()
        rows = [
            (
                self.run_id,
                queue_name,
                str(item["item_key"]),
                self._json_dumps(item.get("payload")) if item.get("payload") is not None else None,
                max_attempts,
                now,
                now,
            )
            for item in items
        ]
        inserted = 0
        with self.transaction() as conn:
            if queue_max_size is not None and queue_max_size > 0:
                missing = 0
                with conn.cursor() as cur:
                    for _, _, item_key, *_rest in rows:
                        cur.execute(
                            """
                            SELECT 1
                            FROM queue_items
                            WHERE run_id=%s AND queue_name=%s AND item_key=%s
                            LIMIT 1
                            """,
                            (self.run_id, queue_name, item_key),
                        )
                        missing += int(cur.fetchone() is None)
                self._check_queue_capacity(
                    conn,
                    queue_name=queue_name,
                    requested=missing,
                    max_size=queue_max_size,
                )
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO queue_items (
                        run_id, queue_name, item_key, payload_json, max_attempts,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, queue_name, item_key) DO NOTHING
                    """,
                    rows,
                    returning=True,
                )
                while True:
                    inserted += cur.rowcount if cur.rowcount > 0 else 0
                    if not cur.nextset():
                        break
        return inserted


    def claim(self, queue_name: str, worker_id: str, lease_seconds: float) -> Optional[ClaimedItem]:
        now = self._now()
        lease_expires_at = now + lease_seconds
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH candidate AS (
                        SELECT id
                        FROM queue_items
                        WHERE run_id=%s
                          AND queue_name=%s
                          AND (
                              status='ready'
                              OR (status='leased' AND lease_expires_at <= %s)
                          )
                        ORDER BY created_at, id
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                    )
                    UPDATE queue_items AS q
                    SET status='leased',
                        attempts=attempts + 1,
                        lease_owner=%s,
                        lease_expires_at=%s,
                        updated_at=%s
                    FROM candidate
                    WHERE q.id=candidate.id
                    RETURNING q.*
                    """,
                    (self.run_id, queue_name, now, worker_id, lease_expires_at, now),
                )
                row = cur.fetchone()
        if row is None:
            return None
        return ClaimedItem(
            id=int(row["id"]),
            queue_name=self._text(row["queue_name"]) or "",
            item_key=self._text(row["item_key"]) or "",
            payload_ref=self._text(row["payload_ref"]),
            payload_json=self._text(row["payload_json"]),
            attempts=int(row["attempts"]),
        )

    def ack_and_enqueue(
        self,
        item_id: int,
        *,
        worker_id: str,
        outputs: Sequence[OutputItem],
        payload_refs: Sequence[str],
        queue_max_sizes: Optional[Mapping[str, Optional[int]]] = None,
        clear_input_payload_ref: bool = False,
    ) -> QueueAckResult:
        if len(outputs) != len(payload_refs):
            raise ValueError("outputs and payload_refs length mismatch")
        now = self._now()
        input_payload_ref: Optional[str] = None
        discarded_payload_refs: List[str] = []
        with self.transaction() as conn:
            self._check_output_capacity(conn, outputs, queue_max_sizes)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT payload_ref
                    FROM queue_items
                    WHERE run_id=%s AND id=%s AND status='leased' AND lease_owner=%s
                    FOR UPDATE
                    """,
                    (self.run_id, item_id, worker_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError(f"queue item {item_id} is not leased by {worker_id}")
                input_payload_ref = self._text(row["payload_ref"]) if clear_input_payload_ref else None
                cur.execute(
                    """
                    UPDATE queue_items
                    SET status='done',
                        payload_ref=CASE WHEN %s THEN NULL ELSE payload_ref END,
                        lease_owner=NULL,
                        lease_expires_at=NULL,
                        updated_at=%s
                    WHERE run_id=%s AND id=%s AND status='leased' AND lease_owner=%s
                    """,
                    (clear_input_payload_ref, now, self.run_id, item_id, worker_id),
                )
                if cur.rowcount != 1:
                    raise RuntimeError(f"queue item {item_id} is not leased by {worker_id}")
                for output, payload_ref in zip(outputs, payload_refs):
                    cur.execute(
                        """
                        INSERT INTO queue_items (
                            run_id, queue_name, item_key, payload_ref, max_attempts,
                            created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (run_id, queue_name, item_key) DO NOTHING
                        """,
                        (
                            self.run_id,
                            output.queue_name,
                            output.item_key,
                            payload_ref,
                            output.max_attempts,
                            now,
                            now,
                        ),
                    )
                    if cur.rowcount == 0:
                        discarded_payload_refs.append(payload_ref)
        return QueueAckResult(
            input_payload_ref=input_payload_ref,
            discarded_payload_refs=discarded_payload_refs,
        )

    def artifact_payload_ref_summary(self, *, statuses: Sequence[str]) -> List[Dict[str, Any]]:
        if not statuses:
            raise ValueError("statuses must not be empty")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT queue_name,
                           status,
                           COUNT(*) AS item_count,
                           COALESCE(SUM((payload_ref::jsonb ->> 'size_bytes')::bigint), 0) AS size_bytes
                    FROM queue_items
                    WHERE run_id=%s
                      AND status = ANY(%s)
                      AND payload_ref IS NOT NULL
                    GROUP BY queue_name, status
                    ORDER BY queue_name, status
                    """,
                    (self.run_id, list(statuses)),
                )
                rows = cur.fetchall()
        return [
            {
                "queue_name": self._text(row["queue_name"]) or "",
                "status": self._text(row["status"]) or "",
                "item_count": int(row["item_count"]),
                "size_bytes": int(row["size_bytes"] or 0),
            }
            for row in rows
        ]

    def artifact_payload_ref_batch(
        self,
        *,
        statuses: Sequence[str],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if not statuses:
            raise ValueError("statuses must not be empty")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, queue_name, item_key, status, payload_ref
                    FROM queue_items
                    WHERE run_id=%s
                      AND status = ANY(%s)
                      AND payload_ref IS NOT NULL
                    ORDER BY id
                    LIMIT %s
                    """,
                    (self.run_id, list(statuses), max(1, int(limit))),
                )
                rows = cur.fetchall()
        return [
            {
                "id": int(row["id"]),
                "queue_name": self._text(row["queue_name"]) or "",
                "item_key": self._text(row["item_key"]) or "",
                "status": self._text(row["status"]) or "",
                "payload_ref": self._text(row["payload_ref"]) or "",
            }
            for row in rows
        ]

    def clear_artifact_payload_refs(
        self,
        *,
        item_ids: Sequence[int],
        statuses: Sequence[str],
    ) -> int:
        if not item_ids:
            return 0
        if not statuses:
            raise ValueError("statuses must not be empty")
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET payload_ref=NULL, updated_at=%s
                    WHERE run_id=%s
                      AND id = ANY(%s)
                      AND status = ANY(%s)
                      AND payload_ref IS NOT NULL
                    """,
                    (now, self.run_id, [int(item_id) for item_id in item_ids], list(statuses)),
                )
                return int(cur.rowcount)

    def record_parquet_part(
        self,
        *,
        output_path: str,
        writer_id: str,
        row_count: int,
        byte_count: int,
        duration_sec: Optional[float] = None,
    ) -> None:
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO parquet_parts (
                        run_id, output_path, writer_id, row_count, byte_count,
                        duration_sec, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, output_path) DO UPDATE SET
                        writer_id=excluded.writer_id,
                        row_count=excluded.row_count,
                        byte_count=excluded.byte_count,
                        duration_sec=excluded.duration_sec
                    """,
                    (
                        self.run_id,
                        output_path,
                        writer_id,
                        int(row_count),
                        int(byte_count),
                        duration_sec,
                        now,
                    ),
                )

    def release_item(self, item_id: int, *, worker_id: str, reason: str = "") -> None:
        """Return a leased queue item to ready without consuming an attempt."""
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET status='ready',
                        attempts=GREATEST(attempts - 1, 0),
                        lease_owner=NULL,
                        lease_expires_at=NULL,
                        error=%s,
                        updated_at=%s
                    WHERE run_id=%s AND id=%s AND status='leased' AND lease_owner=%s
                    """,
                    (reason[-4000:] if reason else None, now, self.run_id, item_id, worker_id),
                )


    def item_is_leased_by(self, item_id: int, *, worker_id: str) -> bool:
        now = self._now()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT 1
                    FROM queue_items
                    WHERE run_id=%s
                      AND id=%s
                      AND status='leased'
                      AND lease_owner=%s
                      AND (lease_expires_at IS NULL OR lease_expires_at > %s)
                    LIMIT 1
                    """,
                    (self.run_id, item_id, worker_id, now),
                )
                return cur.fetchone() is not None

    def fail_item(self, item_id: int, *, worker_id: str, error: str) -> None:
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT attempts, max_attempts FROM queue_items WHERE run_id=%s AND id=%s",
                    (self.run_id, item_id),
                )
                row = cur.fetchone()
                if row is None:
                    return
                status = "dead" if int(row["attempts"]) >= int(row["max_attempts"]) else "ready"
                cur.execute(
                    """
                    UPDATE queue_items
                    SET status=%s, lease_owner=NULL, lease_expires_at=NULL, error=%s, updated_at=%s
                    WHERE run_id=%s AND id=%s AND status='leased' AND lease_owner=%s
                    """,
                    (status, error[-4000:], now, self.run_id, item_id, worker_id),
                )

    def reclaim_expired_leases(self) -> None:
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE queue_items
                    SET status='ready', lease_owner=NULL, lease_expires_at=NULL, updated_at=%s
                    WHERE run_id=%s AND status='leased' AND lease_expires_at <= %s
                    """,
                    (now, self.run_id, now),
                )

    def queue_stats(self) -> Dict[str, Dict[str, int]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT queue_name, status, COUNT(*) AS n
                    FROM queue_items
                    WHERE run_id=%s
                    GROUP BY queue_name, status
                    """,
                    (self.run_id,),
                )
                rows = cur.fetchall()
        stats: Dict[str, Dict[str, int]] = {}
        for row in rows:
            queue_name = self._text(row["queue_name"]) or ""
            status = self._text(row["status"]) or ""
            stats.setdefault(queue_name, {})[status] = int(row["n"])
        return stats


    def has_queue_capacity(self, queue_name: str, *, requested: int, max_size: Optional[int]) -> bool:
        if max_size is None or max_size <= 0:
            return True
        with self._connect() as conn:
            return self._pending_queue_size(conn, queue_name) + requested <= int(max_size)


    def create_run(self) -> None:
        """Reserve a run identity before seeding, including across concurrent submitters."""
        with self.transaction() as conn:
            row = conn.execute(
                "INSERT INTO runtime_run_state (run_id, desired_state, updated_at) "
                "VALUES (%s, 'preparing', %s) ON CONFLICT DO NOTHING RETURNING run_id",
                (self.run_id, self._now()),
            ).fetchone()
            if row is None:
                raise ValueError(f"run_id already exists: {self.run_id}")

    def set_run_state(
        self,
        *,
        desired_state: str,
        updated_by: Optional[str] = None,
    ) -> None:
        now = self._now()
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO runtime_run_state (
                        run_id, desired_state, updated_at, updated_by
                    ) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        desired_state=excluded.desired_state,
                        updated_at=excluded.updated_at,
                        updated_by=excluded.updated_by
                    """,
                    (self.run_id, desired_state, now, updated_by),
                )

    def get_run_state(self) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT desired_state, updated_at, updated_by
                    FROM runtime_run_state
                    WHERE run_id=%s
                    """,
                    (self.run_id,),
                )
                row = cur.fetchone()
        if row is None:
            return {
                "desired_state": "running",
                "updated_at": None,
                "updated_by": None,
            }
        return {
            "desired_state": self._text(row["desired_state"]) or "running",
            "updated_at": float(row["updated_at"]),
            "updated_by": self._text(row["updated_by"]),
        }


    def completion_stats(self) -> Dict[str, int]:
        self.reclaim_expired_leases()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT status, COUNT(*) AS n
                    FROM queue_items
                    WHERE run_id=%s
                    GROUP BY status
                    """,
                    (self.run_id,),
                )
                queue_rows = cur.fetchall()
        stats: Dict[str, int] = {}
        queue_total = 0
        for row in queue_rows:
            count = int(row["n"])
            queue_total += count
            stats[f"queue_{self._text(row['status']) or ''}"] = count
        stats["queue_total"] = queue_total
        stats["run_complete"] = int(
            stats.get("queue_total", 0) > 0
            and stats.get("queue_ready", 0) == 0
            and stats.get("queue_leased", 0) == 0
            and stats.get("queue_dead", 0) == 0
        )
        return stats

    def heartbeat(
        self,
        *,
        worker_id: str,
        pipeline: str,
        stage: str,
        pid: Optional[int],
        status: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = self._now()
        metadata_json = self._json_dumps(metadata) if metadata is not None else None
        with self.transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO worker_heartbeats (
                        run_id, worker_id, pipeline, stage, pid, status,
                        first_hbeat_at, updated_at, metadata_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, worker_id) DO UPDATE SET
                        pipeline=excluded.pipeline,
                        stage=excluded.stage,
                        pid=excluded.pid,
                        status=excluded.status,
                        first_hbeat_at=COALESCE(worker_heartbeats.first_hbeat_at, excluded.first_hbeat_at),
                        updated_at=excluded.updated_at,
                        metadata_json=excluded.metadata_json
                    """,
                    (self.run_id, worker_id, pipeline, stage, pid, status, now, now, metadata_json),
                )
