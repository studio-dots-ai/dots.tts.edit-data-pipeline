"""Atomic artifact storage for queue payloads."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a blob stored below an ArtifactStore root."""

    path: str
    sha256: str
    size_bytes: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_json(cls, payload: str | Dict[str, Any]) -> "ArtifactRef":
        data = json.loads(payload) if isinstance(payload, str) else payload
        return cls(
            path=str(data["path"]),
            sha256=str(data["sha256"]),
            size_bytes=int(data["size_bytes"]),
        )


@dataclass(frozen=True)
class ArtifactDeleteFailure:
    """One artifact deletion failure."""

    path: str
    error: str


@dataclass(frozen=True)
class ArtifactDeleteResult:
    """Summary returned by best-effort artifact deletion."""

    deleted_count: int = 0
    bytes_freed: int = 0
    failures: List[ArtifactDeleteFailure] = field(default_factory=list)

    def merge(self, other: "ArtifactDeleteResult") -> "ArtifactDeleteResult":
        return ArtifactDeleteResult(
            deleted_count=self.deleted_count + other.deleted_count,
            bytes_freed=self.bytes_freed + other.bytes_freed,
            failures=[*self.failures, *other.failures],
        )


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.=-]+")


def safe_name(value: str, *, max_len: int = 96) -> str:
    cleaned = _SAFE_NAME_RE.sub("_", value.strip())
    cleaned = cleaned.strip("._") or "item"
    if len(cleaned) <= max_len:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{cleaned[: max_len - 13]}_{digest}"


class ArtifactStore:
    """Content-addressed-ish blob store rooted outside final sample output."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, ref: ArtifactRef | str | Dict[str, Any]) -> Path:
        artifact_ref = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_json(ref)
        path = self.root / artifact_ref.path
        resolved = path.resolve()
        root_resolved = self.root.resolve()
        if root_resolved not in (resolved, *resolved.parents):
            raise ValueError(f"artifact path escapes root: {artifact_ref.path}")
        return resolved

    def _write_bytes(self, relpath: str, payload: bytes) -> ArtifactRef:
        target = self.root / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
        digest = hashlib.sha256(payload).hexdigest()
        return ArtifactRef(
            path=relpath,
            sha256=digest,
            size_bytes=len(payload),
        )

    def write_pickle(self, kind: str, key: str, payload: Any) -> ArtifactRef:
        relpath = f"{safe_name(kind)}/{safe_name(key)}.pkl"
        return self._write_bytes(relpath, pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))

    def read_pickle(self, ref: ArtifactRef | str | Dict[str, Any]) -> Any:
        artifact_ref = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_json(ref)
        payload = self.resolve(artifact_ref).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact_ref.sha256:
            raise ValueError(f"artifact checksum mismatch: {artifact_ref.path}")
        return pickle.loads(payload)

    def write_json(self, kind: str, key: str, payload: Any) -> ArtifactRef:
        relpath = f"{safe_name(kind)}/{safe_name(key)}.json"
        data = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return self._write_bytes(relpath, data)

    def read_json(self, ref: ArtifactRef | str | Dict[str, Any]) -> Any:
        artifact_ref = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_json(ref)
        payload = self.resolve(artifact_ref).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != artifact_ref.sha256:
            raise ValueError(f"artifact checksum mismatch: {artifact_ref.path}")
        return json.loads(payload.decode("utf-8"))

    def delete_ref(
        self,
        ref: ArtifactRef | str | Dict[str, Any],
        *,
        prune_empty_parents: bool = True,
    ) -> ArtifactDeleteResult:
        """Delete one artifact below this store root.

        Missing files are treated as already deleted. Invalid references and IO
        errors are reported in the result instead of being raised, so callers can
        keep queue ack semantics independent from best-effort filesystem cleanup.
        """

        try:
            artifact_ref = ref if isinstance(ref, ArtifactRef) else ArtifactRef.from_json(ref)
            path = self.resolve(artifact_ref)
        except Exception as exc:  # noqa: BLE001 - return a structured cleanup failure.
            raw_path = _raw_ref_path(ref)
            return ArtifactDeleteResult(
                failures=[ArtifactDeleteFailure(path=raw_path, error=str(exc))]
            )

        try:
            if not path.exists():
                return ArtifactDeleteResult()
            size = int(path.stat().st_size)
            path.unlink()
            if prune_empty_parents:
                self._remove_empty_parents(path.parent)
            return ArtifactDeleteResult(deleted_count=1, bytes_freed=size)
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort.
            return ArtifactDeleteResult(
                failures=[ArtifactDeleteFailure(path=artifact_ref.path, error=str(exc))]
            )

    def delete_refs(
        self,
        refs: List[ArtifactRef | str | Dict[str, Any]],
        *,
        prune_empty_parents: bool = True,
    ) -> ArtifactDeleteResult:
        result = ArtifactDeleteResult()
        for ref in refs:
            result = result.merge(
                self.delete_ref(ref, prune_empty_parents=prune_empty_parents)
            )
        return result

    def _remove_empty_parents(self, start: Path) -> None:
        root_resolved = self.root.resolve()
        current = start.resolve()
        while current != root_resolved and root_resolved in current.parents:
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent


def _raw_ref_path(ref: ArtifactRef | str | Dict[str, Any]) -> str:
    if isinstance(ref, ArtifactRef):
        return ref.path
    if isinstance(ref, dict):
        return str(ref.get("path", ""))
    if isinstance(ref, str):
        try:
            data = json.loads(ref)
            if isinstance(data, dict):
                return str(data.get("path", ""))
        except json.JSONDecodeError:
            pass
        return ref
    return ""
