"""Helpers for resolving parquet sources from recursive manifest files."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence


def is_parquet_list_path(path: Path) -> bool:
    """Return True when the path uses the .list extension."""
    return path.suffix.lower() == ".list"


def is_parquet_path(path: Path) -> bool:
    """Return True when the path uses the .parquet extension."""
    return path.suffix.lower() == ".parquet"


def _is_inprogress_path(path: Path) -> bool:
    return any(part.endswith(".inprogress") or ".inprogress" in part for part in path.parts)


def _read_list_entries(list_path: Path) -> List[Path]:
    """Read one manifest and return non-empty, non-comment entries."""
    entries: List[Path] = []
    with list_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            entry = Path(line)
            if not entry.is_absolute():
                entry = list_path.parent / entry
            entries.append(entry)

    if not entries:
        raise ValueError(f"Parquet list file is empty: {list_path}")

    return entries


def resolve_parquet_sources(parquet_sources: Sequence[str | Path]) -> List[Path]:
    """Resolve parquet files from .parquet, directory, and nested .list sources."""
    if not parquet_sources:
        raise ValueError("parquet_sources must contain at least one path")

    resolved_files: List[Path] = []
    seen_files: set[Path] = set()

    def _expand(source: Path, stack: tuple[Path, ...]) -> None:
        if is_parquet_list_path(source):
            if not source.is_file():
                raise FileNotFoundError(f"Parquet source does not exist: {source}")
            if source in stack:
                cycle = " -> ".join(str(path) for path in (*stack, source))
                raise ValueError(f"Recursive parquet list cycle detected: {cycle}")

            for entry in _read_list_entries(source):
                _expand(entry, (*stack, source))
            return

        if source.is_dir():
            for entry in sorted(source.rglob("*.parquet")):
                if not _is_inprogress_path(entry):
                    _expand(entry, stack)
            return

        if not is_parquet_path(source):
            raise ValueError(
                f"Unsupported parquet source path: {source} (expected .parquet, .list, or directory)"
            )
        if not source.is_file():
            raise FileNotFoundError(f"Parquet files do not exist: {source}")
        if _is_inprogress_path(source):
            return
        if source in seen_files:
            return

        seen_files.add(source)
        resolved_files.append(source)

    for source in parquet_sources:
        _expand(Path(source), ())

    if not resolved_files:
        raise ValueError("No parquet files were resolved from parquet_sources")

    return resolved_files


def resolve_parquet_source(parquet_source: str | Path) -> List[Path]:
    """Resolve one parquet source path into a flat list of parquet files."""
    return resolve_parquet_sources([parquet_source])
