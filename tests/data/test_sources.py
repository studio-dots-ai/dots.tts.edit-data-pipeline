from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from speech_edit.data.sources import resolve_parquet_sources


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_resolve_parquet_sources_accepts_directories_and_skips_inprogress(tmp_path: Path):
    first = tmp_path / "run" / "parquet" / "part-000.parquet"
    nested = tmp_path / "run" / "parquet" / "worker_1" / "part-001.parquet"
    inprogress = tmp_path / "run" / "parquet" / ".part-002.parquet.worker.inprogress"
    _write_parquet(first, [{"x": 1}])
    _write_parquet(nested, [{"x": 2}])
    _write_parquet(inprogress, [{"x": 3}])

    resolved = resolve_parquet_sources([tmp_path / "run" / "parquet"])

    assert resolved == [first, nested]


@pytest.mark.parametrize("suffix", [".list", ".LIST"])
def test_parquet_sources_list_file_expands(tmp_path: Path, suffix: str):
    shard = tmp_path / "part.parquet"
    _write_parquet(shard, [{"x": 1}])
    manifest = tmp_path / f"source{suffix}"
    manifest.write_text(f"{shard.name}\n")
    assert resolve_parquet_sources([manifest]) == [shard]


def test_blank_and_comment_lines_ignored(tmp_path: Path):
    shard = tmp_path / "part.parquet"
    _write_parquet(shard, [{"x": 1}])
    manifest = tmp_path / "source.list"
    manifest.write_text(f"\n# comment\n{shard}\n\n  # another comment\n")
    assert resolve_parquet_sources([manifest]) == [shard]


def test_missing_parquet_referenced_in_list_raises(tmp_path: Path):
    manifest = tmp_path / "source.list"
    manifest.write_text("missing.parquet\n")
    with pytest.raises(FileNotFoundError, match="do not exist"):
        resolve_parquet_sources([manifest])


def test_recursive_list_expands_nested_manifests(tmp_path: Path):
    shard = tmp_path / "nested" / "part.parquet"
    _write_parquet(shard, [{"x": 1}])
    nested = shard.parent / "source.list"
    nested.write_text(f"{shard.name}\n")
    root = tmp_path / "root.list"
    root.write_text("nested/source.list\n")
    assert resolve_parquet_sources([root]) == [shard]


def test_recursive_list_cycle_raises(tmp_path: Path):
    first, second = tmp_path / "first.list", tmp_path / "second.list"
    first.write_text(f"{second.name}\n")
    second.write_text(f"{first.name}\n")
    with pytest.raises(ValueError, match="cycle"):
        resolve_parquet_sources([first])
