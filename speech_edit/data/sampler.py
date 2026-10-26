"""Cyclic, single-segment sampling from the emotion speaker prompt pool."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Sequence

import pandas as pd

from .parquet.reader import Segment, row_to_segment
from .sources import resolve_parquet_sources


class ParquetListSampler:
    """Read one parquet at a time and reshuffle the prompt pool each epoch."""

    def __init__(
        self,
        *,
        parquet_sources: Sequence[str | Path],
        min_single_duration: float,
        max_single_duration: float,
        seed: int | None,
    ):
        self.parquet_files = [
            path.expanduser().resolve()
            for path in resolve_parquet_sources(parquet_sources)
        ]
        if not self.parquet_files:
            raise FileNotFoundError("Speaker prompt pool has no parquet files")
        self.min_single_duration = min_single_duration
        self.max_single_duration = max_single_duration
        self.seed = seed
        self._epoch = 0
        self._segments = self._read_epoch()

    def _read_epoch(self) -> Iterator[Segment]:
        rng = random.Random(
            self.seed * 131 + self._epoch * 997 if self.seed is not None else None
        )
        file_indices = list(range(len(self.parquet_files)))
        rng.shuffle(file_indices)
        for file_index in file_indices:
            path = self.parquet_files[file_index]
            frame = pd.read_parquet(path)
            row_indices = list(range(len(frame)))
            rng.shuffle(row_indices)
            for row_index in row_indices:
                segment = row_to_segment(
                    frame.iloc[row_index], row_index, path,
                    source_file_index=file_index,
                    min_single_duration=self.min_single_duration,
                    max_single_duration=self.max_single_duration,
                )
                if segment is not None:
                    yield segment

    def sample(self) -> Segment:
        segment = next(self._segments, None)
        if segment is None:
            self._epoch += 1
            self._segments = self._read_epoch()
            segment = next(self._segments, None)
        if segment is None:
            raise RuntimeError("Speaker prompt pool has no segments within the duration bounds")
        return segment
