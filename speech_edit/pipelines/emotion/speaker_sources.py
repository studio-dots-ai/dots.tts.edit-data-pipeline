"""Speaker parquet source discovery for emotion/style edit sampling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from speech_edit.data.sampler import ParquetListSampler


logger = logging.getLogger(__name__)

SPEAKER_LIBRARY_MANIFEST_NAME = "parquet_files.list"


def _infer_language_from_source_path(path: Path) -> Optional[str]:
    for part in path.parts:
        if part in {"zh", "en"}:
            return part
    return None


def _manifest_has_entries(path: Path) -> bool:
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line and not line.startswith("#"):
                return True
    return False


def _discover_language_manifests(root_dir: Path) -> Dict[str, List[str]]:
    grouped_sources: Dict[str, List[str]] = {}
    for language in ("zh", "en"):
        language_dir = root_dir / language
        if not language_dir.is_dir():
            continue

        manifest_path = language_dir / SPEAKER_LIBRARY_MANIFEST_NAME
        if manifest_path.is_file() and _manifest_has_entries(manifest_path):
            grouped_sources[language] = [str(manifest_path)]
            continue

        list_files = [
            path for path in sorted(language_dir.glob("*.list")) if _manifest_has_entries(path)
        ]
        if list_files:
            grouped_sources[language] = [str(path) for path in list_files]
            continue

        parquet_files = sorted(language_dir.glob("**/*.parquet"))
        if parquet_files:
            grouped_sources[language] = [str(path) for path in parquet_files]
    return grouped_sources


def resolve_language_speaker_parquet_sources(
    speaker_parquet_sources: list[str],
) -> Dict[str, List[str]]:
    grouped_sources: Dict[str, List[str]] = {}
    for raw_source in speaker_parquet_sources:
        source_path = Path(raw_source)
        if source_path.is_dir():
            discovered = _discover_language_manifests(source_path)
            if not discovered:
                raise FileNotFoundError(
                    f"No zh/en speaker parquet manifests found under {source_path}"
                )
            for language, paths in discovered.items():
                grouped_sources.setdefault(language, []).extend(paths)
            continue

        language = _infer_language_from_source_path(source_path)
        if language is None:
            raise ValueError(
                "Speaker parquet sources must either be the speaker_timbre_library "
                "root directory or a path containing /zh/ or /en/"
            )
        if source_path.suffix.lower() == ".list" and not _manifest_has_entries(source_path):
            continue
        grouped_sources.setdefault(language, []).append(str(source_path))

    if not grouped_sources:
        raise FileNotFoundError(
            "No non-empty zh/en speaker parquet manifests were found in speaker_parquet_sources"
        )
    return grouped_sources


def initialize_language_speaker_samplers(
    speaker_parquet_sources: Optional[list[str]],
    *,
    seed: Optional[int],
    log_prefix: str = "",
) -> Optional[Dict[str, "ParquetListSampler"]]:
    if not speaker_parquet_sources:
        return None

    from speech_edit.data.sampler import ParquetListSampler

    grouped_sources = resolve_language_speaker_parquet_sources(speaker_parquet_sources)
    speaker_samplers: Dict[str, ParquetListSampler] = {}
    for language, sources in sorted(grouped_sources.items()):
        logger.info(
            "%sInitializing %s speaker sampler from %d parquet source(s)",
            log_prefix,
            language,
            len(sources),
        )
        speaker_samplers[language] = ParquetListSampler(
            parquet_sources=sources,
            min_single_duration=2.0,
            max_single_duration=30.0,
            seed=seed,
        )
    return speaker_samplers or None
