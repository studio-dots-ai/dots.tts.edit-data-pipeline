"""In-process Qwen3-ASR Transformers backend for stability-first evaluation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from backends.qwen3_asr import (
    ASRResult,
    _forced_language_name,
)


def _load_qwen_model(
    *,
    model_path: str,
    device: str,
    dtype_name: str,
    max_new_tokens: int,
) -> Any:
    import torch
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "qwen3-asr"))
    import qwen3_asr_service  # Activate the pinned official checkout.
    from qwen_asr import Qwen3ASRModel

    dtype = getattr(torch, dtype_name)
    return Qwen3ASRModel.from_pretrained(
        _resolve_local_model_path(model_path),
        dtype=dtype,
        device_map=device,
        max_inference_batch_size=1,
        max_new_tokens=max_new_tokens,
    )


def _resolve_local_model_path(model_path: str) -> str:
    direct = Path(model_path).expanduser()
    if direct.exists():
        return str(direct.resolve())
    if "/" not in model_path:
        return model_path

    cache_name = f"models--{model_path.replace('/', '--')}"
    roots = [
        os.environ.get("HF_HUB_CACHE"),
        os.environ.get("CHECKPOINTS_DIR"),
    ]
    for root_value in roots:
        if not root_value:
            continue
        repo_cache = (Path(root_value).expanduser() / cache_name).resolve()
        if not repo_cache.is_dir():
            continue
        try:
            revision = (repo_cache / "refs" / "main").read_text(
                encoding="utf-8"
            ).strip()
        except OSError:
            snapshots = sorted((repo_cache / "snapshots").glob("*"))
            revision = snapshots[-1].name if snapshots else ""
        snapshot = repo_cache / "snapshots" / revision
        if revision and (snapshot / "config.json").is_file():
            return str(snapshot)
    return model_path


def _is_exact_silence(audio: Union[str, Tuple[np.ndarray, int]]) -> bool:
    """Return whether a local or in-memory waveform contains only zero PCM."""
    try:
        if isinstance(audio, str):
            if audio.startswith(("http://", "https://", "data:")):
                return False
            waveform, _sample_rate = sf.read(
                audio,
                dtype="float32",
                always_2d=True,
            )
        else:
            waveform, _sample_rate = audio
            waveform = np.asarray(waveform)
    except (OSError, RuntimeError, ValueError):
        return False
    return bool(waveform.size and np.all(waveform == 0.0))


class Qwen3ASRTransformersBackend:
    """Run Qwen3-ASR directly with Transformers, one request at a time."""

    def __init__(
        self,
        *,
        model_path: str,
        backend_name: str = "qwen3-asr",
        device: str = "cuda:0",
        dtype_name: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> None:
        self._backend_name = backend_name
        self._model_path = model_path
        self.device = device
        self.dtype_name = dtype_name
        self.max_new_tokens = int(max_new_tokens)
        self._model: Any = None

    def load(self) -> None:
        if self._model is None:
            self._model = _load_qwen_model(
                model_path=self._model_path,
                device=self.device,
                dtype_name=self.dtype_name,
                max_new_tokens=self.max_new_tokens,
            )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def backend_name(self) -> str:
        return self._backend_name

    @property
    def inference_name(self) -> str:
        return "transformers"

    @property
    def model_path(self) -> str:
        return self._model_path

    def transcribe_batch(
        self,
        audios: List[Union[str, Tuple[np.ndarray, int]]],
        languages: Optional[List[Optional[str]]] = None,
    ) -> List[ASRResult]:
        if self._model is None:
            raise RuntimeError("Qwen3-ASR Transformers backend is not loaded")
        if languages is None:
            languages = [None] * len(audios)
        if len(languages) != len(audios):
            raise ValueError(
                f"Batch size mismatch: audio={len(audios)}, language={len(languages)}"
            )

        results: List[ASRResult] = []
        for audio, language in zip(audios, languages):
            forced_language = _forced_language_name(language)
            if _is_exact_silence(audio):
                results.append(
                    ASRResult(text="", language=forced_language or language)
                )
                continue
            transcription = self._model.transcribe(
                audio=audio,
                language=forced_language,
            )
            if len(transcription) != 1:
                raise RuntimeError(
                    "Qwen3-ASR Transformers backend returned "
                    f"{len(transcription)} results for one input"
                )
            item = transcription[0]
            results.append(
                ASRResult(
                    text=str(item.text),
                    language=str(item.language or forced_language or language or ""),
                )
            )
        return results

    def transcribe_single(
        self,
        audio: Union[str, Tuple[np.ndarray, int]],
        language: Optional[str] = None,
    ) -> ASRResult:
        return self.transcribe_batch([audio], [language])[0]
