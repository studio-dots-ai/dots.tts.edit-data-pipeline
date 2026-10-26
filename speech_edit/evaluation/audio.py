"""Audio quality and content metrics shared by editing pipelines."""

import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, Optional

from speech_edit.data.language import asr_language_name

logger = logging.getLogger(__name__)

# SpkSim service uses a non-thread-safe GPU model (ECAPA-TDNN) that
# corrupts tensors when processing concurrent requests.  Serialise all
# SpkSim calls within a single pipeline worker via this module-level lock.
_spk_sim_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EvalMetrics:
    """Evaluation metrics for a single audio sample."""

    # Content accuracy
    asr_text: str = ""
    wer: float = -1.0  # -1 means not computed
    wer_pct: float = -1.0
    wer_backend: str = "not_computed"
    wer_metric: str = "wer"

    # Audio quality
    utmos_score: float = -1.0  # -1 means not computed

    # Speaker similarity (source vs edited)
    spk_similarity: float = -1.0  # -1 means not computed

    # Timing
    eval_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"eval_time_sec": self.eval_time_sec}
        if self.asr_text or self.wer >= 0:
            payload["transcript"] = self.asr_text
            payload["wer"] = {
                "error_rate": self.wer,
                "metric": self.wer_metric,
                "backend": self.wer_backend,
            }
        if self.utmos_score >= 0:
            payload["utmos"] = {"score": self.utmos_score}
        return payload


@dataclass
class PairEvalResult:
    """Evaluation results for an ordered source-target pair."""

    source_metrics: EvalMetrics  # metrics on the source audio
    target_metrics: EvalMetrics  # metrics on the ordered-pair target audio
    eval_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "src": self.source_metrics.to_dict(),
            "tgt": self.target_metrics.to_dict(),
            "eval_time_sec": self.eval_time_sec,
        }
        if self.target_metrics.spk_similarity >= 0:
            payload["spk_sim"] = {"similarity": self.target_metrics.spk_similarity}
        return payload


# ---------------------------------------------------------------------------
# WER computation
# ---------------------------------------------------------------------------


def _asr_language(sample_language: Optional[str], reference_text: Optional[str] = None) -> str:
    return asr_language_name(sample_language, text=reference_text) or "English"


def _wer_pct(wer: float) -> float:
    if wer < 0 or not math.isfinite(wer):
        return -1.0
    return round(wer * 100.0, 2)


def delta_metrics(
    source_metrics: EvalMetrics,
    target_metrics: EvalMetrics,
) -> Dict[str, Any]:
    if source_metrics.wer < 0 or target_metrics.wer < 0:
        wer = -1.0
        wer_pct = -1.0
    else:
        wer = target_metrics.wer - source_metrics.wer
        wer_pct = round(wer * 100.0, 2)
    return {
        "wer": wer,
        "wer_pct": wer_pct,
    }


def compute_wer(reference_text: str, hypothesis_text: str, *, wer_client, language: str = "auto") -> Dict[str, Any]:
    """Compute WER through the configured service, propagating failures."""
    result = dict(wer_client.compute(reference_text, hypothesis_text, language=language))
    error_rate = float(result["error_rate"])
    if not math.isfinite(error_rate) or error_rate < 0:
        raise ValueError(f"Invalid WER result: {result!r}")
    return {**result, "error_rate": error_rate, "metric": result.get("metric", "mixed"), "backend": "wer_service"}


def evaluate_quality(audio_bytes: bytes, utmos_client) -> Dict[str, Any]:
    result = dict(utmos_client.predict(audio_bytes))
    score = float(result["score"])
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"Invalid UTMOS result: {result!r}")
    return {**result, "score": score}


def evaluate_speaker_similarity(source_audio_bytes: bytes, target_audio_bytes: bytes, spk_sim_client) -> Optional[Dict[str, Any]]:
    if spk_sim_client is None:
        return None
    with _spk_sim_lock:
        result = dict(spk_sim_client.similarity(source_audio_bytes, target_audio_bytes, timeout=30))
    similarity = float(result["similarity"])
    if not math.isfinite(similarity) or not -1 <= similarity <= 1:
        raise ValueError(f"Invalid speaker similarity result: {result!r}")
    return {**result, "similarity": similarity}


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_audio(
    audio_bytes: bytes,
    reference_text: str,
    source_audio_bytes: bytes,
    *,
    asr_client=None,
    wer_client=None,
    utmos_client=None,
    spk_sim_client=None,
    parallel: bool = True,
    sample_language: Optional[str] = None,
) -> EvalMetrics:
    """Evaluate a single audio sample.

    When *parallel* is True (default), ASR, UTMOS, and SpkSim run
    concurrently via ``ThreadPoolExecutor`` since they hit independent
    GPU services with no data dependency between them.

    Args:
        audio_bytes: The audio to evaluate (WAV bytes).
        reference_text: The expected transcript.
        source_audio_bytes: The original source audio for speaker similarity.
        asr_client: ASRClient instance (optional, skips WER if None).
        wer_client: WERClient instance (required when ASR is enabled).
        utmos_client: UTMOSClient instance (optional, skips UTMOS if None).
        spk_sim_client: SpkSimClient instance (optional, skips SpkSim if None).
        parallel: If True, run eval axes concurrently (default True).

    Returns:
        EvalMetrics with all computed metrics.
    """
    if asr_client is not None and wer_client is None:
        raise ValueError("WER service client is required when ASR evaluation is enabled")
    t0 = time.time()
    metrics = EvalMetrics()

    def _run_asr():
        if asr_client is None:
            return
        result = asr_client.transcribe(
            audio_bytes,
            language=_asr_language(sample_language, reference_text),
        )
        metrics.asr_text = result["text"]
        wer_result = compute_wer(reference_text, metrics.asr_text, wer_client=wer_client)
        metrics.wer = wer_result["error_rate"]
        metrics.wer_backend = wer_result["backend"]
        metrics.wer_metric = wer_result["metric"]
        metrics.wer_pct = _wer_pct(metrics.wer)
        logger.debug(
            "WER=%.2f%% backend=%s  ref=%s  hyp=%s",
            metrics.wer_pct,
            metrics.wer_backend,
            reference_text[:60],
            metrics.asr_text[:60],
        )

    def _run_utmos():
        if utmos_client is None:
            return
        metrics.utmos_score = evaluate_quality(audio_bytes, utmos_client)["score"]

    def _run_spk_sim():
        if spk_sim_client is None:
            return
        metrics.spk_similarity = evaluate_speaker_similarity(source_audio_bytes, audio_bytes, spk_sim_client)["similarity"]

    if parallel:
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="eval") as pool:
            futures = {
                pool.submit(_run_asr): "asr",
                pool.submit(_run_utmos): "utmos",
                pool.submit(_run_spk_sim): "spk_sim",
            }
            for future in as_completed(futures):
                future.result()
    else:
        _run_asr()
        _run_utmos()
        _run_spk_sim()

    metrics.eval_time_sec = time.time() - t0
    return metrics


def evaluate_pair(
    source_audio_bytes: bytes,
    edited_audio_bytes: bytes,
    reference_text: str,
    *,
    asr_client=None,
    wer_client=None,
    utmos_client=None,
    spk_sim_client=None,
    parallel: bool = True,
    sample_language: Optional[str] = None,
) -> PairEvalResult:
    """Evaluate a forward-direction edit pair (source -> edited).

    Computes metrics for both the source and edited audio.  When
    *parallel* is True, source and edited evaluations run concurrently
    (each internally also parallelises ASR/UTMOS/SpkSim).

    Args:
        source_audio_bytes: Original source audio WAV bytes.
        edited_audio_bytes: Edited audio WAV bytes.
        reference_text: The transcript text.
        asr_client: ASRClient instance.
        wer_client: WERClient instance.
        utmos_client: UTMOSClient instance.
        spk_sim_client: SpkSimClient instance.
        parallel: If True, run source & edited eval concurrently.

    Returns:
        PairEvalResult with metrics for both directions.
    """
    t0 = time.time()

    if parallel:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pair") as pool:
            src_future = pool.submit(
                evaluate_audio,
                source_audio_bytes,
                reference_text,
                source_audio_bytes,
                asr_client=asr_client,
                wer_client=wer_client,
                utmos_client=utmos_client,
                spk_sim_client=None,
                parallel=True,
                sample_language=sample_language,
            )
            ed_future = pool.submit(
                evaluate_audio,
                edited_audio_bytes,
                reference_text,
                source_audio_bytes,
                asr_client=asr_client,
                wer_client=wer_client,
                utmos_client=utmos_client,
                spk_sim_client=spk_sim_client,
                parallel=True,
                sample_language=sample_language,
            )
            source_metrics = src_future.result()
            target_metrics = ed_future.result()
    else:
        source_metrics = evaluate_audio(
            source_audio_bytes,
            reference_text,
            source_audio_bytes,
            asr_client=asr_client,
            wer_client=wer_client,
            utmos_client=utmos_client,
            spk_sim_client=None,
            parallel=False,
            sample_language=sample_language,
        )
        target_metrics = evaluate_audio(
            edited_audio_bytes,
            reference_text,
            source_audio_bytes,
            asr_client=asr_client,
            wer_client=wer_client,
            utmos_client=utmos_client,
            spk_sim_client=spk_sim_client,
            parallel=False,
            sample_language=sample_language,
        )

    return PairEvalResult(
        source_metrics=source_metrics,
        target_metrics=target_metrics,
        eval_time_sec=time.time() - t0,
    )
