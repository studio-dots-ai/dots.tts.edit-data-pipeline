"""FastAPI service entrypoint for emotion edit execution."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_edit.pipelines import load_parameters, resolve_request_parameters
from speech_edit.runtime.paths import resolve_output_path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from speech_edit.data.language import alignment_language_code
from speech_edit.clients.services import (
    ASRClient,
    AlignmentClient,
    AudioAssemblyClient,
    BoundaryPauseSmootherClient,
    IndexTTSClient,
    SpkSimClient,
    TedTTSClient,
    UTMOSClient,
    WDTWClient,
    WERClient,
)

from .. import PIPELINE_TYPE, PIPELINE_VERSION
from ..evaluator import evaluate_synthesis_result
from ..service_contract import EmotionEditServiceContract, build_service_edit_contract
from ..synthesizer import (
    AssembledEditPair,
    SynthesizedAudio,
    SynthesisResult,
    _assemble_from_base_and_donor,
    _align_edit_pair_outputs,
    _get_wav_duration,
    _numpy_to_wav,
    _wav_to_numpy,
    synthesize_tagged_text_group,
)
from ..synthesis_profile import (
    SYNTHESIS_BACKEND_TED,
    build_synthesis_profile,
    estimate_text_duration_sec,
    normalize_emotion_condition_mode,
    resolve_backend_for_profile,
    service_mode_from_contract,
)
from ..tag_utils import (
    EMOTION_CONTROL_DESCRIPTION,
    TaggedTextGroup,
    parse_tags,
    render_emo_tag,
    strip_tags,
)


DEFAULT_OUTPUT_DIR = "services/" + datetime.now().strftime("%Y-%m-%d") + "/emotion_edit_service"


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    pipeline_parameters: dict = field(default_factory=lambda: load_parameters("service/emotion_style_edit_pipeline/defaults"))
    indextts_url: str = "http://localhost:8002"
    ted_tts_url: str = "http://localhost:8035"
    aligner_url: str = "http://localhost:8029"
    smoother_url: str = "http://localhost:8018"
    assembly_url: str = "http://localhost:8021"
    asr_url: str = "http://localhost:8010"
    wer_url: str = "http://localhost:8011"
    utmos_url: str = "http://localhost:8017"
    spk_sim_url: str = "http://localhost:8019"
    wdtw_url: str = "http://localhost:8032"
    init_evaluate_services: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR


RUNTIME_CONFIG = ServiceRuntimeConfig()


class AudioPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: str = Field(..., description="Base64-encoded WAV bytes.")
    format: str = Field("wav", description="Audio format label.")


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioPayload
    source_text: str
    instruction_st: str
    generation_id: str = "target"
    instruction_nl: Optional[str] = None
    language: Optional[str] = None
    sample_id: Optional[str] = None
    clip_id: Optional[str] = None
    evaluate: Optional[bool] = None
    persist: bool = False
    output_dir: Optional[str] = None
    return_synthetic_target_audio: bool = False
    aligner_backend: Optional[str] = None
    max_candidates: Optional[int] = None
    multi_frequent_smoothing_radius_sec: float = Field(
        0.45, description="Accepted from agent clients; TED synthesis does not use this field."
    )
    synthesis_mode: Optional[str] = Field(
        None,
        description="single, multi_emo_normal, multi_emo_frequent, or infer.",
    )
    emotion_condition_mode: Optional[str] = Field(
        None,
        description="type_level, desc, or infer.",
    )
    debug: bool = False
    debug_output_dir: Optional[str] = None


class ExecuteResponse(BaseModel):
    status: str
    pipeline_type: str
    pipeline_version: str
    sample_id: str
    generation_id: str
    source_text: str
    target_tagged_text: str
    edit_tagged_text: str
    changed_segments: list[Dict[str, Any]]
    audio: AudioPayload
    synthetic_target_audio: Optional[AudioPayload] = None
    eval_results: Dict[str, Any]
    metadata: Dict[str, Any]


def _estimate_service_text_duration_sec(text: str) -> float:
    return estimate_text_duration_sec(
        text,
        en_word_sec=RUNTIME_CONFIG.pipeline_parameters["duration_estimation_en_word_sec"],
        cjk_char_sec=RUNTIME_CONFIG.pipeline_parameters["duration_estimation_cjk_char_sec"],
    )


def _decode_audio(payload: AudioPayload) -> tuple[bytes, int, float]:
    try:
        if payload.format.lower() != "wav":
            raise ValueError("only WAV audio is supported")
        raw = base64.b64decode(payload.data)
        waveform, sample_rate = _wav_to_numpy(raw)
        if len(waveform) == 0:
            raise ValueError("audio must contain at least one sample")
        wav_bytes = _numpy_to_wav(waveform, sample_rate)
        return wav_bytes, sample_rate, len(waveform) / sample_rate
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}") from exc


def _configured_aligner_backend() -> str:
    backend = RUNTIME_CONFIG.pipeline_parameters["aligner_backend"].strip().lower()
    if backend not in {"qwen3"}:
        raise HTTPException(
            status_code=400,
            detail="configured aligner backend must be qwen3",
        )
    return backend


def _resolve_aligner_backend(request: ExecuteRequest) -> str:
    configured_backend = _configured_aligner_backend()
    if request.aligner_backend is None:
        return configured_backend

    backend = request.aligner_backend.strip().lower()
    if backend not in {"qwen3"}:
        raise HTTPException(
            status_code=400,
            detail="aligner_backend must be qwen3 for this service stack",
        )
    if backend != configured_backend:
        raise HTTPException(
            status_code=400,
            detail=(
                "aligner_backend must match the configured service stack backend "
                f"({configured_backend})"
            ),
        )
    return backend


def _resolve_aligner_language(request: ExecuteRequest) -> tuple[str, str]:
    source = "request" if request.language else "inferred"
    language = alignment_language_code(request.language, text=request.source_text)
    if language not in {"eng", "zho"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "emotion edit service v1 only supports English/Chinese language "
                f"hints; got {request.language!r}"
            ),
        )
    return language, source


def _evaluation_enabled_for_request(request: ExecuteRequest) -> bool:
    evaluation_available = RUNTIME_CONFIG.init_evaluate_services
    if request.evaluate is True and not evaluation_available:
        raise HTTPException(
            status_code=400,
            detail=(
                "Evaluation was requested, but this service was launched without "
                "evaluation models. Restart with the evaluate service profile."
            ),
        )
    return bool(request.evaluate if request.evaluate is not None else evaluation_available)


def _service_urls() -> Dict[str, str]:
    return {
        "indextts": RUNTIME_CONFIG.indextts_url,
        "ted_tts": RUNTIME_CONFIG.ted_tts_url,
        "aligner": RUNTIME_CONFIG.aligner_url,
        "smoother": RUNTIME_CONFIG.smoother_url,
        "assembly": RUNTIME_CONFIG.assembly_url,
        "asr": RUNTIME_CONFIG.asr_url,
        "wer": RUNTIME_CONFIG.wer_url,
        "utmos": RUNTIME_CONFIG.utmos_url,
        "spk_sim": RUNTIME_CONFIG.spk_sim_url,
        "wdtw": RUNTIME_CONFIG.wdtw_url,
    }


def _dependency_health_ok(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    status = str(payload.get("status") or "").strip().lower()
    if status and status not in {"healthy", "ok", "ready"}:
        return False
    for key, value in payload.items():
        if key.endswith("_loaded") and value is False:
            return False
    return True


def _make_audio_assembly_client() -> AudioAssemblyClient:
    return AudioAssemblyClient(_service_urls()["assembly"])


def _build_target_group(
    request: ExecuteRequest,
    *,
    contract,
    audio_bytes: bytes,
    aligner_language: str,
) -> TaggedTextGroup:
    metadata: Dict[str, Any] = {
        "source": "service_request",
        "clip_id": request.clip_id or request.sample_id or "service_request",
        "service_contract": {
            "edit_tagged_text": contract.edit_tagged_text,
            "changed_segments": contract.changed_segments,
            "target_spans": contract.target_spans,
        },
    }
    metadata["language"] = aligner_language
    return TaggedTextGroup(
        sub_type=contract.synthesis_sub_type,
        plain_text=contract.plain_text,
        speaker_prompt_audio=audio_bytes,
        speaker_prompt_text=request.source_text,
        tagged_texts={request.generation_id: contract.target_tagged_text},
        edit_pairs=[],
        boundaries=contract.boundaries,
        metadata=metadata,
    )


def _infer_contract_emotion_condition_mode(contract) -> str:
    modes = {
        str(segment.get("target_emotion_control_mode"))
        for segment in contract.changed_segments
        if segment.get("target_emotion_control_mode")
    }
    uses_duration = _contract_uses_duration(contract)
    if uses_duration:
        raise HTTPException(status_code=400, detail="Duration tags are not supported for emotion editing")
    if not modes:
        return "type_level"
    if len(modes) != 1:
        raise HTTPException(
            status_code=400,
            detail="instruction_st must not mix alpha and description emotion controls",
        )
    mode = modes.pop()
    if mode == "description":
        return "desc"
    if mode == "alpha":
        return "type_level"
    raise HTTPException(
        status_code=400,
        detail=f"Unsupported instruction_st emotion control mode: {mode}",
    )


def _resolve_request_emotion_condition_mode(
    request: ExecuteRequest,
    contract: EmotionEditServiceContract,
) -> str:
    actual = _infer_contract_emotion_condition_mode(contract)
    expected = normalize_emotion_condition_mode(
        request.emotion_condition_mode,
        allow_infer=True,
    )
    if expected != "infer" and expected != actual:
        raise HTTPException(
            status_code=400,
            detail=(
                "emotion_condition_mode does not match instruction_st: "
                f"expected={expected}, actual={actual}"
            ),
        )
    if actual == "desc":
        missing_ref = [
            segment.get("segment_idx")
            for segment in contract.changed_segments
            if not segment.get("target_emotion_type_ref")
        ]
        if missing_ref:
            raise HTTPException(
                status_code=400,
                detail="desc emotion_condition_mode requires type_ref on every target segment",
            )
    return actual


def _contract_uses_duration(contract) -> bool:
    return any(
        segment.get("target_dur") is not None
        for segment in contract.changed_segments
    )


def _resolve_request_synthesis_mode(
    request: ExecuteRequest,
    contract: EmotionEditServiceContract,
) -> str:
    is_whole_sentence_edit = contract.synthesis_sub_type == "single"
    return service_mode_from_contract(
        is_whole_sentence_edit=is_whole_sentence_edit,
        requested_mode=request.synthesis_mode,
    )


def _render_ted_target_tagged_text(tagged_text: str) -> str:
    parts = []
    for segment in parse_tags(tagged_text):
        if segment.tag_name is None:
            if segment.text:
                parts.append(segment.text)
            continue
        if segment.emotion_control_mode == EMOTION_CONTROL_DESCRIPTION:
            emotion_desc = segment.emotion_desc
            emotion_type_ref = segment.emotion_type_ref
            if not emotion_type_ref:
                raise HTTPException(
                    status_code=400,
                    detail="backend=ted desc tags must include type_ref",
                )
        elif segment.tag_type == "none":
            emotion_desc = "neutral, calm delivery"
            emotion_type_ref = "calm"
        else:
            raise HTTPException(
                status_code=400,
                detail="backend=ted requires every target segment to use desc",
            )
        parts.append(
            render_emo_tag(
                text=segment.text,
                emotion_desc=emotion_desc,
                emotion_type_ref=emotion_type_ref,
            )
        )
    rendered = "".join(parts)
    estimated_duration = _estimate_service_text_duration_sec(strip_tags(rendered))
    soft_limit = RUNTIME_CONFIG.pipeline_parameters["ted_estimated_duration_soft_limit_sec"]
    if estimated_duration > soft_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                "TED estimated target duration exceeds the soft limit "
                f"({soft_limit:.1f}s): {estimated_duration:.2f}s"
            ),
        )
    return rendered


def _prepare_contract_for_ted(
    contract: EmotionEditServiceContract,
) -> EmotionEditServiceContract:
    target_tagged_text = _render_ted_target_tagged_text(contract.target_tagged_text)
    return replace(
        contract,
        target_tagged_text=target_tagged_text,
    )


def _persist_result(
    *,
    sample_id: str,
    output_dir: str,
    source_audio: bytes,
    synthetic_target_audio: bytes,
    edited_audio: bytes,
    metadata: Dict[str, Any],
    eval_results: Dict[str, Any],
) -> str:
    sample_dir = resolve_output_path(Path(output_dir) / sample_id)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "source.wav").write_bytes(source_audio)
    (sample_dir / "synthetic_target.wav").write_bytes(synthetic_target_audio)
    (sample_dir / "edited.wav").write_bytes(edited_audio)
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if eval_results:
        (sample_dir / "eval_results.json").write_text(
            json.dumps(eval_results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return str(sample_dir)


def _evaluate_service_output(
    *,
    request: ExecuteRequest,
    contract,
    source_audio: bytes,
    source_sample_rate: int,
    aligner_backend: str,
    aligner_language: Optional[str],
    edited_audio: bytes,
    synthetic_target_audio: bytes,
    edited_duration: float,
    synthetic_duration: float,
) -> Dict[str, Any]:
    urls = _service_urls()
    source_duration = _get_wav_duration(source_audio)
    pair = AssembledEditPair(
        source_id="source",
        target_id=request.generation_id,
        nl_instruction=request.instruction_nl or "Apply the marked emotion edit.",
        st_instruction=request.instruction_st,
        source_audio=source_audio,
        target_audio=edited_audio,
        source_duration=source_duration,
        target_duration=edited_duration,
        changed_segments=contract.changed_segments,
        assembly_method="service_splice_and_smooth",
    )
    synthesis_result = SynthesisResult(
        synthesized_audios={
            "source": SynthesizedAudio(
                text_id="source",
                tagged_text=contract.source_tagged_text,
                audio_bytes=source_audio,
                sample_rate=source_sample_rate,
                duration_sec=source_duration,
                segment_timestamps=[],
                synthesis_mode="service_source",
            ),
            request.generation_id: SynthesizedAudio(
                text_id=request.generation_id,
                tagged_text=contract.target_tagged_text,
                audio_bytes=synthetic_target_audio,
                sample_rate=24000,
                duration_sec=synthetic_duration,
                segment_timestamps=[],
                synthesis_mode="service_target",
            ),
        },
        edit_pairs=[pair],
        timing={},
    )
    _align_edit_pair_outputs(
        edit_pairs=synthesis_result.edit_pairs,
        synthesized=synthesis_result.synthesized_audios,
        aligner_client=AlignmentClient(urls["aligner"]),
        aligner_backend=aligner_backend,
        aligner_language=aligner_language,
    )
    eval_result = evaluate_synthesis_result(
        synthesis_result,
        asr_url=urls["asr"],
        wer_url=urls["wer"],
        utmos_url=urls["utmos"],
        spk_sim_url=urls["spk_sim"],
        wdtw_url=urls["wdtw"],
        aligner_language=aligner_language,
        skip_quality=False,
        skip_spk_sim=False,
    )
    return eval_result


app = FastAPI(
    title="Emotion Edit Pipeline Service",
    description="Execute structured emotion edits through the shared emotion pipeline core.",
    version=PIPELINE_VERSION,
)


@app.get("/health")
def health(deep: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "healthy",
        "service": "emotion_edit_pipeline",
        "pipeline_type": PIPELINE_TYPE,
        "pipeline_version": PIPELINE_VERSION,
    }
    if not deep:
        return payload

    urls = _service_urls()
    clients: Dict[str, Any] = {
        "indextts": IndexTTSClient(urls["indextts"]),
        "ted_tts": TedTTSClient(urls["ted_tts"]),
        "aligner": AlignmentClient(urls["aligner"]),
        "smoother": BoundaryPauseSmootherClient(urls["smoother"]),
        "assembly": AudioAssemblyClient(urls["assembly"]),
    }
    if RUNTIME_CONFIG.init_evaluate_services:
        clients.update(
            {
                "asr": ASRClient(urls["asr"]),
                "wer": WERClient(urls["wer"]),
                "utmos": UTMOSClient(urls["utmos"]),
                "spk_sim": SpkSimClient(urls["spk_sim"]),
                "wdtw": WDTWClient(urls["wdtw"]),
            }
        )
    checks: Dict[str, Any] = {}
    for name, client in clients.items():
        try:
            checks[name] = client.health()
            if not _dependency_health_ok(checks[name]):
                payload["status"] = "degraded"
        except Exception as exc:
            checks[name] = {"status": "unhealthy", "error": str(exc)}
            payload["status"] = "degraded"
    payload["dependencies"] = checks
    return payload


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    request = resolve_request_parameters(request, RUNTIME_CONFIG.pipeline_parameters, ('max_candidates',))
    if request.max_candidates < 1:
        raise HTTPException(status_code=400, detail="max_candidates must be >= 1")

    audio_bytes, source_sample_rate, _duration = _decode_audio(request.audio)
    evaluate = _evaluation_enabled_for_request(request)
    aligner_backend = _resolve_aligner_backend(request)
    aligner_language, language_source = _resolve_aligner_language(request)
    sample_id = request.sample_id or f"service_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        contract = build_service_edit_contract(
            source_text=request.source_text,
            instruction_st=request.instruction_st,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        synthesis_mode = _resolve_request_synthesis_mode(
            request,
            contract,
        )
        emotion_condition_mode = _resolve_request_emotion_condition_mode(
            request,
            contract,
        )
        synthesis_backend = resolve_backend_for_profile(
            synthesis_mode,
            emotion_condition_mode,
        )
        if synthesis_backend == SYNTHESIS_BACKEND_TED:
            contract = _prepare_contract_for_ted(contract)
        estimated_duration_sec = (
            _estimate_service_text_duration_sec(strip_tags(contract.target_tagged_text))
            if synthesis_backend == SYNTHESIS_BACKEND_TED
            else None
        )
        synthesis_profile = build_synthesis_profile(
            synthesis_mode=synthesis_mode,
            emotion_condition_mode=emotion_condition_mode,
            estimated_duration_sec=estimated_duration_sec,
            ted_estimated_duration_soft_limit_sec=(
                RUNTIME_CONFIG.pipeline_parameters["ted_estimated_duration_soft_limit_sec"]
            ),
            ted_max_generated_duration_sec=RUNTIME_CONFIG.pipeline_parameters["ted_max_generated_duration_sec"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    urls = _service_urls()
    debug_root = request.debug_output_dir
    if request.debug and not debug_root:
        debug_root = str(
            Path(
                request.output_dir or RUNTIME_CONFIG.output_dir
            )
            / sample_id
            / "debug"
        )
    if request.debug and debug_root:
        debug_root = str(resolve_output_path(debug_root))
    group = _build_target_group(
        request,
        contract=contract,
        audio_bytes=audio_bytes,
        aligner_language=aligner_language,
    )
    group.metadata["emotion_condition_mode"] = emotion_condition_mode
    group.metadata["emotion_control_mode"] = synthesis_profile["emotion_control_mode"]
    group.metadata["synthesis_backend"] = synthesis_backend
    group.metadata["synthesis_mode"] = synthesis_mode
    group.metadata["synthesis_profile"] = synthesis_profile
    group.sub_type = synthesis_mode
    try:
        synth_result = synthesize_tagged_text_group(
            group,
            indextts_url=urls["indextts"],
            ted_tts_url=urls["ted_tts"],
            assembly_f5_steps=RUNTIME_CONFIG.pipeline_parameters["assembly_f5_steps"],
            boundary_pause_smoother_url=urls["smoother"],
            audio_assembly_url=urls["assembly"],
            aligner_url=urls["aligner"],
            max_candidates=request.max_candidates,
            aligner_backend=aligner_backend,
            parallel=False,
            debug_synthesize=request.debug,
            debug_root_dir=debug_root,
        )
        target_synth = synth_result.synthesized_audios[request.generation_id]
        edited_audio = _assemble_from_base_and_donor(
            assembly_f5_steps=RUNTIME_CONFIG.pipeline_parameters["assembly_f5_steps"],
            base_audio_bytes=audio_bytes,
            donor_audio_bytes=target_synth.audio_bytes,
            edit_tagged_text=contract.edit_tagged_text,
            assembly_client=_make_audio_assembly_client(),
            aligner_language=aligner_language,
            aligner_backend=aligner_backend,
            debug=request.debug,
            debug_dir=str(Path(debug_root) / "assembly") if debug_root else None,
        )
        edited_duration = _get_wav_duration(edited_audio)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    eval_results: Dict[str, Any] = {}
    if evaluate:
        try:
            eval_results = _evaluate_service_output(
                request=request,
                contract=contract,
                source_audio=audio_bytes,
                source_sample_rate=source_sample_rate,
                aligner_backend=aligner_backend,
                aligner_language=aligner_language,
                edited_audio=edited_audio,
                synthetic_target_audio=target_synth.audio_bytes,
                edited_duration=edited_duration,
                synthetic_duration=target_synth.duration_sec,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Evaluation failed: {exc}") from exc

    metadata: Dict[str, Any] = {
        "clip_id": request.clip_id,
        "language": aligner_language,
        "language_source": language_source,
        "source_tagged_text": contract.source_tagged_text,
        "target_tagged_text": contract.target_tagged_text,
        "edit_tagged_text": contract.edit_tagged_text,
        "changed_segments": contract.changed_segments,
        "target_spans": contract.target_spans,
        "emotion_condition_mode": emotion_condition_mode,
        "emotion_control_mode": synthesis_profile["emotion_control_mode"],
        "synthesis_backend": synthesis_backend,
        "synthesis_mode": synthesis_mode,
        "estimated_duration_sec": synthesis_profile.get("estimated_duration_sec"),
        "ted_estimated_duration_soft_limit_sec": synthesis_profile.get(
            "ted_estimated_duration_soft_limit_sec"
        ),
        "ted_max_generated_duration_sec": synthesis_profile.get(
            "ted_max_generated_duration_sec"
        ),
        "ted_max_mel_tokens": synthesis_profile.get("ted_max_mel_tokens"),
        "synthesis_profile": synthesis_profile,
        "synthesis": {
            "sub_type": group.sub_type,
            "mode": target_synth.synthesis_mode,
            "duration_sec": target_synth.duration_sec,
            "duration_control": target_synth.duration_control,
            "timing": synth_result.timing,
        },
        "assembly": {
            "method": "source_base_synthetic_target_donor",
            "aligner_backend": aligner_backend,
            "aligner_language": aligner_language,
            "edited_duration_sec": edited_duration,
        },
    }
    if request.persist:
        output_dir = request.output_dir or RUNTIME_CONFIG.output_dir
        metadata["sample_dir"] = _persist_result(
            sample_id=sample_id,
            output_dir=output_dir,
            source_audio=audio_bytes,
            synthetic_target_audio=target_synth.audio_bytes,
            edited_audio=edited_audio,
            metadata=metadata,
            eval_results=eval_results,
        )

    synthetic_payload = None
    if request.return_synthetic_target_audio:
        synthetic_payload = AudioPayload(
            data=base64.b64encode(target_synth.audio_bytes).decode("utf-8"),
            format="wav",
        )

    return ExecuteResponse(
        status="ok",
        pipeline_type=PIPELINE_TYPE,
        pipeline_version=PIPELINE_VERSION,
        sample_id=sample_id,
        generation_id=request.generation_id,
        source_text=request.source_text,
        target_tagged_text=contract.target_tagged_text,
        edit_tagged_text=contract.edit_tagged_text,
        changed_segments=contract.changed_segments,
        audio=AudioPayload(data=base64.b64encode(edited_audio).decode("utf-8"), format="wav"),
        synthetic_target_audio=synthetic_payload,
        eval_results=eval_results,
        metadata=metadata,
    )


def main() -> None:
    global RUNTIME_CONFIG
    parser = argparse.ArgumentParser(description="Emotion edit pipeline service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8037)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--indextts-url", default=RUNTIME_CONFIG.indextts_url)
    parser.add_argument("--ted-tts-url", default=RUNTIME_CONFIG.ted_tts_url)
    parser.add_argument("--aligner-url", default=RUNTIME_CONFIG.aligner_url)
    parser.add_argument("--smoother-url", default=RUNTIME_CONFIG.smoother_url)
    parser.add_argument("--assembly-url", default=RUNTIME_CONFIG.assembly_url)
    parser.add_argument("--asr-url", default=RUNTIME_CONFIG.asr_url)
    parser.add_argument("--wer-url", default=RUNTIME_CONFIG.wer_url)
    parser.add_argument("--utmos-url", default=RUNTIME_CONFIG.utmos_url)
    parser.add_argument("--spk-sim-url", default=RUNTIME_CONFIG.spk_sim_url)
    parser.add_argument("--wdtw-url", default=RUNTIME_CONFIG.wdtw_url)
    parser.add_argument("--output-dir", default=RUNTIME_CONFIG.output_dir)
    parser.add_argument("--init-evaluate-services", action="store_true")
    parser.add_argument("--pipeline-defaults", required=True)
    parser.add_argument("--pipeline-parameters", type=json.loads, required=True)
    args = parser.parse_args()
    RUNTIME_CONFIG = ServiceRuntimeConfig(
        pipeline_parameters=load_parameters(args.pipeline_defaults, args.pipeline_parameters),
        indextts_url=args.indextts_url,
        ted_tts_url=args.ted_tts_url,
        aligner_url=args.aligner_url,
        smoother_url=args.smoother_url,
        assembly_url=args.assembly_url,
        asr_url=args.asr_url,
        wer_url=args.wer_url,
        utmos_url=args.utmos_url,
        spk_sim_url=args.spk_sim_url,
        wdtw_url=args.wdtw_url,
        init_evaluate_services=args.init_evaluate_services,
        output_dir=str(resolve_output_path(args.output_dir)),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
