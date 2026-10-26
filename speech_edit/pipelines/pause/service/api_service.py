"""FastAPI service entrypoint for pause edit execution.

Callers provide source audio plus one structured pause-edit instruction. The
instruction may contain one or more pause insertions or reductions. The service
reuses the shared pause synthesis core and returns generated audio plus
pause/evaluation metadata.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_edit.pipelines import load_parameters, resolve_request_parameters
from speech_edit.runtime.paths import resolve_output_path

import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from speech_edit.clients.services import (
    AlignmentClient,
    ASRClient,
    BoundaryPauseSmootherClient,
    SpkSimClient,
    UTMOSClient,
    WERClient,
    WDTWClient,
)
from speech_edit.data.language import classify_text_language
from speech_edit.pipelines.pause import PIPELINE_TYPE, PIPELINE_VERSION
from speech_edit.pipelines.pause.evaluator import (
    PauseEvalResult,
    evaluate_pause_edit,
    serialize_eval_results,
)
from speech_edit.pipelines.pause.synthesizer import synthesize_pause_edit
from speech_edit.pipelines.pause.tag_utils import (
    INSTRUCTION_ACTIONS,
    PAUSE_LEVEL_TO_DURATION_MS,
    PauseEditGroup,
    PauseEditProfile,
    build_pause_nl_instructions,
    make_text_versions,
    parse_instruction_pause_tags,
    strip_pause_tags,
    reverse_operation,
    validate_pause_tags,
)


DEFAULT_OUTPUT_DIR = "services/" + datetime.now().strftime("%Y-%m-%d") + "/pause_edit_service"
DEFAULT_DURATION_POOL = [
    PAUSE_LEVEL_TO_DURATION_MS[level] for level in sorted(PAUSE_LEVEL_TO_DURATION_MS)
]


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    pipeline_parameters: dict = field(default_factory=lambda: load_parameters("service/pause_edit_pipeline/defaults"))
    aligner_url: str = "http://localhost:8029"
    smoother_url: str = "http://localhost:8018"
    asr_url: str = "http://localhost:8010"
    wer_url: str = "http://localhost:8011"
    utmos_url: str = "http://localhost:8017"
    spk_sim_url: str = "http://localhost:8019"
    wdtw_url: str = "http://localhost:8032"
    init_evaluate_services: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR


RUNTIME_CONFIG = ServiceRuntimeConfig()


class AudioPayload(BaseModel):
    data: str = Field(..., description="Base64-encoded source audio bytes.")
    format: str = Field("wav", description="Input audio format label.")


class PauseInstructionBundle(BaseModel):
    increase_pause: List[str] = Field(default_factory=list)
    decrease_pause: List[str] = Field(default_factory=list)


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioPayload
    source_text: str = Field(..., min_length=1)
    instruction_st: str = Field(..., min_length=1)
    profile_id: int = Field(1, ge=1)
    sample_id: Optional[str] = None
    clip_id: Optional[str] = None
    language: Optional[str] = None
    instructions_nl: Optional[PauseInstructionBundle] = None
    persist: bool = False
    output_dir: Optional[str] = None
    evaluate: Optional[bool] = None
    aligner_backend: Optional[Literal["qwen3"]] = None
    synth_method: Optional[Literal["insert_only", "single_region", "dual_boundary"]] = None
    expand_radius: Optional[float] = Field(None, ge=0.0)
    center_regen_second_pass: Optional[bool] = None
    center_regen_margin_sec: Optional[float] = Field(None, ge=0.0)


class ExecuteResponse(BaseModel):
    status: str
    pipeline_type: str
    pipeline_version: str
    sample_id: str
    profile_id: int
    source_text: str
    instruction_st: str
    generated_text: str
    audio: AudioPayload
    pause_edit_plan: Dict[str, Any]
    eval_results: Dict[str, Any]
    metadata: Dict[str, Any]


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_pause_edit_group(
    group: PauseEditGroup,
    output_dir: str,
    sample_id: str,
) -> str:
    sample_dir = str(resolve_output_path(Path(output_dir) / sample_id))
    os.makedirs(sample_dir, exist_ok=True)

    original_path = os.path.join(sample_dir, "original_audio.wav")
    with open(original_path, "wb") as f:
        f.write(group.original_audio)

    stage1_info = dict(group.metadata or {})
    metadata = {
        "version": 1,
        "pipeline_version": PIPELINE_VERSION,
        "pipeline_type": PIPELINE_TYPE,
        "sample_id": sample_id,
        "input_id": stage1_info.get("input_id"),
        "timestamp": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "clip_id": stage1_info.get("clip_id") or stage1_info.get("segment_clip_id"),
        "language": stage1_info.get("language"),
        "from_dataset": stage1_info.get("source_dataset"),
        "sample_rate": stage1_info.get("sample_rate"),
        "original_duration_sec": stage1_info.get("gen_duration_sec"),
        "normed_text": group.normed_text,
        "original_audio_path": "original_audio.wav",
        "duration_pool": group.duration_pool,
        "num_operations": group.num_operations,
        "profiles": [
            {
                "profile_id": p.profile_id,
                "operation": p.operation,
                "st_inst": p.st_inst,
                "nl_inst": p.nl_inst,
                "tags": [
                    {
                        "char_idx": t.char_idx,
                        "act": t.act,
                        "level": t.level,
                        "delta_dur_ms": t.delta_dur_ms,
                        "punctuation": t.punctuation,
                    }
                    for t in p.tags
                ],
            }
            for p in group.profiles
        ],
        "stage1_info": stage1_info,
    }
    save_json(os.path.join(sample_dir, "metadata.json"), metadata)
    return sample_dir


def save_synthesis_result(
    synth_pair,
    eval_result: Optional[PauseEvalResult],
    sample_dir: str,
    profile_id: int,
) -> str:
    profile_dir = os.path.join(sample_dir, f"profile_{profile_id}")
    os.makedirs(profile_dir, exist_ok=True)

    with open(os.path.join(profile_dir, "original.wav"), "wb") as f:
        f.write(synth_pair.original_audio_bytes)
    with open(os.path.join(profile_dir, "generated.wav"), "wb") as f:
        f.write(synth_pair.generated_audio_bytes)

    text_versions = make_text_versions(synth_pair.normed_text, synth_pair.profile.tags)
    if eval_result is None:
        evaluation_payload: Dict[str, Any] = {"skipped": True}
    else:
        evaluation_payload = serialize_eval_results(eval_result)
    result = {
        "profile_id": profile_id,
        "normed_text": synth_pair.normed_text,
        "generated_text": synth_pair.generated_text,
        "text_versions": text_versions,
        "operation": synth_pair.profile.operation,
        "st_inst": synth_pair.profile.st_inst,
        "nl_inst": synth_pair.profile.nl_inst,
        "pause_edit_plan": synth_pair.pause_edit_plan,
        "original_duration_sec": round(synth_pair.original_duration_sec, 3),
        "generated_duration_sec": round(synth_pair.generated_duration_sec, 3),
        "evaluation": evaluation_payload,
        "synthesis_metadata": synth_pair.metadata,
    }
    save_json(os.path.join(profile_dir, "result.json"), result)
    return profile_dir


def _decode_audio(payload: AudioPayload) -> tuple[bytes, int, float]:
    try:
        raw = base64.b64decode(payload.data)
        with io.BytesIO(raw) as buf:
            audio, sample_rate = sf.read(buf, dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=-1)
        duration = len(audio) / sample_rate
        out = io.BytesIO()
        sf.write(out, audio, sample_rate, format="WAV", subtype="PCM_16")
        return out.getvalue(), int(sample_rate), float(duration)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}") from exc


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


def _validate_request_controls(request: ExecuteRequest) -> None:
    valid, error = validate_pause_tags(
        request.instruction_st,
        request.source_text,
        allowed_actions=INSTRUCTION_ACTIONS,
    )
    if not valid:
        raise HTTPException(status_code=400, detail=error)


def _make_services(*, evaluate: bool):
    return {
        "aligner_client": AlignmentClient(RUNTIME_CONFIG.aligner_url),
        "smoother_client": BoundaryPauseSmootherClient(RUNTIME_CONFIG.smoother_url),
        "asr_client": ASRClient(RUNTIME_CONFIG.asr_url) if evaluate else None,
        "wer_client": WERClient(RUNTIME_CONFIG.wer_url) if evaluate else None,
        "utmos_client": UTMOSClient(RUNTIME_CONFIG.utmos_url) if evaluate else None,
        "spk_sim_client": SpkSimClient(RUNTIME_CONFIG.spk_sim_url) if evaluate else None,
        "wdtw_client": WDTWClient(RUNTIME_CONFIG.wdtw_url) if evaluate else None,
    }


def _default_instruction_metadata(
    operation: str,
) -> tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    tags = parse_instruction_pause_tags(operation, strip_pause_tags(operation))
    reverse = reverse_operation(operation)
    actions = {tag.act for tag in tags}
    if actions == {"ins"}:
        return (
            {"increase_pause": [operation], "decrease_pause": [reverse]},
            {
                "increase_pause": build_pause_nl_instructions(operation),
                "decrease_pause": build_pause_nl_instructions(reverse),
            },
        )
    if actions == {"red"}:
        return (
            {"increase_pause": [reverse], "decrease_pause": [operation]},
            {
                "increase_pause": build_pause_nl_instructions(reverse),
                "decrease_pause": build_pause_nl_instructions(operation),
            },
        )
    # The legacy schema has only all-increase and all-decrease buckets. Keep
    # those buckets intact and persist a separate additive key for a mixed
    # operation instead of assigning it a false direction.
    return (
        {"increase_pause": [], "decrease_pause": [], "mixed_pause": [operation]},
        {
            "increase_pause": [],
            "decrease_pause": [],
            "mixed_pause": build_pause_nl_instructions(operation),
        },
    )


def _profile_from_request(request: ExecuteRequest) -> PauseEditProfile:
    operation = request.instruction_st
    st_inst, default_nl_inst = _default_instruction_metadata(operation)
    nl_inst = (
        request.instructions_nl.model_dump()
        if request.instructions_nl is not None
        else default_nl_inst
    )
    return PauseEditProfile(
        profile_id=request.profile_id,
        operation=operation,
        st_inst=st_inst,
        nl_inst=nl_inst,
        tags=parse_instruction_pause_tags(operation, request.source_text),
    )


def _group_from_request(
    request: ExecuteRequest,
    *,
    audio_bytes: bytes,
    sample_rate: int,
    duration: float,
    sample_id: str,
) -> PauseEditGroup:
    profile = _profile_from_request(request)
    num_operations = len(profile.tags)
    metadata: Dict[str, Any] = {
        "input_id": sample_id,
        "clip_id": request.clip_id or sample_id,
        "language": request.language or classify_text_language(request.source_text),
        "sample_rate": sample_rate,
        "source_dataset": "service_request",
        "original_plain_text": request.source_text,
        "gen_duration_sec": round(duration, 2),
        "num_profiles_requested": 1,
        "num_profiles_valid": 1,
        "duration_pool": list(DEFAULT_DURATION_POOL),
        "num_operations": num_operations,
        "segment_clip_id": request.clip_id or sample_id,
        "service_request": True,
    }
    return PauseEditGroup(
        normed_text=request.source_text,
        profiles=[profile],
        duration_pool=list(DEFAULT_DURATION_POOL),
        num_operations=num_operations,
        original_audio=audio_bytes,
        metadata=metadata,
    )


def _serialize_eval_results(eval_result: Optional[PauseEvalResult]) -> Dict[str, Any]:
    return serialize_eval_results(eval_result)


app = FastAPI(
    title="Pause Edit Pipeline Service",
    description="Execute structured pause edits through the shared pipeline core.",
    version=PIPELINE_VERSION,
)


@app.get("/health")
def health(deep: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "healthy",
        "service": "pause_edit_pipeline",
        "pipeline_type": PIPELINE_TYPE,
        "pipeline_version": PIPELINE_VERSION,
    }
    if not deep:
        return payload
    evaluate = RUNTIME_CONFIG.init_evaluate_services
    services = _make_services(evaluate=evaluate)
    checks: Dict[str, Any] = {}
    for name, client in services.items():
        if client is None:
            continue
        dep_name = name.removesuffix("_client")
        try:
            checks[dep_name] = client.health()
        except Exception as exc:
            checks[dep_name] = {"status": "unhealthy", "error": str(exc)}
            payload["status"] = "degraded"
    payload["dependencies"] = checks
    return payload


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    request = resolve_request_parameters(request, RUNTIME_CONFIG.pipeline_parameters, ('aligner_backend', 'synth_method', 'expand_radius', 'center_regen_second_pass', 'center_regen_margin_sec'))
    _validate_request_controls(request)
    audio_bytes, sample_rate, duration = _decode_audio(request.audio)
    evaluate = _evaluation_enabled_for_request(request)
    sample_id = request.sample_id or f"service_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    group = _group_from_request(
        request,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        duration=duration,
        sample_id=sample_id,
    )
    services = _make_services(evaluate=evaluate)
    sample_dir = None
    profile_dir = None
    try:
        if request.persist:
            sample_dir = save_pause_edit_group(
                group,
                request.output_dir or RUNTIME_CONFIG.output_dir,
                sample_id,
            )
        profile = group.profiles[0]
        synth_pair = synthesize_pause_edit(
            profile=profile,
            group=group,
            smoother_client=services["smoother_client"],
            alignment_client=services["aligner_client"],
            output_dir=os.path.join(sample_dir, "f5_output") if sample_dir else None,
            aligner_backend=request.aligner_backend,
            synth_method=request.synth_method,
            expand_radius=request.expand_radius,
            center_regen_second_pass=request.center_regen_second_pass,
            center_regen_margin_sec=request.center_regen_margin_sec,
        )
        eval_result = None
        if evaluate:
            eval_result = evaluate_pause_edit(
                synth_pair=synth_pair,
                asr_client=services.get("asr_client"),
                alignment_client=services.get("aligner_client"),
                wer_client=services.get("wer_client"),
                aligner_backend=request.aligner_backend,
                utmos_client=services.get("utmos_client"),
                spk_sim_client=services.get("spk_sim_client"),
                wdtw_client=services.get("wdtw_client"),
                sample_language=request.language,
            )
        if request.persist and sample_dir is not None:
            profile_dir = save_synthesis_result(
                synth_pair=synth_pair,
                eval_result=eval_result,
                sample_dir=sample_dir,
                profile_id=profile.profile_id,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generated_audio = AudioPayload(
        data=base64.b64encode(synth_pair.generated_audio_bytes).decode("utf-8"),
        format="wav",
    )
    eval_results = _serialize_eval_results(eval_result)
    metadata: Dict[str, Any] = {
        "persist": request.persist,
        "sample_dir": sample_dir,
        "profile_dir": profile_dir,
        "stage1_info": dict(group.metadata or {}),
        "pause_edit_plan": synth_pair.pause_edit_plan,
        "synthesis_metadata": synth_pair.metadata,
    }
    return ExecuteResponse(
        status="ok",
        pipeline_type=PIPELINE_TYPE,
        pipeline_version=PIPELINE_VERSION,
        sample_id=sample_id,
        profile_id=request.profile_id,
        source_text=request.source_text,
        instruction_st=request.instruction_st,
        generated_text=synth_pair.generated_text,
        audio=generated_audio,
        pause_edit_plan=synth_pair.pause_edit_plan,
        eval_results=eval_results,
        metadata=metadata,
    )


def main() -> None:
    global RUNTIME_CONFIG
    parser = argparse.ArgumentParser(description="Pause edit pipeline service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8037)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--aligner-url", default=RUNTIME_CONFIG.aligner_url)
    parser.add_argument("--smoother-url", default=RUNTIME_CONFIG.smoother_url)
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
        aligner_url=args.aligner_url,
        smoother_url=args.smoother_url,
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
