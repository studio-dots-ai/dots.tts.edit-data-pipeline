"""FastAPI service entrypoint for tagged text edit execution.

This service is a thin orchestration wrapper around the shared Stage 2+ core:
callers provide source audio plus a structured tagged instruction, and the
service returns generated audio plus spans/evaluation metadata.
"""

from __future__ import annotations

import argparse
import json
import base64
import sys
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from speech_edit.pipelines import load_parameters, resolve_request_parameters
from speech_edit.runtime.paths import resolve_output_path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from speech_edit.clients.services import (
    AlignmentClient,
    ASRClient,
    F5TTSClient,
    SpkSimClient,
    UTMOSClient,
    WERClient,
    WDTWClient,
)

if __package__ in (None, ""):
    from speech_edit.pipelines.text import (  # noqa: E402
        PIPELINE_TYPE,
        PIPELINE_VERSION,
    )
    from speech_edit.pipelines.text.audio_timeline import (  # noqa: E402
        mono_float_to_wav_bytes,
        wav_bytes_to_mono_float,
    )
    from speech_edit.pipelines.text.pipeline import (  # noqa: E402
        PipelineConfig,
        build_tagged_plan_from_st,
        execute_tagged_plan,
    )
    from speech_edit.data.text_markup import (  # noqa: E402
        render_target,
    )
else:
    from .. import PIPELINE_TYPE, PIPELINE_VERSION
    from ..audio_timeline import wav_bytes_to_mono_float, mono_float_to_wav_bytes
    from ..pipeline import (
        PipelineConfig,
        build_tagged_plan_from_st,
        execute_tagged_plan,
    )
    from speech_edit.data.text_markup import render_target

DEFAULT_OUTPUT_DIR = "services/" + datetime.now().strftime("%Y-%m-%d") + "/text_edit_service"


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    pipeline_parameters: dict = field(default_factory=lambda: load_parameters("service/text_edit_pipeline/defaults"))
    aligner_url: str = "http://localhost:8029"
    f5_url: str = "http://localhost:8001"
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


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioPayload
    source_text: str
    instruction_st: str
    generation_id: str = "000"
    instructions_nl: Optional[Dict[str, List[str]]] = None
    reverse_instructions_nl: Optional[Dict[str, List[str]]] = None
    language: Optional[str] = None
    sample_id: Optional[str] = None
    clip_id: Optional[str] = None
    persist: bool = False
    output_dir: Optional[str] = None
    evaluate: Optional[bool] = None
    duration_mode: Optional[str] = None
    expand_radius: Optional[float] = None
    merge_gap: Optional[float] = None
    nfe_step: Optional[int] = None
    cfg_strength: Optional[float] = None
    sway_sampling_coef: Optional[float] = None
    source_crossfade_sec: Optional[float] = None
    regen_crossfade_sec: Optional[float] = None
    seed: Optional[int] = None
    disable_prepend_context: Optional[bool] = None
    fail_if_cannot_prepend_context: Optional[bool] = None
    min_unmasked_context_sec: Optional[float] = None
    min_unmasked_context_ratio: Optional[float] = None
    synthesis_mode: Optional[Literal["one_shot", "per_op"]] = None
    debug: bool = False
    debug_output_dir: Optional[str] = None


class ExecuteResponse(BaseModel):
    status: str
    pipeline_type: str
    pipeline_version: str
    sample_id: str
    generation_id: str
    target_text: str
    audio: AudioPayload
    source_spans: List[List[float]]
    target_spans: List[List[float]]
    target_aligned_spans: List[List[float]]
    eval_results: Dict[str, Any]
    metadata: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None


def _decode_audio(payload: AudioPayload) -> tuple[bytes, int, float]:
    try:
        raw = base64.b64decode(payload.data)
        waveform, sample_rate = wav_bytes_to_mono_float(raw)
        wav_bytes = mono_float_to_wav_bytes(waveform, sample_rate)
        return wav_bytes, sample_rate, len(waveform) / sample_rate
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid audio payload: {exc}") from exc


def _pipeline_config(**overrides) -> PipelineConfig:
    values = {f.name: RUNTIME_CONFIG.pipeline_parameters[f.name]
              for f in fields(PipelineConfig) if f.name != "output_dir"}
    return PipelineConfig(**(values | overrides))


def _config_from_request(request: ExecuteRequest) -> PipelineConfig:
    request = resolve_request_parameters(request, RUNTIME_CONFIG.pipeline_parameters,
        ("duration_mode", "expand_radius", "merge_gap", "nfe_step", "cfg_strength", "sway_sampling_coef",
         "source_crossfade_sec", "regen_crossfade_sec", "seed", "fail_if_cannot_prepend_context",
         "min_unmasked_context_sec", "min_unmasked_context_ratio", "synthesis_mode"), {
        "nfe_step": "f5_steps", "cfg_strength": "f5_cfg_strength",
        "sway_sampling_coef": "f5_sway_sampling_coef",
    })
    if request.disable_prepend_context is None:
        request.disable_prepend_context = not RUNTIME_CONFIG.pipeline_parameters["prepend_context_enabled"]
    evaluation_available = RUNTIME_CONFIG.init_evaluate_services
    if request.evaluate is True and not evaluation_available:
        raise HTTPException(
            status_code=400,
            detail=(
                "Evaluation was requested, but this service was launched without "
                "evaluation models. Restart with the evaluate service profile."
            ),
        )
    evaluate = (
        request.evaluate
        if request.evaluate is not None
        else evaluation_available
    )
    output_dir = request.output_dir or RUNTIME_CONFIG.output_dir
    if request.persist or request.debug:
        output_dir = str(resolve_output_path(output_dir))
        resolve_output_path(Path(output_dir) / (request.sample_id or ""))
    debug_output_dir = request.debug_output_dir
    if request.debug:
        debug_root = resolve_output_path(debug_output_dir or Path(output_dir) / "debug_artifacts")
        resolve_output_path(debug_root / (request.sample_id or "service_request") / request.generation_id)
        if debug_output_dir:
            debug_output_dir = str(debug_root)
    return _pipeline_config(
        output_dir=output_dir,
        duration_mode=request.duration_mode,
        seed=request.seed,
        expand_radius=request.expand_radius,
        merge_gap=request.merge_gap,
        aligner_backend=RUNTIME_CONFIG.pipeline_parameters["aligner_backend"],
        f5_steps=request.nfe_step,
        f5_cfg_strength=request.cfg_strength,
        f5_sway_sampling_coef=request.sway_sampling_coef,
        source_crossfade_sec=request.source_crossfade_sec,
        regen_crossfade_sec=request.regen_crossfade_sec,
        prepend_context_enabled=not request.disable_prepend_context,
        fail_if_cannot_prepend_context=request.fail_if_cannot_prepend_context,
        min_unmasked_context_sec=request.min_unmasked_context_sec,
        min_unmasked_context_ratio=request.min_unmasked_context_ratio,
        synthesis_mode=request.synthesis_mode,
        debug=request.debug,
        debug_output_dir=debug_output_dir,
        evaluate=bool(evaluate),
    )


def _make_clients(config: PipelineConfig) -> Dict[str, Any]:
    return dict(
        aligner=AlignmentClient(RUNTIME_CONFIG.aligner_url),
        f5=F5TTSClient(RUNTIME_CONFIG.f5_url),
        asr=ASRClient(RUNTIME_CONFIG.asr_url) if config.evaluate else None,
        wer=WERClient(RUNTIME_CONFIG.wer_url) if config.evaluate else None,
        utmos=UTMOSClient(RUNTIME_CONFIG.utmos_url) if config.evaluate else None,
        spk_sim=SpkSimClient(RUNTIME_CONFIG.spk_sim_url) if config.evaluate else None,
        wdtw=WDTWClient(RUNTIME_CONFIG.wdtw_url) if config.evaluate else None,
    )


def _segment_for_request(
    request: ExecuteRequest,
    *,
    audio_bytes: bytes,
    sample_rate: int,
    duration: float,
):
    from types import SimpleNamespace

    metadata: Dict[str, Any] = {}
    if request.language:
        metadata["language"] = request.language
    return SimpleNamespace(
        clip_id=request.clip_id or request.sample_id or "service_request",
        text=request.source_text,
        audio_bytes=audio_bytes,
        duration=duration,
        sample_rate=sample_rate,
        metadata=metadata,
    )


app = FastAPI(
    title="Text Edit Pipeline Service",
    description="Execute structured tagged text edits through the shared pipeline core.",
    version=PIPELINE_VERSION,
)


@app.get("/health")
def health(deep: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "healthy",
        "service": "text_edit_pipeline",
        "pipeline_type": PIPELINE_TYPE,
        "pipeline_version": PIPELINE_VERSION,
    }
    if not deep:
        return payload
    config = _pipeline_config(
        output_dir=DEFAULT_OUTPUT_DIR,
        evaluate=RUNTIME_CONFIG.init_evaluate_services,
    )
    clients = _make_clients(config)
    checks: Dict[str, Any] = {}
    for name, client in clients.items():
        if client is None:
            continue
        try:
            checks[name] = client.health()
        except Exception as exc:
            checks[name] = {"status": "unhealthy", "error": str(exc)}
            payload["status"] = "degraded"
    payload["dependencies"] = checks
    return payload


@app.post("/execute", response_model=ExecuteResponse)
def execute(request: ExecuteRequest) -> ExecuteResponse:
    audio_bytes, sample_rate, duration = _decode_audio(request.audio)
    config = _config_from_request(request)
    clients = _make_clients(config)
    sample_id = request.sample_id or f"service_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    try:
        plan = build_tagged_plan_from_st(
            source_text=request.source_text,
            st=request.instruction_st,
            generation_id=request.generation_id,
            forward_nl=request.instructions_nl,
            reverse_nl=request.reverse_instructions_nl,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        segment = _segment_for_request(
            request,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            duration=duration,
        )
        result = execute_tagged_plan(
            segment=segment,
            sample_id=sample_id,
            plan=plan,
            config=config,
            **clients,
            persist=request.persist,
            include_artifacts=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    generation = result["metadata"]["generations"][request.generation_id]
    generated_bytes = result["generated_audio_bytes"][request.generation_id]
    forward_pair = next(
        pair
        for pair in result["pair_results"]
        if pair.get("direction") == "original_to_generated"
        and pair.get("generation_id") == request.generation_id
    )
    span_plan = generation["span_plan"]
    return ExecuteResponse(
        status="ok",
        pipeline_type=PIPELINE_TYPE,
        pipeline_version=PIPELINE_VERSION,
        sample_id=result["sample_id"],
        generation_id=request.generation_id,
        target_text=render_target(request.instruction_st),
        audio=AudioPayload(
            data=base64.b64encode(generated_bytes).decode("utf-8"),
            format="wav",
        ),
        source_spans=span_plan["source_spans"],
        target_spans=span_plan["target_spans"],
        target_aligned_spans=span_plan["target_aligned_spans"],
        eval_results=forward_pair.get("eval_results", {}),
        metadata=result["metadata"],
        debug=result.get("debug_results", {}).get(request.generation_id),
    )


def main() -> None:
    global RUNTIME_CONFIG
    parser = argparse.ArgumentParser(description="Text edit pipeline service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8035)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--aligner-url", default=RUNTIME_CONFIG.aligner_url)
    parser.add_argument("--f5-url", default=RUNTIME_CONFIG.f5_url)
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
        f5_url=args.f5_url,
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
