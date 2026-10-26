"""FastAPI service entrypoint for prosody edit execution.

Callers provide source audio plus one structured prosody tagged instruction.
The service reuses the shared Stage II-IV pipeline core and returns generated
audio plus edit/evaluation metadata.
"""

from __future__ import annotations

import argparse
import json
import base64
import sys
from dataclasses import dataclass, field
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

from speech_edit.clients.services import (
    AlignmentClient,
    ASRClient,
    F5TTSClient,
    F0EvalClient,
    PitchConversionClient,
    SpkSimClient,
    SpeechRateClient,
    UTMOSClient,
    WERClient,
    WDTWClient,
)

from .. import PIPELINE_TYPE, PIPELINE_VERSION
from ..editor import (
    numpy_to_wav,
    parse_psola_indextts2_cco_pitch_boost,
    wav_to_numpy,
)
from ..input_generator import PipelineInput
from ..pipeline import edit_sample, evaluate_sample
from ..sample_output import save_sample_outputs
from speech_edit.pipelines.prosody.editor import PITCH_METHOD_CHOICES, RATE_METHOD_CHOICES, SMOOTH_POLICY_CHOICES
from ..tag_utils import build_wave_group

DEFAULT_OUTPUT_DIR = "services/" + datetime.now().strftime("%Y-%m-%d") + "/prosody_edit_service"


@dataclass(frozen=True)
class ServiceRuntimeConfig:
    pipeline_parameters: dict = field(default_factory=lambda: load_parameters("service/prosody_edit_pipeline/defaults"))
    aligner_url: str = "http://localhost:8029"
    pitch_url: str = "http://localhost:8027"
    rate_url: str = "http://localhost:8004"
    f5_url: str = "http://localhost:8001"
    asr_url: str = "http://localhost:8010"
    wer_url: str = "http://localhost:8011"
    utmos_url: str = "http://localhost:8017"
    spk_sim_url: str = "http://localhost:8019"
    wdtw_url: str = "http://localhost:8032"
    f0_url: str = "http://localhost:8034"
    init_evaluate_services: bool = False
    output_dir: str = DEFAULT_OUTPUT_DIR


RUNTIME_CONFIG = ServiceRuntimeConfig()


class AudioPayload(BaseModel):
    data: str = Field(..., description="Base64-encoded WAV bytes.")
    format: str = Field("wav", description="Audio format label.")


class NaturalLanguageInstruction(BaseModel):
    precise: str
    vague: str


class ExecuteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio: AudioPayload
    source_text: str
    instruction_st: str
    generation_id: str = "1"
    instructions_nl: Optional[NaturalLanguageInstruction] = None
    reverse_instructions_nl: Optional[NaturalLanguageInstruction] = None
    language: Optional[str] = None
    sample_id: Optional[str] = None
    clip_id: Optional[str] = None
    persist: bool = False
    output_dir: Optional[str] = None
    evaluate: Optional[bool] = None
    aligner_backend: Optional[str] = None
    pitch_method: Optional[str] = None
    rate_method: Optional[str] = None
    psola_indextts2_cco_pitch_boost: Optional[str] = None
    crossfade_ms: Optional[float] = None
    smooth_policy: Optional[str] = None
    smooth_margin_sec: Optional[float] = None


class ExecuteResponse(BaseModel):
    status: str
    pipeline_type: str
    pipeline_version: str
    sample_id: str
    generation_id: str
    source_text: str
    instruction_st: str
    plain_text: str
    audio: AudioPayload
    edit_result: Dict[str, Any]
    edit_pairs: list[Dict[str, Any]]
    eval_results: Dict[str, Any]
    metadata: Dict[str, Any]


def _decode_audio(payload: AudioPayload) -> tuple[bytes, int, float]:
    try:
        raw = base64.b64decode(payload.data)
        waveform, sample_rate = wav_to_numpy(raw)
        wav_bytes = numpy_to_wav(waveform, sample_rate)
        return wav_bytes, sample_rate, len(waveform) / sample_rate
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
    if request.aligner_backend not in {"qwen3"}:
        raise HTTPException(status_code=400, detail="aligner_backend must be 'qwen3'")
    if request.pitch_method not in PITCH_METHOD_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"pitch_method must be one of {list(PITCH_METHOD_CHOICES)}",
        )
    if request.rate_method not in RATE_METHOD_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"rate_method must be one of {list(RATE_METHOD_CHOICES)}",
        )
    try:
        parse_psola_indextts2_cco_pitch_boost(
            request.psola_indextts2_cco_pitch_boost
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"psola_indextts2_cco_pitch_boost: {exc}",
        ) from exc
    if request.smooth_policy not in SMOOTH_POLICY_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"smooth_policy must be one of {list(SMOOTH_POLICY_CHOICES)}",
        )
    if request.crossfade_ms < 0:
        raise HTTPException(status_code=400, detail="crossfade_ms must be non-negative")
    if request.smooth_margin_sec < 0:
        raise HTTPException(status_code=400, detail="smooth_margin_sec must be non-negative")


def _make_services(*, evaluate: bool, smooth_policy: str):
    return {
        "aligner_client": AlignmentClient(RUNTIME_CONFIG.aligner_url),
        "pitch_client": PitchConversionClient(RUNTIME_CONFIG.pitch_url),
        "rate_client": SpeechRateClient(RUNTIME_CONFIG.rate_url),
        "f5_client": F5TTSClient(RUNTIME_CONFIG.f5_url) if smooth_policy != "never" else None,
        "asr_client": ASRClient(RUNTIME_CONFIG.asr_url) if evaluate else None,
        "wer_client": WERClient(RUNTIME_CONFIG.wer_url) if evaluate else None,
        "utmos_client": UTMOSClient(RUNTIME_CONFIG.utmos_url) if evaluate else None,
        "spk_sim_client": SpkSimClient(RUNTIME_CONFIG.spk_sim_url) if evaluate else None,
        "wdtw_client": WDTWClient(RUNTIME_CONFIG.wdtw_url) if evaluate else None,
        "f0_client": F0EvalClient(RUNTIME_CONFIG.f0_url) if evaluate else None,
    }


def _pair_instructions_from_request(request: ExecuteRequest) -> Optional[Dict[str, Any]]:
    if request.instructions_nl is None and request.reverse_instructions_nl is None:
        return None
    forward_key = f"original_to_{request.generation_id}"
    reverse_key = f"{request.generation_id}_to_original"
    return {
        request.generation_id: {
            forward_key: {
                "nl": (
                    request.instructions_nl.model_dump()
                    if request.instructions_nl is not None
                    else {
                        "precise": "Apply the marked pitch/rate edit.",
                        "vague": "Change the marked delivery.",
                    }
                )
            },
            reverse_key: {
                "nl": (
                    request.reverse_instructions_nl.model_dump()
                    if request.reverse_instructions_nl is not None
                    else {
                        "precise": "Restore the marked pitch/rate edit.",
                        "vague": "Return the marked delivery to the original.",
                    }
                )
            },
        }
    }


def _pipeline_input_from_request(
    request: ExecuteRequest,
    *,
    audio_bytes: bytes,
    sample_rate: int,
    duration: float,
    sample_id: str,
) -> PipelineInput:
    metadata: Dict[str, Any] = {
        "input_id": sample_id,
        "source": "service_request",
        "stage0_info": {
            "source": "service_request",
            "clip_id": request.clip_id or sample_id,
        },
    }
    if request.language:
        metadata["language"] = request.language
        metadata["sample_language"] = request.language
    pair_instructions = _pair_instructions_from_request(request)
    if pair_instructions is not None:
        metadata["pair_instructions"] = pair_instructions
    try:
        wave_group = build_wave_group(
            request.source_text,
            {request.generation_id: request.instruction_st},
            metadata=metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PipelineInput(
        clip_id=request.clip_id or sample_id,
        transcript=request.source_text,
        normed_text=request.source_text,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        duration=duration,
        wave_group=wave_group,
        metadata=metadata,
    )


app = FastAPI(
    title="Prosody Edit Pipeline Service",
    description="Execute structured prosody edits through the shared pipeline core.",
    version=PIPELINE_VERSION,
)


@app.get("/health")
def health(deep: bool = False) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "status": "healthy",
        "service": "prosody_edit_pipeline",
        "pipeline_type": PIPELINE_TYPE,
        "pipeline_version": PIPELINE_VERSION,
    }
    if not deep:
        return payload
    evaluate = RUNTIME_CONFIG.init_evaluate_services
    smooth_policy = RUNTIME_CONFIG.pipeline_parameters["smooth_policy"]
    services = _make_services(evaluate=evaluate, smooth_policy=smooth_policy)
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
    request = resolve_request_parameters(request, RUNTIME_CONFIG.pipeline_parameters, ('aligner_backend', 'pitch_method', 'rate_method', 'psola_indextts2_cco_pitch_boost', 'crossfade_ms', 'smooth_policy', 'smooth_margin_sec'))
    _validate_request_controls(request)
    audio_bytes, sample_rate, duration = _decode_audio(request.audio)
    evaluate = _evaluation_enabled_for_request(request)
    sample_id = request.sample_id or f"service_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = request.output_dir or RUNTIME_CONFIG.output_dir
    if request.persist:
        output_dir = str(resolve_output_path(output_dir))
        resolve_output_path(Path(output_dir) / sample_id)
    pipeline_input = _pipeline_input_from_request(
        request,
        audio_bytes=audio_bytes,
        sample_rate=sample_rate,
        duration=duration,
        sample_id=sample_id,
    )
    services = _make_services(evaluate=evaluate, smooth_policy=request.smooth_policy)
    try:
        stage2_state = edit_sample(
            pipeline_input,
            aligner_client=services["aligner_client"],
            aligner_backend=request.aligner_backend,
            pitch_method=request.pitch_method,
            rate_method=request.rate_method,
            psola_indextts2_cco_pitch_boost=request.psola_indextts2_cco_pitch_boost,
            pitch_client=services["pitch_client"],
            rate_client=services["rate_client"],
            f5_client=services["f5_client"],
            crossfade_ms=request.crossfade_ms,
            smooth_margin_sec=request.smooth_margin_sec,
            smooth_policy=request.smooth_policy,
            f5_steps=RUNTIME_CONFIG.pipeline_parameters["f5_steps"],
            f5_seed=RUNTIME_CONFIG.pipeline_parameters["f5_seed"],
            f5_source_crossfade_sec=RUNTIME_CONFIG.pipeline_parameters["f5_source_crossfade_sec"],
            f5_regen_crossfade_sec=RUNTIME_CONFIG.pipeline_parameters["f5_regen_crossfade_sec"],
            t_start=datetime.now().timestamp(),
        )
        stage3_state = evaluate_sample(
            stage2_state,
            asr_client=services["asr_client"],
            wer_client=services["wer_client"],
            utmos_client=services["utmos_client"],
            spk_sim_client=services["spk_sim_client"],
            f0_client=services["f0_client"],
            wdtw_client=services["wdtw_client"],
            do_evaluate=evaluate,
        )
        metadata = stage3_state["metadata"]
        if request.persist:
            save_sample_outputs(
                str(Path(output_dir) / stage3_state["sample_id"]),
                original_audio_bytes=stage3_state["pipeline_input"].audio_bytes,
                wave_audio_bytes=stage3_state["wave_audio_bytes"],
                pair_results=stage3_state["pair_results"],
                metadata=metadata,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    wave_result = stage2_state["wave_results"][request.generation_id]
    generated_bytes = stage2_state["wave_audio_bytes"][request.generation_id]
    forward_pair = next(
        pair
        for pair in stage3_state["pair_results"]
        if pair.get("source_id") == "original" and pair.get("target_id") == request.generation_id
    )
    edit_pairs = [
        {key: value for key, value in pair.items() if key not in {"source_audio_bytes", "target_audio_bytes"}}
        for pair in stage3_state["pair_results"]
    ]
    return ExecuteResponse(
        status="ok",
        pipeline_type=PIPELINE_TYPE,
        pipeline_version=PIPELINE_VERSION,
        sample_id=sample_id,
        generation_id=request.generation_id,
        source_text=request.source_text,
        instruction_st=request.instruction_st,
        plain_text=pipeline_input.wave_group.plain_text,
        audio=AudioPayload(
            data=base64.b64encode(generated_bytes).decode("utf-8"),
            format="wav",
        ),
        edit_result=wave_result["result"],
        edit_pairs=edit_pairs,
        eval_results=forward_pair.get("eval_results", {}),
        metadata=metadata,
    )


def main() -> None:
    global RUNTIME_CONFIG
    parser = argparse.ArgumentParser(description="Pitch/rate edit pipeline service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8036)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--aligner-url", default=RUNTIME_CONFIG.aligner_url)
    parser.add_argument("--pitch-url", default=RUNTIME_CONFIG.pitch_url)
    parser.add_argument("--rate-url", default=RUNTIME_CONFIG.rate_url)
    parser.add_argument("--f5-url", default=RUNTIME_CONFIG.f5_url)
    parser.add_argument("--asr-url", default=RUNTIME_CONFIG.asr_url)
    parser.add_argument("--wer-url", default=RUNTIME_CONFIG.wer_url)
    parser.add_argument("--utmos-url", default=RUNTIME_CONFIG.utmos_url)
    parser.add_argument("--spk-sim-url", default=RUNTIME_CONFIG.spk_sim_url)
    parser.add_argument("--wdtw-url", default=RUNTIME_CONFIG.wdtw_url)
    parser.add_argument("--f0-url", default=RUNTIME_CONFIG.f0_url)
    parser.add_argument("--output-dir", default=RUNTIME_CONFIG.output_dir)
    parser.add_argument("--init-evaluate-services", action="store_true")
    parser.add_argument("--pipeline-defaults", required=True)
    parser.add_argument("--pipeline-parameters", type=json.loads, required=True)
    args = parser.parse_args()
    RUNTIME_CONFIG = ServiceRuntimeConfig(
        pipeline_parameters=load_parameters(args.pipeline_defaults, args.pipeline_parameters),
        aligner_url=args.aligner_url,
        pitch_url=args.pitch_url,
        rate_url=args.rate_url,
        f5_url=args.f5_url,
        asr_url=args.asr_url,
        wer_url=args.wer_url,
        utmos_url=args.utmos_url,
        spk_sim_url=args.spk_sim_url,
        f0_url=args.f0_url,
        wdtw_url=args.wdtw_url,
        init_evaluate_services=args.init_evaluate_services,
        output_dir=str(resolve_output_path(args.output_dir)),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
