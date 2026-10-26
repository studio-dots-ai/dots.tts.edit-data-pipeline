import base64
import io
import json
from types import SimpleNamespace

import pytest
from pathlib import Path

import soundfile as sf
from fastapi.testclient import TestClient

from speech_edit.pipelines.pause.parquet_rows import PauseEditParquetRowProcessor
from speech_edit.pipelines.pause.evaluator import PauseEvalResult
from speech_edit.pipelines.pause.service import api_service
from speech_edit.pipelines.pause.synthesizer import SynthesizedPair


def _wav_payload(duration_sec: float = 1.0, sample_rate: int = 24000) -> dict:
    samples = [0.0] * int(duration_sec * sample_rate)
    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    return {
        "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "format": "wav",
    }


def _fake_services(*, evaluate=False):
    class _FakeClient:
        def health(self):
            return {"status": "ok"}

    return {
        "aligner_client": _FakeClient(),
        "smoother_client": _FakeClient(),
        "asr_client": _FakeClient() if evaluate else None,
        "wer_client": _FakeClient() if evaluate else None,
        "utmos_client": _FakeClient() if evaluate else None,
        "spk_sim_client": _FakeClient() if evaluate else None,
        "wdtw_client": _FakeClient() if evaluate else None,
    }


def _fake_synth_pair(profile, normed_text: str) -> SynthesizedPair:
    return SynthesizedPair(
        profile=profile,
        normed_text=normed_text,
        generated_text=normed_text,
        original_audio_bytes=b"orig-wav",
        generated_audio_bytes=b"gen-wav",
        original_duration_sec=1.0,
        generated_duration_sec=1.5,
        pause_edit_plan={
            "mode": "single_region_smooth_only",
            "pause_duration_sec": [0.5],
        },
        metadata={
            "synth_method": "single_region",
            "aligner_backend": "qwen3",
        },
    )


def test_execute_pause_service_generation_only(monkeypatch):
    client = TestClient(api_service.app)

    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda evaluate: _fake_services(
            evaluate=evaluate,
        ),
    )
    monkeypatch.setattr(
        api_service,
        "synthesize_pause_edit",
        lambda **kwargs: _fake_synth_pair(kwargs["profile"], kwargs["group"].normed_text),
    )

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> world.',
            "evaluate": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["pipeline_type"] == "pause_edit"
    assert data["generated_text"] == "Hello world."
    assert data["pause_edit_plan"]["mode"] == "single_region_smooth_only"
    assert data["eval_results"] == {}
    assert data["metadata"]["sample_dir"] is None


def test_execute_pause_service_accepts_multi_pause_instruction(monkeypatch):
    client = TestClient(api_service.app)

    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda evaluate: _fake_services(
            evaluate=evaluate,
        ),
    )
    monkeypatch.setattr(
        api_service,
        "synthesize_pause_edit",
        lambda **kwargs: _fake_synth_pair(kwargs["profile"], kwargs["group"].normed_text),
    )

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world, again.",
            "instruction_st": 'Hello<pause act="ins" level="1"/> world,<pause act="ins" level="2"/> again.',
            "evaluate": False,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["metadata"]["stage1_info"]["num_operations"] == 2


def test_execute_pause_service_accepts_reduction_and_mixed_actions(monkeypatch):
    client = TestClient(api_service.app)
    captured = {}

    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda evaluate: _fake_services(
            evaluate=evaluate,
        ),
    )

    def fake_synthesize(**kwargs):
        captured["actions"] = [tag.act for tag in kwargs["profile"].tags]
        captured["st_inst"] = kwargs["profile"].st_inst
        captured["nl_inst"] = kwargs["profile"].nl_inst
        return _fake_synth_pair(kwargs["profile"], kwargs["group"].normed_text)

    monkeypatch.setattr(api_service, "synthesize_pause_edit", fake_synthesize)
    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world, again.",
            "instruction_st": (
                'Hello<pause act="red" level="1"/> world,'
                '<pause act="ins" level="2"/> again.'
            ),
            "evaluate": False,
        },
    )

    assert response.status_code == 200
    assert captured["actions"] == ["red", "ins"]
    assert captured["st_inst"]["increase_pause"] == []
    assert captured["st_inst"]["decrease_pause"] == []
    assert captured["st_inst"]["mixed_pause"] == [
        (
            'Hello<pause act="red" level="1"/> world,'
            '<pause act="ins" level="2"/> again.'
        )
    ]
    assert captured["nl_inst"]["mixed_pause"][0].startswith(
        "Edit the pause timing:"
    )
    assert response.json()["metadata"]["stage1_info"]["num_operations"] == 2


def test_reduction_profile_metadata_uses_decrease_direction():
    operation = 'Hello<pause act="red" level="2"/> world.'
    request = api_service.ExecuteRequest.model_validate(
        {
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": operation,
        }
    )

    profile = api_service._profile_from_request(request)

    assert profile.st_inst["decrease_pause"] == [operation]
    assert profile.st_inst["increase_pause"] == [
        'Hello<pause act="ins" level="2"/> world.'
    ]
    assert profile.nl_inst["decrease_pause"][0].startswith(
        "Reduce the pause timing:"
    )


def test_execute_pause_service_rejects_invalid_instruction():
    client = TestClient(api_service.app)

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> there.',
        },
    )

    assert response.status_code == 400
    assert "Text mismatch after stripping tags" in response.json()["detail"]


def test_execute_pause_service_rejects_evaluate_when_disabled(monkeypatch):
    client = TestClient(api_service.app)

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> world.',
            "evaluate": True,
        },
    )

    assert response.status_code == 400
    assert "launched without evaluation models" in response.json()["detail"]


def test_execute_pause_service_rejects_retired_save_parquet_field():
    client = TestClient(api_service.app)

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> world.',
            "save_parquet": True,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "save_parquet"]


def test_execute_pause_service_with_eval_and_persistence(monkeypatch, tmp_path):
    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(
            init_evaluate_services=True,
        ),
    )
    client = TestClient(api_service.app)

    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda evaluate: _fake_services(
            evaluate=evaluate,
        ),
    )
    monkeypatch.setattr(
        api_service,
        "synthesize_pause_edit",
        lambda **kwargs: _fake_synth_pair(kwargs["profile"], kwargs["group"].normed_text),
    )
    monkeypatch.setattr(api_service, "save_pause_edit_group", lambda group, output_dir, sample_id: str(tmp_path / sample_id))
    monkeypatch.setattr(api_service, "save_synthesis_result", lambda synth_pair, eval_result, sample_dir, profile_id: str(Path(sample_dir) / f"profile_{profile_id}"))
    captured_eval_kwargs = {}

    def fake_evaluate_pause_edit(**kwargs):
        captured_eval_kwargs.update(kwargs)
        return PauseEvalResult(
            profile_id=1,
            original_wer=0.01,
            generated_wer=0.03,
            content_passed=True,
            pause_measurements=[
                {
                    "target_added_pause_ms": 500,
                    "actual_added_pause_ms": 430,
                    "target_vs_actual_gap_error_ms": 70,
                }
            ],
            metadata={"original_transcription": "hello world", "generated_transcription": "hello world", "original_wer_result": {"error_rate": 0.01, "metric": "mixed"}, "generated_wer_result": {"error_rate": 0.03, "metric": "mixed"}},
        )

    monkeypatch.setattr(api_service, "evaluate_pause_edit", fake_evaluate_pause_edit)

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> world.',
            "persist": True,
            "evaluate": True,
            "output_dir": str(tmp_path / "outputs"),
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["eval_results"]["tgt"]["wer"]["error_rate"] == 0.03
    assert data["eval_results"]["pause"]["operations"][0][
        "expected_tgt_minus_src_pause_gap_ms"
    ] == 500.0
    assert data["metadata"]["sample_dir"].endswith(data["sample_id"])
    assert data["metadata"]["profile_dir"].endswith("profile_1")
    assert "save_parquet" not in data["metadata"]
    assert captured_eval_kwargs["utmos_client"] is not None


def test_execute_pause_service_rejects_unknown_fields():
    client = TestClient(api_service.app)

    response = client.post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "Hello world.",
            "instruction_st": 'Hello<pause act="ins" level="2"/> world.',
            "operations": [],
        },
    )

    assert response.status_code == 422


def test_pause_service_deep_health_reports_degraded(monkeypatch):
    client = TestClient(api_service.app)
    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(init_evaluate_services=True),
    )

    class _Healthy:
        def health(self):
            return {"status": "ok"}

    class _Broken:
        def health(self):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda evaluate: {
            "aligner_client": _Healthy(),
            "smoother_client": _Broken(),
            "asr_client": _Healthy() if evaluate else None,
            "wer_client": _Healthy() if evaluate else None,
            "utmos_client": _Healthy() if evaluate else None,
            "spk_sim_client": _Healthy() if evaluate else None,
            "wdtw_client": _Healthy() if evaluate else None,
        },
    )

    response = client.get("/health?deep=true")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["dependencies"]["smoother"]["status"] == "unhealthy"


@pytest.mark.parametrize(
    ("source_text", "instruction", "expected_positions"),
    [
        (
            "One two three four.",
            'One <pause act="ins" level="1"/> two <pause act="ins" level="2"/> '
            'three <pause act="ins" level="3"/> four.',
            [3, 7, 13],
        ),
        (
            "我们今天一起出发。",
            '我们 今天 <pause act="ins" level="1"/> 一起出发。',
            [4],
        ),
        (
            "  One   two three.",
            'One two <pause act="ins" level="1"/> three.',
            [11],
        ),
    ],
)
def test_execute_and_parquet_share_reference_coordinates(
    monkeypatch, source_text, instruction, expected_positions,
):
    captured = {}
    wav = base64.b64decode(_wav_payload()["data"])

    def synthesize(**kwargs):
        profile = kwargs["profile"]
        group = kwargs["group"]
        assert [tag.char_idx for tag in profile.tags] == expected_positions
        pair = _fake_synth_pair(profile, group.normed_text)
        pair.original_audio_bytes = wav
        pair.generated_audio_bytes = wav
        captured.update(profile=profile, group=group, pair=pair)
        return pair

    monkeypatch.setattr(api_service, "_make_services", _fake_services)
    monkeypatch.setattr(api_service, "synthesize_pause_edit", synthesize)
    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": source_text,
            "instruction_st": instruction,
            "evaluate": False,
        },
    )
    assert response.status_code == 200, response.text
    work_item = SimpleNamespace(
        sample_id="reference-coordinates",
        profile=captured["profile"],
        normed_text=source_text,
        metadata={"sample_rate": 24000, "language": "en"},
    )
    rows = PauseEditParquetRowProcessor().rows_from_payload(
        SimpleNamespace(work_item=work_item, synth_pair=captured["pair"], eval_result=None),
        dataset_name="Emilia-Pause-Test",
    )
    assert len(rows) == 2
    for row, action in zip(rows, ("ins", "red")):
        operations = json.loads(row["edit_metadata"])["operations"]
        assert [item["char_idx"] for item in operations] == expected_positions
        assert [item["act"] for item in operations] == [action] * len(expected_positions)


def test_execute_rejects_contraction_change_before_synthesis(monkeypatch):
    def unexpected_synthesis(**kwargs):
        pytest.fail("A rewritten instruction must not reach synthesis")

    monkeypatch.setattr(api_service, "synthesize_pause_edit", unexpected_synthesis)
    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": _wav_payload(),
            "source_text": "I am a big cat.",
            "instruction_st": 'I\'m a<pause act="ins" level="1"/> big cat.',
            "evaluate": False,
        },
    )
    assert response.status_code == 400
    assert "Text mismatch" in response.json()["detail"]
