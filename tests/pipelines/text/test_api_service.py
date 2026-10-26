"""Contract checks for the structured text editing service."""

import base64

import numpy as np
import pytest
from fastapi.testclient import TestClient

from speech_edit.pipelines.text.audio_timeline import mono_float_to_wav_bytes
from speech_edit.pipelines.text.service import api_service


def _payload():
    audio = mono_float_to_wav_bytes(np.zeros(2400, dtype=np.float32), 24000)
    return {
        "audio": {"data": base64.b64encode(audio).decode("ascii")},
        "source_text": "Hello world.",
        "instruction_st": 'Hello <sub targ="everyone">world</sub>.',
    }


@pytest.mark.parametrize("options", [{"save_parquet": True}, {"parquet_output_dir": "output/parquet"}])
def test_execute_rejects_unavailable_parquet_output(options, monkeypatch):
    monkeypatch.setattr(api_service, "_make_clients", lambda config: pytest.fail("Must reject before starting services"))
    response = TestClient(api_service.app).post("/execute", json={**_payload(), **options})
    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_agent_request_preserves_execution_defaults():
    request = api_service.ExecuteRequest(**_payload())
    config = api_service._config_from_request(request)
    assert request.duration_mode is None
    assert config.duration_mode == api_service.RUNTIME_CONFIG.pipeline_parameters["duration_mode"]
    assert config.merge_gap == api_service.RUNTIME_CONFIG.pipeline_parameters["merge_gap"]
    assert config.aligner_backend == "qwen3"


def test_persist_path_resolves_under_artifact_root(tmp_path, monkeypatch):
    from speech_edit.runtime.paths import get_artifact_paths

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    request = api_service.ExecuteRequest(**_payload(), persist=True,
                                        output_dir="requests")
    config = api_service._config_from_request(request)
    root = get_artifact_paths().worktree
    assert config.output_dir == str(root / "requests")
    assert not root.exists()


def test_memory_only_request_needs_no_artifact_root(monkeypatch):
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    request = api_service.ExecuteRequest(**_payload(), persist=False, debug=False)
    api_service._config_from_request(request)


def test_persist_rejects_sample_id_escaping_artifact_root(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    request = api_service.ExecuteRequest(**_payload(), persist=True,
                                        sample_id=str(tmp_path / "outside"))
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        api_service._config_from_request(request)
    assert not (tmp_path / "outside").exists()




def test_execution_parameters_inherit_yaml_and_preserve_explicit_values(monkeypatch):
    from speech_edit.pipelines.text.service import api_service as api

    parameters = dict(api.RUNTIME_CONFIG.pipeline_parameters)
    parameters.update(f5_steps=19, expand_radius=0.27, prepend_context_enabled=False)
    monkeypatch.setattr(api, "RUNTIME_CONFIG", api.ServiceRuntimeConfig(pipeline_parameters=parameters))
    payload = dict(audio={"data": "unused"}, source_text="Hello", instruction_st="Hello")
    for overrides in ({}, {"nfe_step": None, "expand_radius": None}):
        config = api._config_from_request(api.ExecuteRequest(**payload, **overrides))
        assert config.f5_steps == 19
        assert config.expand_radius == 0.27
        assert config.prepend_context_enabled is False
        assert config.evaluate is False
    config = api._config_from_request(api.ExecuteRequest(
        **payload, nfe_step=7, expand_radius=0.0, disable_prepend_context=False,
    ))
    assert config.f5_steps == 7
    assert config.expand_radius == 0.0
    assert config.prepend_context_enabled is True
    del parameters["f5_steps"]
    with pytest.raises(KeyError, match="f5_steps"):
        api._config_from_request(api.ExecuteRequest(**payload))


@pytest.mark.parametrize("evaluate", [False, True])
def test_http_execution_forwards_direct_clients_and_health_checks(monkeypatch, evaluate):
    import inspect
    from types import SimpleNamespace

    monkeypatch.setattr(api_service, "RUNTIME_CONFIG", api_service.ServiceRuntimeConfig(
        init_evaluate_services=evaluate,
    ))
    created = {}
    checked = []
    def client(url):
        def health():
            checked.append(url)
            return {"status": "healthy"}
        created[url] = SimpleNamespace(base_url=url, health=health)
        return created[url]
    for name in ("AlignmentClient", "F5TTSClient", "ASRClient", "WERClient",
                 "UTMOSClient", "SpkSimClient", "WDTWClient"):
        monkeypatch.setattr(api_service, name, client)
    signature = inspect.signature(api_service.execute_tagged_plan)
    def execute(**kwargs):
        signature.bind(**kwargs)
        assert kwargs["config"].evaluate is evaluate
        for key in ("aligner", "f5", "asr", "wer", "utmos", "spk_sim", "wdtw"):
            if key in {"aligner", "f5"} or evaluate:
                url = getattr(api_service.RUNTIME_CONFIG, key + "_url")
                assert kwargs[key] is created[url]
            else:
                assert kwargs[key] is None
        generation_id = kwargs["plan"].generations[0].generation_id
        return {
            "sample_id": kwargs["sample_id"],
            "metadata": {"generations": {generation_id: {"span_plan": {
                "source_spans": [], "target_spans": [], "target_aligned_spans": [],
            }}}},
            "generated_audio_bytes": {generation_id: kwargs["segment"].audio_bytes},
            "pair_results": [{"direction": "original_to_generated", "generation_id": generation_id}],
        }
    monkeypatch.setattr(api_service, "execute_tagged_plan", execute)
    http = TestClient(api_service.app)
    response = http.post("/execute", json={**_payload(), "evaluate": evaluate, "persist": False})
    assert response.status_code == 200, response.text
    assert response.json()["audio"]["data"] == _payload()["audio"]["data"]
    response = http.get("/health?deep=true")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert set(checked) == set(created)
    assert len(checked) == (7 if evaluate else 2)
