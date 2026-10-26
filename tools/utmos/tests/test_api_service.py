from pathlib import Path

from tools.utmos import api_service


class _FakeModel:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self


def test_load_model_uses_torch_hub_skip_validation_and_lock(tmp_path, monkeypatch):
    hub_dir = tmp_path / "torch_hub"
    calls = []

    def fake_load(*args, **kwargs):
        calls.append((args, kwargs))
        lock_path = hub_dir / ".utmos_load.lock"
        assert lock_path.exists()
        return _FakeModel()

    monkeypatch.setattr(api_service.torch.hub, "get_dir", lambda: str(hub_dir))
    monkeypatch.setattr(api_service.torch.hub, "load", fake_load)

    api_service.load_model("cpu")

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == ("tarepan/SpeechMOS:v1.2.0", "utmos22_strong")
    assert kwargs["trust_repo"] is True
    assert kwargs["skip_validation"] is True
    assert isinstance(api_service.model, _FakeModel)
    assert api_service.model.device == "cpu"
    assert api_service.model.eval_called is True
    assert Path(api_service.torch.hub.get_dir(), ".utmos_load.lock").exists()


def test_batch_fails_instead_of_returning_sentinel_score(monkeypatch):
    import pytest
    from fastapi import HTTPException

    monkeypatch.setattr(api_service, "model", object())
    def fail(_):
        raise RuntimeError("inference failed")
    monkeypatch.setattr(api_service, "predict_single", fail)
    with pytest.raises(HTTPException) as error:
        api_service.predict_batch(api_service.PredictBatchRequest(audio_list=["YQ=="]))
    assert error.value.status_code == 500
    assert error.value.detail == "inference failed"
