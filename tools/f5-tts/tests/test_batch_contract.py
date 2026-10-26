import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from f5_service import synthesis_service as service


def test_batch_rejects_wav_before_inference_and_does_not_mutate_request(monkeypatch):
    monkeypatch.setattr(service, "MODEL_OBJ", object())
    async def unexpected(_):
        pytest.fail("Invalid batch must fail before inference")
    monkeypatch.setattr(service, "sample", unexpected)
    item = SimpleNamespace(output_format="wav")
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.sample_batch(SimpleNamespace(requests=[item])))
    assert error.value.status_code == 400
    assert item.output_format == "wav"


def test_batch_propagates_item_failure_and_stops(monkeypatch):
    monkeypatch.setattr(service, "MODEL_OBJ", object())
    calls = []
    async def fail(item):
        calls.append(item)
        raise HTTPException(503, "model unavailable")
    monkeypatch.setattr(service, "sample", fail)
    items = [SimpleNamespace(output_format="both") for _ in range(2)]
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.sample_batch(SimpleNamespace(requests=items)))
    assert error.value.status_code == 503
    assert calls == items[:1]


@pytest.mark.parametrize("preserve", [True, False])
def test_sample_duration_control_and_waveform_preservation(monkeypatch, preserve):
    import numpy as np
    import torch

    calls = []
    def model_sample(**kwargs):
        calls.append(kwargs)
        return kwargs["cond"].clone(), None

    monkeypatch.setattr(service, "DEVICE", "cpu")
    monkeypatch.setattr(service, "MODEL_OBJ", SimpleNamespace(sample=model_sample))
    source = np.linspace(-0.5, 0.5, 5 * service.HOP_LENGTH, dtype=np.float32)
    monkeypatch.setattr(service, "vocode_mel_to_numpy", lambda _: np.zeros_like(source))
    request = service.SampleRequest(
        cond=service.encode_tensor(torch.zeros(1, 5, 2)),
        text=[["a"]], duration=5, steps=2, seed=42,
        edit_mask=service.encode_tensor(torch.tensor([[True, False, True, True, True]])),
        preserve_unmasked_waveform=preserve,
        preserve_waveform=service.encode_numpy(source), output_format="numpy",
    )
    response = asyncio.run(service.sample(request))
    assert calls[0]["duration"] == 5
    assert calls[0]["steps"] == 2 and calls[0]["seed"] == 42
    assert "allow_duration_auto_extend" not in calls[0]
    if preserve:
        assert calls[0]["max_duration"] == 5
        output = service.decode_numpy(response.tensor)
        assert output.shape == source.shape
        assert np.array_equal(output[:service.HOP_LENGTH], source[:service.HOP_LENGTH])
        assert np.array_equal(output[2 * service.HOP_LENGTH:], source[2 * service.HOP_LENGTH:])
        assert np.all(output[service.HOP_LENGTH:2 * service.HOP_LENGTH] == 0)
    else:
        assert "max_duration" not in calls[0]


@pytest.mark.parametrize("duration, message", [
    (4, "must be >= conditioning length"),
    (4097, "exceeds the single-call duration limit"),
])
def test_invalid_preserve_duration_is_rejected_before_inference(monkeypatch, duration, message):
    import numpy as np
    import torch

    def unexpected(**kwargs):
        pytest.fail("Invalid duration must not reach model inference")
    monkeypatch.setattr(service, "DEVICE", "cpu")
    monkeypatch.setattr(service, "MODEL_OBJ", SimpleNamespace(sample=unexpected))
    request = service.SampleRequest(
        cond=service.encode_tensor(torch.zeros(1, 5, 2)),
        text=[["a"]], duration=duration,
        edit_mask=service.encode_tensor(torch.tensor([[True, False, True, True, True]])),
        preserve_unmasked_waveform=True,
        preserve_waveform=service.encode_numpy(np.zeros(5 * service.HOP_LENGTH, dtype=np.float32)),
        output_format="numpy",
    )
    with pytest.raises(HTTPException) as error:
        asyncio.run(service.sample(request))
    assert error.value.status_code == 400
    assert message in error.value.detail
