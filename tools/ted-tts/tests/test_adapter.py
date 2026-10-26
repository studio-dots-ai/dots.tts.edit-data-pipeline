"""Behavior contracts for the service's adaptations of official TED-TTS."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import adapter


def test_imports_official_code_and_installs_only_service_adapters():
    assert Path(adapter.upstream_infer.__file__).resolve().is_relative_to(adapter.UPSTREAM)
    assert Path(adapter.upstream_gpt.__file__).resolve().is_relative_to(adapter.UPSTREAM)
    assert adapter.upstream_infer.UnifiedVoice is adapter.UnifiedVoice
    assert adapter.upstream_gpt.GPT2InferenceModel is adapter.GPT2InferenceModel
    assert adapter._bigvgan_loader.load is adapter.load_bigvgan_cuda
    assert adapter._s2mel_loader.load is adapter.load_s2mel_cuda


def test_conditioning_swaps_only_the_two_duration_slots(monkeypatch):
    original_class = adapter.UnifiedVoice.__mro__[1]
    captured = {}
    def prepare(self, latents, text, full_text=False):
        captured.update(latents=latents, text=text, full_text=full_text)
        return "prepared"
    monkeypatch.setattr(original_class, "prepare_gpt_inputs", prepare)
    model = adapter.UnifiedVoice.__new__(adapter.UnifiedVoice)
    latents = [torch.arange(24).reshape(2, 4, 3), torch.arange(30).reshape(2, 5, 3)]
    before = [value.clone() for value in latents]
    text = [torch.tensor([[1, 2]])]
    assert model.prepare_gpt_inputs(latents, text, True) == "prepared"
    assert captured["text"] is text
    assert captured["full_text"] is True
    for source, unchanged, actual in zip(latents, before, captured["latents"]):
        assert torch.equal(source, unchanged)
        assert torch.equal(actual[:, :-2], source[:, :-2])
        assert torch.equal(actual[:, -2], source[:, -1])
        assert torch.equal(actual[:, -1], source[:, -2])


def test_generation_resets_state_and_preserves_upstream_arguments(monkeypatch):
    original_class = adapter.UnifiedVoice.__mro__[1]
    calls = []
    def inference(self, condition, text, target_duration_tokens=None, **kwargs):
        assert self.inference_model.duration_processor is None
        assert self.inference_model.duration_controller is None
        assert self.inference_model.segment_positions is None
        calls.append((condition, text, kwargs))
        return "generated"
    monkeypatch.setattr(original_class, "inference_speech", inference)
    model = adapter.UnifiedVoice.__new__(adapter.UnifiedVoice)
    torch.nn.Module.__init__(model)
    model.inference_model = SimpleNamespace(duration_processor=1, duration_controller=2, segment_positions={0: [3]})
    assert model.inference_speech("audio", "text", num_beams=3, temperature=0.8) == "generated"
    assert calls == [("audio", "text", {"num_beams": 3, "temperature": 0.8})]
    with pytest.raises(RuntimeError, match="duration-control.*disabled"):
        model.inference_speech("audio", "text", [20])
    assert len(calls) == 1


@pytest.mark.parametrize("positions", [{2: [7]}, {}, {1: [3], 2: [7]}, None])
def test_finished_generation_discards_unused_beam_statistics(monkeypatch, positions):
    original_class = adapter.GPT2InferenceModel.__mro__[1]
    output = SimpleNamespace(beam_indices=torch.tensor([[1]]), sequences=torch.tensor([[11, 12]]))
    monkeypatch.setattr(original_class, "generate", lambda self, *args, **kwargs: output)
    model = adapter.GPT2InferenceModel.__new__(adapter.GPT2InferenceModel)
    torch.nn.Module.__init__(model)
    model.segment_positions = positions
    assert model.generate() is output
    assert model.segment_positions is None


@pytest.mark.parametrize("device", ["cuda:1", "cpu", "mps"])
def test_emotion_model_is_loaded_on_the_requested_device(device):
    calls = []
    model = SimpleNamespace(to=lambda destination: calls.append(destination) or "mps-model")
    loader = SimpleNamespace(from_pretrained=lambda path, **kwargs: calls.append((path, kwargs)) or model)
    result = adapter._EmotionModelLoader(loader, device).from_pretrained(
        "/models/emotion", torch_dtype="float16", device_map="auto",
    )
    expected = {"torch_dtype": "float16"}
    if device != "mps":
        expected["device_map"] = {"": device}
    assert calls[0] == ("/models/emotion", expected)
    assert result == ("mps-model" if device == "mps" else model)
    assert calls[1:] == (["mps"] if device == "mps" else [])


def test_auto_device_does_not_change_upstream_cuda_kernel_defaults(monkeypatch):
    original_class = adapter.IndexTTS2.__mro__[1]
    original_loader = adapter.upstream_infer.AutoModelForCausalLM
    captured = {}
    def initialize(self, **kwargs):
        self.use_cuda_kernel = True
        captured.update(kwargs)
        assert adapter.upstream_infer.AutoModelForCausalLM.device == "cuda:0"
    monkeypatch.setattr(original_class, "__init__", initialize)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    adapter.IndexTTS2("/models/config.yaml", "/models", is_fp16=True)
    assert captured == dict(cfg_path="/models/config.yaml", model_dir="/models", is_fp16=True, device=None, use_cuda_kernel=None)
    assert adapter.upstream_infer.AutoModelForCausalLM is original_loader


def test_model_cache_import_is_not_redirected_to_checkout():
    from huggingface_hub import constants
    if os.environ.get("HF_HUB_CACHE"):
        assert constants.HF_HUB_CACHE == os.environ["HF_HUB_CACHE"]
    assert not Path(constants.HF_HUB_CACHE).resolve().is_relative_to(adapter.TOOL_ROOT)


@pytest.mark.parametrize("device,use_cuda_kernel", [("cuda:0", True), (None, None), ("cpu", True)])
def test_requested_cuda_kernel_cannot_silently_fall_back(monkeypatch, device, use_cuda_kernel):
    original_class = adapter.IndexTTS2.__mro__[1]
    def initialize(self, **kwargs):
        self.use_cuda_kernel = False
    monkeypatch.setattr(original_class, "__init__", initialize)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="requested the CUDA kernel"):
        adapter.IndexTTS2("/models/config.yaml", "/models", device=device, use_cuda_kernel=use_cuda_kernel)


@pytest.mark.parametrize("device,use_cuda_kernel", [("cuda:0", None), (None, False), ("cpu", None)])
def test_upstream_torch_kernel_choice_is_preserved(monkeypatch, device, use_cuda_kernel):
    original_class = adapter.IndexTTS2.__mro__[1]
    def initialize(self, **kwargs):
        self.use_cuda_kernel = False
    monkeypatch.setattr(original_class, "__init__", initialize)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = adapter.IndexTTS2("/models/config.yaml", "/models", device=device, use_cuda_kernel=use_cuda_kernel)
    assert model.use_cuda_kernel is False
