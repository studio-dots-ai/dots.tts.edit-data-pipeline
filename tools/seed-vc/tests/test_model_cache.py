"""Verify that model resolution honors the configured cache."""
from pathlib import Path


def test_default_model_files_use_configured_cache(tmp_path, monkeypatch):
    """Exercise model resolution before allocating or loading inference weights."""
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace
    import pytest

    path = Path(__file__).resolve().parents[1] / "seed_vc_service.py"
    spec = importlib.util.spec_from_file_location("seedvc_cache_contract", path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    config = tmp_path / "model.yml"
    config.write_text("model_params: {}\n")
    calls = []

    def cached_file(repo_id, filename, *, cache_dir):
        calls.append((repo_id, filename, cache_dir))
        return str(config)

    class ModelResolutionFinished(Exception):
        pass

    def stop_before_weights(*args, **kwargs):
        raise ModelResolutionFinished

    commons = ModuleType("modules.commons")
    commons.recursive_munch = lambda value: SimpleNamespace(**value)
    commons.build_model = stop_before_weights
    commons.load_checkpoint = stop_before_weights
    monkeypatch.setitem(sys.modules, "modules.commons", commons)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", cached_file)
    for variable,expected in [("HF_HUB_CACHE",str(tmp_path/'hub')), ("CHECKPOINTS_DIR",str(tmp_path/'weights')), (None,None)]:
        monkeypatch.delenv("HF_HUB_CACHE",raising=False)
        monkeypatch.delenv("CHECKPOINTS_DIR",raising=False)
        if variable:
            monkeypatch.setenv(variable,expected)
        calls.clear()
        with pytest.raises(ModelResolutionFinished):
            service.load_models("cpu",True,False,None,None)
        assert calls == [
            ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", expected),
            ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", expected),
        ]


def test_auxiliary_model_files_use_configured_cache(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace
    import pytest

    path = Path(__file__).resolve().parents[1] / "seed_vc_service.py"
    spec = importlib.util.spec_from_file_location("seedvc_aux_cache_contract", path)
    service = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(service)
    cache = str(tmp_path / "model-cache")
    monkeypatch.setenv("HF_HUB_CACHE", cache)
    config = tmp_path / "model.yml"
    config.write_text("model_params: {}\npreprocess_params:\n  spect_params: {hop_length: 256}\n  sr: 22050\n")
    calls = []

    def cached_file(repo_id, filename, *, cache_dir):
        calls.append((repo_id, filename, cache_dir))
        return str(config)

    class ResolutionFinished(Exception):
        pass

    class Model:
        def load_state_dict(self, *args, **kwargs):
            pass
        def eval(self):
            return self
        def to(self, device):
            return self

    class Models(dict):
        cfm = SimpleNamespace(estimator=SimpleNamespace(setup_caches=lambda **kwargs: None))

    def module(name, **attrs):
        value = ModuleType(name)
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    def stop(*args, **kwargs):
        raise ResolutionFinished

    monkeypatch.setattr("huggingface_hub.hf_hub_download", cached_file)
    monkeypatch.setattr(service.torch, "load", lambda *args, **kwargs: {})
    module("modules.campplus.DTDNN", CAMPPlus=lambda **kwargs: Model())
    module("modules.rmvpe", RMVPE=lambda *args, **kwargs: SimpleNamespace(infer_from_audio=lambda value: value))
    module("modules.bigvgan", bigvgan=SimpleNamespace(BigVGAN=SimpleNamespace(from_pretrained=stop)))
    module("modules.hifigan.f0_predictor", ConvRNNF0Predictor=lambda **kwargs: None)
    hift = Model()
    hift.load_state_dict = stop
    module("modules.hifigan.generator", HiFTGenerator=lambda **kwargs: hift)
    for f0,vocoder in [(False,"bigvgan"),(True,"bigvgan"),(False,"hifigan")]:
        params = SimpleNamespace(vocoder=SimpleNamespace(type=vocoder,name="test-vocoder"))
        module("modules.commons", recursive_munch=lambda value: params,
               build_model=lambda *args, **kwargs: Models(),
               load_checkpoint=lambda model, *args, **kwargs: (model,None,None,None))
        calls.clear()
        with pytest.raises(ResolutionFinished):
            service.load_models("cpu",True,f0,None,None)
        expected = [
            ("Plachta/Seed-VC", "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth" if f0 else "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth", cache),
            ("Plachta/Seed-VC", "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml" if f0 else "config_dit_mel_seed_uvit_whisper_small_wavenet.yml", cache),
        ]
        if f0:
            expected.append(("lj1995/VoiceConversionWebUI","rmvpe.pt",cache))
        expected.append(("funasr/campplus","campplus_cn_common.bin",cache))
        if vocoder == "hifigan":
            expected.append(("FunAudioLLM/CosyVoice-300M","hift.pt",cache))
        assert calls == expected
