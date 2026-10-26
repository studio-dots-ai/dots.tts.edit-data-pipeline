"""Exercise retained model-loading options independently of GPU allocation."""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace


def test_loading_forwards_all_retained_options(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root))
    core = importlib.import_module('qwen3_tts_service.core')
    import qwen_tts
    calls = []
    model = SimpleNamespace(get_supported_speakers=lambda: ['speaker'], get_supported_languages=lambda: ['English'])
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return model
    monkeypatch.setattr(qwen_tts.Qwen3TTSModel, 'from_pretrained', load)
    monkeypatch.setattr(qwen_tts.Qwen3TTSTokenizer, 'from_pretrained', load)
    monkeypatch.setitem(sys.modules, 'flash_attn', None)
    for name in ['_custom_voice_model', '_base_model', '_tokenizer']:
        monkeypatch.setattr(core, name, None)
    core.init_models(custom_voice_path='custom', base_path='base', tokenizer_path='codec', device='cpu', dtype='float32')
    assert calls == [(name, {'device_map': 'cpu', 'torch_dtype': core.torch.float32}) for name in ['custom', 'base', 'codec']]
    assert core.get_models() == (model, model, model)
    info = core.get_model_info()
    assert info == {'voice_design_path': None, 'custom_voice_path': 'custom', 'base_path': 'base', 'tokenizer_path': 'codec', 'device': 'cpu', 'dtype': 'float32', 'voice_design_loaded': False, 'custom_voice_loaded': True, 'base_loaded': True, 'tokenizer_loaded': True, 'supported_speakers': ['speaker'], 'supported_languages': ['English']}
    calls.clear()
    core.init_models(load_custom_voice=False, load_base=False, load_tokenizer=False)
    assert calls == []
    assert Path(qwen_tts.__file__).resolve().is_relative_to(root / 'upstream')
