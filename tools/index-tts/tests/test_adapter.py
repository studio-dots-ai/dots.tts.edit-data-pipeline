"""Direct upstream imports and the service's narrowly scoped adaptations."""
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import adapter


def test_model_inherits_adjacent_official_source():
    base = adapter.IndexTTS2.__bases__[0]
    assert Path(inspect.getfile(base)).resolve().is_relative_to(adapter.UPSTREAM_ROOT)
    assert adapter.IndexTTS2.get_emb is base.get_emb
    assert subprocess.check_output(
        ['git', '-C', str(adapter.UPSTREAM_ROOT), 'status', '--porcelain'], text=True,
    ).strip() == ''


def test_fresh_service_import_does_not_need_git_or_lfs():
    code = (
        "import sys,subprocess; "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
        "subprocess.check_output=lambda *a,**k: (_ for _ in ()).throw(AssertionError('git invoked')); "
        "import api_service; "
        "assert api_service.IndexTTS2.__module__ == 'adapter'"
    )
    subprocess.run([sys.executable, '-c', code], check=True, timeout=120)


def test_upstream_import_preserves_configured_hub_cache(tmp_path):
    environment = dict(os.environ, HF_HUB_CACHE=str(tmp_path))
    code = (
        "import sys,os; "
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r}); "
        "import adapter,huggingface_hub.constants; "
        f"assert os.environ['HF_HUB_CACHE'] == {str(tmp_path)!r}; "
        f"assert huggingface_hub.constants.HF_HUB_CACHE == {str(tmp_path)!r}"
    )
    subprocess.run([sys.executable, '-c', code], env=environment, check=True, timeout=120)


@pytest.mark.parametrize('prefix', [None, object()])
@pytest.mark.parametrize('fail', [False, True])
def test_continuation_selects_hf_path_and_restores_engine(monkeypatch, prefix, fail):
    voice = adapter.UnifiedVoice.__new__(adapter.UnifiedVoice)
    # Avoid torch Module attribute registration in this model-free test.
    object.__setattr__(voice, 'accel_engine', object())
    engine = voice.accel_engine
    captured = {}
    def original(self, *args, **kwargs):
        captured['engine'] = self.accel_engine
        captured['args'] = args
        captured['kwargs'] = kwargs
        if fail:
            raise ValueError('generation failed')
        return 'tokens'
    monkeypatch.setattr(adapter.UpstreamUnifiedVoice, 'inference_speech', original)
    if fail:
        with pytest.raises(ValueError, match='generation failed'):
            voice.inference_speech('speech', 'text', input_tokens=prefix, temperature=.8)
    else:
        assert voice.inference_speech('speech', 'text', input_tokens=prefix, temperature=.8) == 'tokens'
    assert captured['engine'] is (engine if prefix is None else None)
    assert captured['args'][7] is prefix
    assert captured['kwargs'] == {'temperature': .8}
    assert voice.accel_engine is engine


@pytest.mark.parametrize('explicit', [False, True])
def test_cuda_build_uses_official_sources_and_torch_cache(monkeypatch, tmp_path, explicit):
    from speech_edit.runtime import paths
    monkeypatch.setattr(paths, 'get_artifact_paths', lambda: SimpleNamespace(shared=tmp_path))
    if explicit:
        monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    else:
        monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.setattr(adapter.cuda_load, '_get_cuda_bare_metal_version', lambda _: ('', '12', '8'))
    build = Mock(return_value='kernel')
    monkeypatch.setattr(adapter.cpp_extension, 'load', build)
    assert adapter.load_cuda_kernel() == 'kernel'
    kwargs = build.call_args.kwargs
    assert Path(kwargs['build_directory']).is_relative_to((tmp_path if explicit else tmp_path / 'torch_extensions') / 'index_tts')
    assert {Path(p).suffix for p in kwargs['sources']} == {'.cpp', '.cu'}
    assert all(Path(p).is_relative_to(adapter.UPSTREAM_ROOT) for p in kwargs['sources'])
    assert kwargs['extra_cflags'] == ['-O3']
    assert '--use_fast_math' in kwargs['extra_cuda_cflags']
    assert 'arch=compute_80,code=sm_80' in kwargs['extra_cuda_cflags']


def test_model_paths_are_explicit_without_blocking_help(monkeypatch):
    import argparse
    import ast
    import os
    import sys
    import pytest

    source = Path(__file__).resolve().parents[1] / 'api_service.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    parse_index = next(i for i, node in enumerate(main.body)
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets))
    main.body = main.body[:parse_index + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    namespace = {"argparse": argparse, "os": os, '_CHECKPOINTS_DIR': None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["service", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        namespace["main"]()
    assert help_exit.value.code == 0
    monkeypatch.setattr(sys, "argv", ["service"])
    with pytest.raises(SystemExit) as missing_exit:
        namespace["main"]()
    assert missing_exit.value.code == 2
    monkeypatch.setattr(sys, "argv", ["service", *['--model-dir', '/external/model', '--cfg-path', '/external/model/config.yaml']])
    assert getattr(namespace["main"](), 'model_dir') == '/external/model'


def test_emotion_model_device_override_preserves_all_other_arguments(monkeypatch):
    loader = Mock(return_value='model')
    monkeypatch.setattr(adapter, '_load_causal_model', loader)
    assert adapter.load_emotion_model('model-dir', torch_dtype='float16', device_map='auto') == 'model'
    loader.assert_called_once_with('model-dir', torch_dtype='float16', device_map=0)


def test_internal_inference_index_error_is_not_swallowed(monkeypatch):
    model = adapter.IndexTTS2.__new__(adapter.IndexTTS2)
    def fail(*args, **kwargs):
        raise IndexError('invalid token index')
    monkeypatch.setattr(model, 'infer_generator', fail)
    with pytest.raises(IndexError, match='invalid token index'):
        model.infer('prompt', 'text', None)


@pytest.mark.parametrize('skip_vocode', [False, True])
def test_infer_preserves_cached_conditioning_continuation_and_generation_options(monkeypatch, skip_vocode):
    model = adapter.IndexTTS2.__new__(adapter.IndexTTS2)
    model.normalizer = SimpleNamespace(use_chinese=lambda text: False)
    speaker = {'spk_cond': object()}
    emotion = {'emo_cond': object()}
    tokens = {'tokens': [1, 2, 3], 'token_rate': 50.0}
    result = tokens if skip_vocode else (('wave', 24000), tokens)
    generator = Mock(return_value=iter([result]))
    monkeypatch.setattr(model, 'infer_generator', generator)
    output, cache = model.infer(
        'prompt', 'world', None, speaker_cache=speaker, emotion_cache=emotion,
        return_cache=True, prefix_tokens=[1], prefix_text='Hello', return_tokens=True,
        skip_vocode=skip_vocode, temperature=.7, max_mel_tokens=500,
    )
    assert output is result
    assert cache == {'speaker': speaker, 'emotion': emotion,
                     'continuation': {'tokens': [1, 2, 3], 'text': 'Hello world'}}
    assert generator.call_args.kwargs == {
        'speaker_cache': speaker, 'emotion_cache': emotion,
        'prefix_tokens': [1], 'prefix_text': 'Hello', 'return_tokens': True,
        'skip_vocode': skip_vocode, 'temperature': .7, 'max_mel_tokens': 500,
    }


@pytest.mark.parametrize('requested,message', [
    ('use_cuda_kernel', 'CUDA kernel'), ('use_deepspeed', 'DeepSpeed'), ('use_accel', 'Acceleration'),
])
def test_requested_engine_cannot_silently_disable(monkeypatch, requested, message):
    import api_service
    monkeypatch.setitem(sys.modules, 'deepspeed', SimpleNamespace())
    model = SimpleNamespace(use_cuda_kernel=False, gpt=SimpleNamespace(accel_engine=None))
    monkeypatch.setattr(api_service, 'IndexTTS2', lambda **kwargs: model)
    options = dict(model_dir='model', cfg_path='config', device='cuda:0', use_fp16=False,
                   use_deepspeed=False, use_accel=False, use_cuda_kernel=False, use_torch_compile=False)
    options[requested] = True
    with pytest.raises(RuntimeError, match=message):
        api_service.load_model(**options)


def test_deepspeed_import_failure_is_not_suppressed(monkeypatch):
    import api_service
    monkeypatch.setitem(sys.modules, 'deepspeed', None)
    constructor = Mock()
    monkeypatch.setattr(api_service, 'IndexTTS2', constructor)
    with pytest.raises(ImportError):
        api_service.load_model(model_dir='model', cfg_path='config', device='cuda:0', use_fp16=False,
                               use_deepspeed=True, use_accel=False, use_cuda_kernel=False, use_torch_compile=False)
    constructor.assert_not_called()
