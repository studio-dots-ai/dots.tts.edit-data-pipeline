"""Forced-alignment duration and concurrency contracts."""

import importlib.util
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _load_service_module():
    module_path = Path(__file__).resolve().parents[1] / "forced_alignment_service.py"
    spec = importlib.util.spec_from_file_location(
        "qwen3_forced_alignment_service_duration_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module



def test_qwen3_duration_guard_rejects_overlong_tuple_audio():
    service = _load_service_module()
    sr = 16000
    audio = np.zeros(
        int((service.MAX_QWEN3_FORCE_ALIGN_SECONDS + 0.1) * sr),
        dtype=np.float32,
    )

    with pytest.raises(Exception) as exc_info:
        service._validate_audio_duration((audio, sr))

    assert getattr(exc_info.value, "status_code", None) == 400
    assert "too long" in getattr(exc_info.value, "detail", "")



def test_align_one_rejects_overlong_audio_before_model_load(monkeypatch):
    service = _load_service_module()
    sr = 16000
    audio = np.zeros(
        int((service.MAX_QWEN3_FORCE_ALIGN_SECONDS + 0.1) * sr),
        dtype=np.float32,
    )

    monkeypatch.setattr(service, "_decode_audio", lambda _: (audio, sr))
    monkeypatch.setattr(
        service,
        "_ensure_model_loaded",
        lambda: (_ for _ in ()).throw(RuntimeError("model should not load")),
    )

    req = SimpleNamespace(
        audio="dummy.wav",
        text="hello",
        language="English",
        separate_punctuation=False,
        debug=False,
        debug_dir=None,
        queries=None,
        word_level=False,
    )
    with pytest.raises(Exception) as exc_info:
        service._align_one(req)

    assert getattr(exc_info.value, "status_code", None) == 400
    assert "too long" in getattr(exc_info.value, "detail", "")



def test_align_many_serializes_model_inference(monkeypatch):
    service = _load_service_module()
    active = 0
    max_active = 0
    counter_lock = threading.Lock()

    class FakeAlignerProcessor:
        def encode_timestamp(self, text, language):
            return [text], text

        def build_aligner_input_text(self, word_list):
            return " ".join(word_list)

    class FakeAligner:
        aligner_processor = FakeAlignerProcessor()

        def _align_prepared(self, *, audios, word_lists, aligner_input_texts):
            nonlocal active, max_active
            with counter_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with counter_lock:
                active -= 1
            return [
                [SimpleNamespace(text=word_list[0], start_time=0.0, end_time=0.1)]
                for word_list in word_lists
            ]

    fake_aligner = FakeAligner()
    monkeypatch.setattr(service, "_ensure_model_loaded", lambda: fake_aligner)
    monkeypatch.setattr(service, "_decode_audio", lambda _: (np.zeros(16, dtype=np.float32), 16000))
    monkeypatch.setattr(service, "_validate_audio_duration", lambda _: None)
    monkeypatch.setattr(service, "_validate_language", lambda language: language)
    monkeypatch.setattr(service, "normalize_audios", lambda audios: audios)

    req = SimpleNamespace(
        audio="dummy.wav",
        text="hello",
        language="English",
        separate_punctuation=False,
        debug=False,
        debug_dir=None,
        queries=None,
        word_level=False,
    )
    start = threading.Barrier(3)
    errors = []

    def run_align():
        try:
            start.wait(timeout=5)
            service._align_many([req])
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=run_align) for _ in range(2)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert max_active == 1


def test_model_paths_are_explicit_without_blocking_help(monkeypatch):
    import argparse
    import ast
    import os
    import sys
    import pytest

    source = Path(__file__).resolve().parents[1] / 'forced_alignment_service.py'
    tree = ast.parse(source.read_text())
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    parse_index = next(i for i, node in enumerate(main.body)
                       if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "args" for t in node.targets))
    main.body = main.body[:parse_index + 1] + [ast.Return(value=ast.Name(id="args", ctx=ast.Load()))]
    namespace = {"argparse": argparse, "os": os, '_DEFAULT_CHECKPOINTS_DIR': None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["service", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        namespace["main"]()
    assert help_exit.value.code == 0
    monkeypatch.setattr(sys, "argv", ["service"])
    with pytest.raises(SystemExit) as missing_exit:
        namespace["main"]()
    assert missing_exit.value.code == 2
    monkeypatch.setattr(sys, "argv", ["service", *['--cache-dir', '/external/cache']])
    assert getattr(namespace["main"](), 'cache_dir') == '/external/cache'
