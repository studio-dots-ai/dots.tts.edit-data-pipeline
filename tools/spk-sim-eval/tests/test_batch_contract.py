import importlib.util
from pathlib import Path

import pytest
from fastapi import HTTPException

spec = importlib.util.spec_from_file_location("spk_sim_batch_service", Path(__file__).resolve().parents[1] / "api_service.py")
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


def test_batch_failure_never_becomes_similarity_sentinel(monkeypatch):
    monkeypatch.setattr(service, "model", object())
    monkeypatch.setattr(service, "compute_embedding", lambda _: object())
    def fail(_):
        raise RuntimeError("audio decoding failed")
    monkeypatch.setattr(service, "_decode_audio", fail)
    with pytest.raises(HTTPException) as error:
        service.similarity_batch(service.BatchSimilarityRequest(audio_ref="YQ==", audio_list=["Yg=="]))
    assert error.value.status_code == 500
    assert error.value.detail == "audio decoding failed"


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
    namespace = {"argparse": argparse, "os": os, 'DEFAULT_CHECKPOINT': None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[])), str(source), "exec"), namespace)
    monkeypatch.setattr(sys, "argv", ["service", "--help"])
    with pytest.raises(SystemExit) as help_exit:
        namespace["main"]()
    assert help_exit.value.code == 0
    monkeypatch.setattr(sys, "argv", ["service"])
    with pytest.raises(SystemExit) as missing_exit:
        namespace["main"]()
    assert missing_exit.value.code == 2
    monkeypatch.setattr(sys, "argv", ["service", *['--checkpoint', '/external/speaker.pth']])
    assert getattr(namespace["main"](), 'checkpoint') == '/external/speaker.pth'
