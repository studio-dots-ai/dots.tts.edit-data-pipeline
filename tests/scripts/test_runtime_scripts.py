"""Public setup must be explicit, portable and independent of deployment secrets."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_worker_entry_resolves_checkout_from_outside_repo(tmp_path):
    env = {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "RUN_CONFIG": "unused.yaml",
        "PIPELINE": "text_edit",
        "STAGE": "sample",
    }
    env.pop("PROJECT_ROOT", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        ["bash", str(ROOT / "speech_edit/runtime/entry.sh"), "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert "--run-config" in result.stdout
    assert "--worker-id" in result.stdout
    assert not list(tmp_path.iterdir())


def test_env_can_be_sourced_outside_shared_storage(tmp_path):
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "PROJECT_ROOT": str(tmp_path / "checkout"),
           "GEMINI_API_KEY": "external-test-value",
           "UV_CACHE_DIR": str(tmp_path / "existing-cache")}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$CHECKPOINTS_DIR" "$HF_HUB_OFFLINE" "$GEMINI_API_KEY" "$UV_CACHE_DIR"',
         "bash", str(ROOT / "scripts/env.sh")],
        env=env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout.splitlines() == ["", "1",
                                           "external-test-value", str(tmp_path / "existing-cache")]
    assert not list(tmp_path.iterdir())


def test_setup_dry_run_does_not_invoke_uv(tmp_path):
    fake_uv = tmp_path / "uv"
    marker = tmp_path / "invoked"
    fake_uv.write_text('#!/bin/sh\ntouch "$MARKER"\nexit 1\n')
    fake_uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "MARKER": str(marker)}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/setup_venvs.sh"), "--env", "pipe-cpu",
         "--python", "/opt/shared/python", "--dry-run"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert "sync --frozen --project" in result.stdout
    assert "--python /opt/shared/python" in result.stdout
    assert "envs/pipe-cpu" in result.stdout
    assert "envs/pipe-gpu" not in result.stdout
    assert not marker.exists()


@pytest.mark.parametrize("args", [["--env", "../pipe-cpu"], ["--env"], ["--unknown"]])
def test_setup_rejects_invalid_arguments(args):
    result = subprocess.run(["bash", str(ROOT / "scripts/setup_venvs.sh"), *args], capture_output=True)
    assert result.returncode == 2




def test_env_uses_external_models_and_artifact_module_cache(tmp_path):
    checkpoints = tmp_path / "models"
    artifacts = tmp_path / "artifacts"
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path),
           "PROJECT_ROOT": str(ROOT), "CHECKPOINTS_DIR": str(checkpoints),
           "ARTIFACT_DIR": str(artifacts)}
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s\\n" "$CHECKPOINTS_DIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$HF_MODULES_CACHE"',
         "bash", str(ROOT / "scripts/env.sh")],
        env=env, cwd=tmp_path, text=True, capture_output=True, check=True,
    )
    assert result.stdout.splitlines() == [
        str(checkpoints), str(checkpoints), str(checkpoints / "torch"),
        str(artifacts / "shared/cache/hf_modules"),
    ]
    assert not list(tmp_path.iterdir())
