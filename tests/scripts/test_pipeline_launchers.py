"""Contract checks for the production YAML service launcher."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("pipeline", ["text", "emotion", "prosody", "pause"])
@pytest.mark.parametrize("profile", ["evaluate", "defaults"])
def test_launcher_dry_run_outside_checkout(tmp_path, pipeline, profile):
    from speech_edit.runtime.paths import get_artifact_paths

    artifact_root = tmp_path / "artifacts with spaces"
    env = {**os.environ, "ARTIFACT_DIR": str(artifact_root),
           "PYTHONPATH": str(ROOT)}
    env.pop("PROJECT_ROOT", None)
    pipeline_name = {"text": "text_edit", "emotion": "emotion_style_edit", "prosody": "prosody_edit", "pause": "pause_edit"}[pipeline]
    command = [sys.executable, "-m", "speech_edit.runtime.services", "--config", f"service/{pipeline_name}_pipeline/{profile}", "--dry-run"]
    result = subprocess.run(
        command, cwd=tmp_path, env=env,
        capture_output=True, text=True, check=True,
    )
    commands = json.loads(result.stdout)
    paths = get_artifact_paths(ROOT, artifact_dir=artifact_root)
    assert commands
    assert any(str(paths.worktree) in spec["command"] for spec in commands)
    assert not artifact_root.exists()
    env.pop("ARTIFACT_DIR")
    result = subprocess.run(
        command, cwd=tmp_path, env=env,
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "ARTIFACT_DIR" in result.stderr
    assert not artifact_root.exists()


@pytest.mark.parametrize("pipeline", ["text", "emotion", "prosody", "pause"])
def test_service_help_needs_no_artifact_configuration(tmp_path, pipeline):
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("ARTIFACT_DIR", None)
    result = subprocess.run(
        [sys.executable, "-m", f"speech_edit.pipelines.{pipeline}.service.api_service", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
    )
    assert "--output-dir" in result.stdout
    assert not list(tmp_path.iterdir())
