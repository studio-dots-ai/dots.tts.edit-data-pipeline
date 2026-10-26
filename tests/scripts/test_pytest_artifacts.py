"""Exercise pytest initialization, including its built-in cache/tmp plugins."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("override", [False, True])
def test_pytest_uses_external_storage(tmp_path, override):
    project = tmp_path / "source"
    project.mkdir()
    (project / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    (project / "test_storage.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def test_storage(tmp_path, pytestconfig):\n"
        "    root = Path(os.environ['ARTIFACT_DIR'])\n"
        "    assert tmp_path.is_relative_to(root)\n"
        "    assert pytestconfig.cache.mkdir('probe').is_relative_to(root)\n"
    )
    artifacts = tmp_path / "artifacts"
    argv = [sys.executable, "-m", "pytest", "-q", str(project)]
    if override:
        argv += ["--basetemp", str(artifacts / "explicit/tmp"),
                 "-o", f"cache_dir={artifacts / 'explicit/cache'}"]
    result = subprocess.run(argv, cwd=project, env={**os.environ,
        "ARTIFACT_DIR": str(artifacts), "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1"}, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (project / ".pytest_cache").exists()
    assert not (project / "__pycache__").exists()
    if override:
        assert (artifacts / "explicit/cache").is_dir()
        assert (artifacts / "explicit/tmp").is_dir()


def test_pytest_rejects_output_outside_artifacts(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    (project / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", str(project),
         "-o", f"cache_dir={project / 'cache'}"], cwd=project,
        env={**os.environ, "ARTIFACT_DIR": str(tmp_path / "artifacts"),
             "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "inside ARTIFACT_DIR" in result.stderr
    assert not (project / "cache").exists()


def test_pytest_rejects_namespace_parent_as_basetemp(tmp_path):
    project = tmp_path / "source"
    project.mkdir()
    (project / "conftest.py").write_text((ROOT / "conftest.py").read_text())
    artifacts = tmp_path / "artifacts"
    protected = artifacts / "worktrees"
    protected.mkdir(parents=True)
    marker = protected / "keep"
    marker.write_text("existing output")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", str(project),
         "--basetemp", str(protected)], cwd=project,
        env={**os.environ, "ARTIFACT_DIR": str(artifacts),
             "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "dedicated test directory" in result.stderr
    assert marker.read_text() == "existing output"
