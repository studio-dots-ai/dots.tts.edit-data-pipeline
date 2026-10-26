"""Artifact namespaces and source-tree boundary checks."""

from dataclasses import FrozenInstanceError
import os
from pathlib import Path
import subprocess
import sys

import pytest

from speech_edit.runtime.paths import get_artifact_paths, resolve_output_path


SCRIPT = Path(__file__).resolve().parents[2] / "speech_edit/runtime/paths.py"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.delenv("ARTIFACT_DIR", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "main")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "Initial")
    return repo


def test_git_worktrees_are_isolated_and_branch_rename_is_stable(source, tmp_path):
    artifact = tmp_path / "artifacts"
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-b", "feature", str(linked))
    main_paths = get_artifact_paths(source, artifact_dir=artifact)
    linked_paths = get_artifact_paths(linked, artifact_dir=artifact)
    assert main_paths.root == linked_paths.root == artifact
    assert main_paths.shared == linked_paths.shared == artifact / "shared"
    assert main_paths.worktree != linked_paths.worktree
    git(linked, "branch", "-m", "renamed")
    assert get_artifact_paths(linked, artifact_dir=artifact) == linked_paths
    git(source, "branch", "-m", "renamed-main")
    assert get_artifact_paths(source, artifact_dir=artifact) == main_paths
    moved = tmp_path / "moved-linked"
    git(source, "worktree", "move", str(linked), str(moved))
    assert get_artifact_paths(moved, artifact_dir=artifact) == linked_paths
    assert not artifact.exists()
    with pytest.raises(FrozenInstanceError):
        main_paths.root = source


@pytest.mark.parametrize("linked_checkout", [False, True])
def test_git_infers_non_bare_when_core_bare_is_omitted(source, tmp_path, linked_checkout):
    project = source
    if linked_checkout:
        project = tmp_path / "linked"
        git(source, "worktree", "add", "-b", "feature", str(project))
    git(source, "config", "--unset", "core.bare")
    artifact = tmp_path / "artifacts"
    assert get_artifact_paths(project, artifact_dir=artifact).root == artifact
    with pytest.raises(ValueError, match="outside the repository"):
        get_artifact_paths(project, artifact_dir=source / "outputs")


def test_main_checkout_move_does_not_change_git_identity(source, tmp_path):
    before = get_artifact_paths(source, artifact_dir=tmp_path / "artifacts")
    moved = tmp_path / "moved"
    source.rename(moved)
    assert get_artifact_paths(moved, artifact_dir=tmp_path / "artifacts") == before


def test_artifacts_cannot_enter_another_checkout_of_the_same_repository(source, tmp_path):
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-b", "feature", str(linked))
    for project, forbidden in ((linked, source), (source, linked)):
        with pytest.raises(ValueError, match="outside the repository"):
            get_artifact_paths(project, artifact_dir=forbidden / "outputs")
        # The root can contain checkouts, but individual outputs cannot enter them.
        with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
            resolve_output_path(forbidden / "outputs", project_root=project, artifact_dir=tmp_path)


@pytest.mark.parametrize("name", ['quote"slash\\name', "中文 worktree", "line\nbreak", " trailing "])
def test_quoted_worktree_paths_remain_source_boundaries(source, tmp_path, name):
    linked = tmp_path / name
    git(source, "worktree", "add", "-b", "special", str(linked))
    for project, forbidden in ((source, linked), (linked, source)):
        with pytest.raises(ValueError, match="outside the repository"):
            get_artifact_paths(project, artifact_dir=forbidden / "outputs")
    assert get_artifact_paths(linked, artifact_dir=tmp_path / "artifacts").root == tmp_path / "artifacts"


def test_bare_repository_does_not_exclude_its_parent(source, tmp_path):
    bare = tmp_path / "bare.git"
    git(source, "clone", "--bare", str(source), str(bare))
    linked = tmp_path / "linked-bare"
    git(bare, "worktree", "add", str(linked), "main")
    assert get_artifact_paths(linked, artifact_dir=tmp_path / "artifacts").root == tmp_path / "artifacts"
    with pytest.raises(ValueError, match="outside the repository"):
        get_artifact_paths(linked, artifact_dir=bare / "artifacts")


def test_separate_git_directory_fails_when_main_checkout_cannot_be_verified(source, tmp_path):
    separate = tmp_path / "separate-source"
    separate.mkdir()
    git(separate, "init", "--separate-git-dir", str(tmp_path / "git-storage"))
    git(separate, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--allow-empty", "-m", "Initial")
    linked = tmp_path / "linked-separate"
    git(separate, "worktree", "add", "-b", "feature", str(linked))
    for checkout in (separate, linked):
        with pytest.raises(ValueError, match="Cannot verify Git checkout boundaries"):
            get_artifact_paths(checkout, artifact_dir=tmp_path / "artifacts")


@pytest.mark.parametrize("value", [None, "", "  ", "relative/artifacts"])
def test_missing_or_relative_artifact_root_is_rejected(source, value):
    with pytest.raises(ValueError, match="ARTIFACT_DIR"):
        get_artifact_paths(source, artifact_dir=value)


def test_explicit_root_overrides_environment_without_falling_back(source, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "from-env"))
    assert get_artifact_paths(source).root == tmp_path / "from-env"
    assert get_artifact_paths(source, artifact_dir=tmp_path / "explicit").root == tmp_path / "explicit"
    with pytest.raises(ValueError):
        get_artifact_paths(source, artifact_dir="")


def test_artifact_root_cannot_resolve_into_source(source, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    for path in (source, source / "outputs", alias / "outputs"):
        with pytest.raises(ValueError, match="outside the repository"):
            get_artifact_paths(source, artifact_dir=path)


def test_output_paths_use_scope_and_reject_escape(source, tmp_path):
    artifact = tmp_path / "artifacts"
    paths = get_artifact_paths(source, artifact_dir=artifact)
    assert resolve_output_path("logs/run.log", project_root=source, artifact_dir=artifact) == paths.worktree / "logs/run.log"
    assert resolve_output_path("models", project_root=source, artifact_dir=artifact, scope="shared") == paths.shared / "models"
    absolute = artifact / "custom" / "result.wav"
    assert resolve_output_path(absolute, project_root=source, artifact_dir=artifact) == absolute
    for path in (tmp_path / "outside", source / "outputs", "../../../escape"):
        with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
            resolve_output_path(path, project_root=source, artifact_dir=artifact)
    with pytest.raises(ValueError, match="scope"):
        resolve_output_path("x", project_root=source, artifact_dir=artifact, scope="root")
    assert not artifact.exists()


def test_existing_symlinks_cannot_escape_output_or_namespace(source, tmp_path):
    artifact = tmp_path / "artifacts"
    paths = get_artifact_paths(source, artifact_dir=artifact)
    paths.worktree.mkdir(parents=True)
    (paths.worktree / "escape").symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        resolve_output_path("escape/result", project_root=source, artifact_dir=artifact)
    (artifact / "shared").symlink_to(tmp_path / "outside", target_is_directory=True)
    with pytest.raises(ValueError, match="namespace"):
        get_artifact_paths(source, artifact_dir=artifact)


def test_exported_sources_do_not_inherit_parent_git_identity(source, tmp_path):
    artifact = tmp_path / "artifacts"
    exported = source / "exported"
    exported.mkdir()
    another = tmp_path / "standalone"
    another.mkdir()
    paths = get_artifact_paths(exported, artifact_dir=artifact)
    assert paths.worktree.name.startswith("export-")
    assert paths == get_artifact_paths(exported, artifact_dir=artifact)
    assert paths.worktree != get_artifact_paths(source, artifact_dir=artifact).worktree
    assert paths.worktree != get_artifact_paths(another, artifact_dir=artifact).worktree


def test_cli_help_needs_no_environment_and_paths_match_python(source, tmp_path):
    env = dict(os.environ)
    env.pop("ARTIFACT_DIR", None)
    help_result = subprocess.run([sys.executable, str(SCRIPT), "--help"], env=env, capture_output=True, text=True)
    assert help_result.returncode == 0
    assert "--path" in help_result.stdout
    env["ARTIFACT_DIR"] = str(tmp_path / "artifacts")
    for scope in ("root", "shared", "worktree"):
        result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(source), "--scope", scope], env=env, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == str(getattr(get_artifact_paths(source, artifact_dir=env["ARTIFACT_DIR"]), scope))
    command = [sys.executable, str(SCRIPT), "--project-root", str(source), "--scope", "shared", "--path", "weights"]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == str(tmp_path / "artifacts/shared/weights")
    command[command.index("shared")] = "root"
    assert subprocess.run(command, env=env, capture_output=True).returncode == 2
    assert not (tmp_path / "artifacts").exists()


def test_git_checkout_boundary_failure_does_not_fall_back_to_export(source, tmp_path, monkeypatch):
    linked = tmp_path / "linked"
    git(source, "worktree", "add", "-b", "feature", str(linked))
    real_read_bytes = Path.read_bytes

    def fail_worktree_pointer(path):
        if path.name == "gitdir":
            raise PermissionError("unreadable worktree registration")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_worktree_pointer)
    with pytest.raises(ValueError, match="Cannot verify Git"):
        get_artifact_paths(source, artifact_dir=tmp_path / "artifacts")


def test_snapshot_resolves_multiple_outputs_without_reprobing_git(source, tmp_path, monkeypatch):
    import speech_edit.runtime.paths as module
    paths = get_artifact_paths(source, artifact_dir=tmp_path / "artifacts")
    monkeypatch.setattr(module, "_git_context", lambda *args: pytest.fail("snapshot reprobed Git"))
    assert paths.resolve("a") == paths.worktree / "a"
    assert paths.resolve("b", scope="shared") == paths.shared / "b"
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        paths.resolve(source / "output")
    with pytest.raises(ValueError, match="scope"):
        paths.resolve("x", scope="root")
