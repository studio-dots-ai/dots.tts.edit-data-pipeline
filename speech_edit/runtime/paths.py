"""Resolve artifact locations without creating directories or importing the runtime.

``ARTIFACT_DIR`` is the only environment setting read by this module. Git
worktree identity comes from the Git directory relative to its common directory,
so branch names and checkout mount points do not affect it. Exported source
trees without their own Git checkout use a hash of the resolved project path;
moving an exported tree therefore gives it a new artifact namespace.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import subprocess


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    worktree: Path
    shared: Path
    _source_roots: tuple[Path, ...] = field(repr=False, compare=False)

    def resolve(self, value: str | os.PathLike[str], *, scope: str = "worktree") -> Path:
        """Resolve an output using this call's verified Git boundary snapshot.

        Reuse within one configuration operation to avoid repeated Git queries.
        Obtain a fresh snapshot for subsequent operations or changed checkouts.
        """
        if scope not in {"worktree", "shared"}:
            raise ValueError("scope must be 'worktree' or 'shared'")
        path = Path(value)
        if not path.is_absolute():
            path = getattr(self, scope) / path
        path = path.resolve()
        if not _inside(path, self.root) or any(_inside(path, source) for source in self._source_roots):
            raise ValueError("Output path must resolve inside ARTIFACT_DIR and outside the repository")
        return path


def _inside(path: Path, directory: Path) -> bool:
    return path == directory or directory in path.parents


def _git_context(project_root: Path) -> tuple[str | None, tuple[Path, ...]]:
    # Do not let an enclosing process's Git overrides select another checkout.
    env = {
        key: value for key, value in os.environ.items()
        if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_COMMON_DIR", "GIT_INDEX_FILE"}
    }

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project_root), *args],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, errors="surrogateescape", env=env, timeout=5,
        ).stdout.removesuffix("\n")

    try:
        # An exported directory nested in a different repository is not one of
        # that repository's worktrees.
        if Path(git("rev-parse", "--show-toplevel")).resolve() != project_root:
            return None, (project_root,)
    except (OSError, subprocess.SubprocessError):
        if (project_root / ".git").exists():
            raise ValueError("Cannot verify Git checkout boundaries") from None
        return None, (project_root,)
    try:
        git_dir = Path(git("rev-parse", "--absolute-git-dir")).resolve()
        common_dir = Path(git("rev-parse", "--git-common-dir"))
        if not common_dir.is_absolute():
            common_dir = project_root / common_dir
        identity = git_dir.relative_to(common_dir.resolve()).as_posix()
        common_dir = common_dir.resolve()
        roots = {project_root, common_dir}
        is_bare = git("--git-dir", str(common_dir), "rev-parse", "--is-bare-repository")
        if is_bare == "false":
            if (common_dir.parent / ".git").resolve() != common_dir:
                raise ValueError("Cannot locate the main checkout of a separate Git directory")
            roots.add(common_dir.parent)
        elif is_bare != "true":
            raise ValueError("Cannot determine whether the common Git directory is bare")
        # Administrative pointers preserve exact path bytes even on Git versions
        # whose worktree-list porcelain cannot delimit embedded newlines.
        for pointer in (common_dir / "worktrees").glob("*/gitdir"):
            checkout_git = Path(os.fsdecode(pointer.read_bytes()).removesuffix("\n"))
            if not checkout_git.is_absolute():
                raise ValueError("Git worktree pointer must be absolute")
            roots.add(checkout_git.parent.resolve())
        return identity, tuple(roots)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ValueError("Cannot verify Git checkout boundaries") from exc


def _project_root(value: str | os.PathLike[str] | None) -> Path:
    return (Path(value) if value is not None else Path(__file__).parents[2]).resolve()


def get_artifact_paths(
    project_root: str | os.PathLike[str] | None = None,
    *,
    artifact_dir: str | os.PathLike[str] | None = None,
) -> ArtifactPaths:
    """Return artifact roots, requiring an absolute root outside the source tree.

    An explicit ``artifact_dir`` overrides ``ARTIFACT_DIR``, including when it is
    invalid. Existing symlinks are resolved before containment checks. No path
    is created. A project directory with no Git repository of its own is treated
    as exported source, even if an unrelated parent directory is a repository.
    """
    project = _project_root(project_root)
    raw = artifact_dir if artifact_dir is not None else os.environ.get("ARTIFACT_DIR")
    if raw is None or not os.fspath(raw).strip():
        raise ValueError("Set ARTIFACT_DIR to an absolute directory outside the repository")
    root = Path(raw)
    if not root.is_absolute():
        raise ValueError("ARTIFACT_DIR must be an absolute path")
    root = root.resolve()
    identity, source_roots = _git_context(project)
    if any(_inside(root, source) for source in source_roots):
        raise ValueError("ARTIFACT_DIR must be outside the repository")

    prefix = "git" if identity is not None else "export"
    identity = identity if identity is not None else str(project)
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    worktree = (root / "worktrees" / f"{prefix}-{digest}").resolve()
    shared = (root / "shared").resolve()
    for path in (worktree, shared):
        if not _inside(path, root) or any(_inside(path, source) for source in source_roots):
            raise ValueError("Artifact namespace resolves outside ARTIFACT_DIR or into the repository")
    return ArtifactPaths(root=root, worktree=worktree, shared=shared, _source_roots=source_roots)


def resolve_output_path(
    value: str | os.PathLike[str],
    *,
    project_root: str | os.PathLike[str] | None = None,
    artifact_dir: str | os.PathLike[str] | None = None,
    scope: str = "worktree",
) -> Path:
    """Resolve an output below ARTIFACT_DIR, relative to worktree or shared.

    Absolute inputs and relative ``..`` components must still resolve inside the
    artifact root and outside the project tree. Existing symlinks are checked;
    callers remain responsible for directory creation and later filesystem use.
    """
    if scope not in {"worktree", "shared"}:
        raise ValueError("scope must be 'worktree' or 'shared'")
    return get_artifact_paths(project_root, artifact_dir=artifact_dir).resolve(value, scope=scope)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--scope", choices=("worktree", "shared", "root"), default="worktree")
    parser.add_argument("--path", help="Resolve an output path within the selected scope")
    args = parser.parse_args(argv)
    if args.path is not None and args.scope == "root":
        parser.error("--path requires --scope worktree or shared")
    try:
        if args.path is None:
            result = getattr(get_artifact_paths(args.project_root), args.scope)
        else:
            result = resolve_output_path(args.path, project_root=args.project_root, scope=args.scope)
    except ValueError as exc:
        parser.error(str(exc))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
