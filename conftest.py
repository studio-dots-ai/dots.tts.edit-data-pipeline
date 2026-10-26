"""Keep pytest state and temporary files outside source checkouts."""

import uuid

import pytest

from speech_edit.runtime.paths import get_artifact_paths


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    if not config.option.help:
        try:
            paths = get_artifact_paths()
            cache = config.inicfg.get("cache_dir", "tests/pytest-cache")
            for override in config.option.override_ini or ():
                if override.startswith("cache_dir="):
                    cache = override.split("=", 1)[1]
            cache = str(paths.resolve(cache))
            # Run before cacheprovider and tmpdir configure their storage.
            config.inicfg["cache_dir"] = cache
            config.option.override_ini = [
                item for item in config.option.override_ini or ()
                if not item.startswith("cache_dir=")
            ] + [f"cache_dir={cache}"]
            temporary = paths.resolve(
                config.option.basetemp or f"tests/runs/{uuid.uuid4().hex}/tmp"
            )
            if any(temporary == root or temporary in root.parents
                   for root in (paths.root, paths.worktree, paths.shared)):
                raise ValueError("pytest basetemp must be a dedicated test directory")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            config.option.basetemp = str(temporary)
        except ValueError as exc:
            raise pytest.UsageError(str(exc)) from exc
