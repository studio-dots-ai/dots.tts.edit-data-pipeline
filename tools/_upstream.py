"""Activate a pinned official source checkout without altering shared environments."""
from pathlib import Path
import json
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
import subprocess
import sys


def activate(tool_root, packages):
    tool_root = Path(tool_root).resolve()
    source = tool_root / "upstream"
    manifest = json.loads((tool_root / "upstream.json").read_text())
    if not (source / ".git").exists():
        raise RuntimeError(f"Initialize the official submodule: {source}")
    actual = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != manifest["commit"]:
        raise RuntimeError(f"Wrong upstream commit at {source}: {actual}")
    if subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"]).returncode:
        raise RuntimeError(f"Official upstream checkout has tracked modifications: {source}")
    for name, module in tuple(sys.modules.items()):
        if any(name == package or name.startswith(package + ".") for package in packages):
            paths = getattr(module, "__path__", ())
            if any(not Path(path).resolve().is_relative_to(source) for path in paths):
                raise RuntimeError(f"{name} has a package path from another checkout: {list(paths)}")
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(source):
                raise RuntimeError(f"{name} was already imported from another checkout: {filename}")
    sys.path.insert(0, str(source))
    for package in packages:
        spec = PathFinder.find_spec(package, [str(source)])
        if spec is not None and spec.origin is None and spec.submodule_search_locations is not None:
            # Namespace paths otherwise merge with legacy editable locations.
            paths = list(spec.submodule_search_locations)
            spec.submodule_search_locations = paths
            module = sys.modules.setdefault(package, module_from_spec(spec))
            module.__path__ = paths
    return source
