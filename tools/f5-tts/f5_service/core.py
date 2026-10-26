"""Select the pinned official F5 inference utilities for this service."""
import json
import sys
import subprocess
from pathlib import Path

TOOL_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = TOOL_ROOT / "upstream"
SOURCE_ROOT = UPSTREAM / "src"


def verify_upstream():
    manifest = json.loads((TOOL_ROOT / "upstream.json").read_text())
    actual = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != manifest["commit"]:
        raise RuntimeError(f"F5-TTS upstream commit differs from manifest: {actual}")
    status = subprocess.check_output(
        ["git", "-C", str(UPSTREAM), "status", "--porcelain", "--untracked-files=normal"],
        text=True,
    ).strip()
    if status:
        raise RuntimeError(f"F5-TTS official upstream checkout is dirty: {status}")


verify_upstream()
for name, module in tuple(sys.modules.items()):
    if name == "f5_tts" or name.startswith("f5_tts."):
        paths = getattr(module, "__path__", ())
        if any(not Path(path).resolve().is_relative_to(SOURCE_ROOT) for path in paths):
            raise RuntimeError(f"F5-TTS package path belongs to another checkout: {list(paths)}")
        origin = getattr(module, "__file__", None)
        if origin and not Path(origin).resolve().is_relative_to(SOURCE_ROOT):
            raise RuntimeError(f"F5-TTS was imported from another checkout: {origin}")
sys.path[:] = [path for path in sys.path
               if not (Path(path) / "f5_tts").is_dir() or Path(path).resolve() == SOURCE_ROOT]
sys.path.insert(0, str(SOURCE_ROOT))
# Namespace packages must not fall through to a pre-existing editable install.
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
_spec = PathFinder.find_spec("f5_tts", [str(SOURCE_ROOT)])
if _spec is None:
    raise RuntimeError("Initialize tools/f5-tts/upstream before starting the service")
sys.modules.setdefault("f5_tts", module_from_spec(_spec))

from f5_tts.infer import utils_infer as _official

load_model = _official.load_model
load_vocoder = _official.load_vocoder
cfg_strength = _official.cfg_strength
device = _official.device
mel_spec_type = _official.mel_spec_type
nfe_step = _official.nfe_step
sway_sampling_coef = _official.sway_sampling_coef
target_sample_rate = _official.target_sample_rate
