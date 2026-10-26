"""Service adapters for the pinned official qwen3-tts source."""
from pathlib import Path
import sys

_TOOL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TOOL_ROOT.parents[1]))
from tools._upstream import activate

UPSTREAM_ROOT = activate(_TOOL_ROOT, ("qwen_tts",))
