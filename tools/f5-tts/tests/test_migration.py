"""Migration invariants independent of HTTP/GPU availability."""
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_official_source_and_model_loader_are_isolated():
    from f5_service import core
    from f5_tts.model import cfm
    from importlib.resources import files
    assert Path(core._official.__file__).resolve().is_relative_to(ROOT / "upstream")
    import f5_tts.infer
    assert all(Path(path).resolve().is_relative_to(ROOT / "upstream") for path in f5_tts.infer.__path__)
    assert str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")).startswith(str(ROOT / "upstream"))
    assert core.load_model is core._official.load_model
    assert core.load_model.__globals__["CFM"] is cfm.CFM
    assert "allow_duration_auto_extend" not in inspect.signature(cfm.CFM.sample).parameters


@pytest.mark.parametrize("text", ["a", "a" * 30])
@pytest.mark.parametrize("duration", [5, 8])
def test_exact_duration_and_preserved_mel_frames(text, duration):
    import torch
    from torch import nn
    from f5_service import core
    CFM = core._official.CFM

    class Transformer(nn.Module):
        dim = 2
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(1))
        def forward(self, x, **kwargs):
            return torch.cat([torch.zeros_like(x)] * (2 if kwargs.get("cfg_infer") else 1), dim=0)
        def clear_cache(self):
            pass

    model = CFM(transformer=Transformer(), num_channels=2)
    cond = torch.arange(10, dtype=torch.float32).reshape(1, 5, 2)
    mask = torch.tensor([[True, False, True, False, True]])
    output, _ = model.sample(cond=cond, text=[text], duration=duration, edit_mask=mask,
                             max_duration=duration, steps=2, seed=42)
    assert output.shape == (1, duration, 2)
    assert torch.equal(output[:, :5][mask], cond[mask])


def test_foreign_namespace_is_rejected_before_import():
    import subprocess
    script = "import sys,types; m=types.ModuleType('f5_tts'); m.__path__=['/foreign/f5_tts']; sys.modules['f5_tts']=m; from f5_service import core"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "package path belongs to another checkout" in result.stderr


def test_wrong_official_commit_is_rejected(monkeypatch):
    from f5_service import core
    original = core.subprocess.check_output
    def output(command, **kwargs):
        if "rev-parse" in command:
            return "0" * 40
        return original(command, **kwargs)
    monkeypatch.setattr(core.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="commit differs from manifest"):
        core.verify_upstream()


def test_other_dirty_upstream_files_are_rejected(monkeypatch):
    from f5_service import core
    original = core.subprocess.check_output
    def output(command, **kwargs):
        if "status" in command:
            return " M src/f5_tts/model/backbones/dit.py\n"
        return original(command, **kwargs)
    monkeypatch.setattr(core.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="upstream checkout is dirty"):
        core.verify_upstream()
