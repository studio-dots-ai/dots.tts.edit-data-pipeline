"""The official TED kernels compile outside the checkout and preserve preload selection."""
from pathlib import Path
import sys

import pytest
import adapter


@pytest.mark.parametrize("kernel", ["bigvgan", "s2mel"])
def test_compiles_official_sources_in_external_tool_cache(tmp_path, monkeypatch, kernel):
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    monkeypatch.setitem(sys.modules, "anti_alias_activation_cuda", None)
    monkeypatch.setattr(adapter._bigvgan_loader, "_get_cuda_bare_metal_version", lambda _: ("", "12", "8"))
    captured = {}
    sentinel = object()
    def compile_extension(**kwargs):
        captured.update(kwargs)
        return sentinel
    monkeypatch.setattr(adapter.cpp_extension, "load", compile_extension)
    load = adapter.load_bigvgan_cuda if kernel == "bigvgan" else adapter.load_s2mel_cuda
    assert load() is sentinel
    assert Path(captured["build_directory"]) == tmp_path / "ted-tts" / kernel
    assert {path.suffix for path in captured["sources"]} == {".cpp", ".cu"}
    assert all(path.is_relative_to(adapter.UPSTREAM) for path in captured["sources"])
    expected_dir = adapter._bigvgan_loader if kernel == "bigvgan" else adapter._s2mel_loader
    assert {path.parent for path in captured["sources"]} == {Path(expected_dir.__file__).parent}
    assert "arch=compute_80,code=sm_80" in captured["extra_cuda_cflags"]
    assert sys.modules["anti_alias_activation_cuda"] is sentinel


def test_both_loaders_reuse_first_loaded_kernel(monkeypatch):
    sentinel = object()
    monkeypatch.setitem(sys.modules, "anti_alias_activation_cuda", sentinel)
    def forbidden(*args, **kwargs):
        raise AssertionError("An already loaded kernel must not be compiled again")
    monkeypatch.setattr(adapter.cpp_extension, "load", forbidden)
    monkeypatch.setattr(adapter._bigvgan_loader, "_get_cuda_bare_metal_version", forbidden)
    assert adapter.load_bigvgan_cuda() is sentinel
    assert adapter.load_s2mel_cuda() is sentinel
