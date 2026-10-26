"""Service adaptations of the official TED-TTS inference implementation."""
from __future__ import annotations

import importlib
import inspect
import os
from pathlib import Path
import sys

import torch
from torch.utils import cpp_extension

TOOL_ROOT = Path(__file__).resolve().parent
UPSTREAM = TOOL_ROOT / "upstream"
loaded = sys.modules.get("indextts")
if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(UPSTREAM):
    raise RuntimeError("A different indextts package was imported before TED initialization")
sys.path.insert(0, str(UPSTREAM))

# Initialize model-cache constants before upstream's import-time assignment,
# then restore the explicitly configured environment before any model is loaded.
import huggingface_hub.constants  # noqa: E402,F401
import transformers  # noqa: E402,F401
import modelscope  # noqa: E402,F401

_hf_cache = os.environ.get("HF_HUB_CACHE")
try:
    from indextts import infer_v2 as upstream_infer
finally:
    if _hf_cache is None:
        os.environ.pop("HF_HUB_CACHE", None)
    else:
        os.environ["HF_HUB_CACHE"] = _hf_cache
from indextts.gpt import model_v2 as upstream_gpt  # noqa: E402


class UnifiedVoice(upstream_gpt.UnifiedVoice):
    def prepare_gpt_inputs(self, conditional_latents, text_inputs_list, full_text=False):
        # Upstream builds [free, control]; the checkpoint expects [control, free].
        conditional_latents = [
            torch.cat((latent[:, :-2], latent[:, -1:], latent[:, -2:-1]), dim=1)
            for latent in conditional_latents
        ]
        return super().prepare_gpt_inputs(conditional_latents, text_inputs_list, full_text)

    def inference_speech(self, *args, **kwargs):
        arguments = inspect.signature(super().inference_speech).bind(*args, **kwargs)
        if arguments.arguments.get("target_duration_tokens") is not None:
            raise RuntimeError("TED-TTS duration-control generation is temporarily disabled")
        self.inference_model.duration_processor = None
        self.inference_model.duration_controller = None
        self.inference_model.segment_positions = None
        return super().inference_speech(*args, **kwargs)


class GPT2InferenceModel(upstream_gpt.GPT2InferenceModel):
    def generate(self, *args, **kwargs):
        output = super().generate(*args, **kwargs)
        # The service consumes generated audio, not upstream's optional beam
        # statistics. Discard them only after decoding has finished.
        self.segment_positions = None
        return output


class _EmotionModelLoader:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device

    def from_pretrained(self, model_dir, **kwargs):
        kwargs.pop("device_map", None)
        if self.device != "mps":
            kwargs["device_map"] = {"": self.device}
        model = self.loader.from_pretrained(model_dir, **kwargs)
        return model.to(self.device) if self.device == "mps" else model


class IndexTTS2(upstream_infer.IndexTTS2):
    def __init__(self, cfg_path, model_dir, is_fp16=False, device=None, use_cuda_kernel=None):
        selected_device = device or (
            "cuda:0" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        loader = upstream_infer.AutoModelForCausalLM
        upstream_infer.AutoModelForCausalLM = _EmotionModelLoader(loader, selected_device)
        try:
            super().__init__(
                cfg_path=cfg_path, model_dir=model_dir, is_fp16=is_fp16,
                device=device, use_cuda_kernel=use_cuda_kernel,
            )
        finally:
            upstream_infer.AutoModelForCausalLM = loader
        requested_cuda_kernel = (
            use_cuda_kernel is True
            or (device is None and use_cuda_kernel is None and selected_device.startswith("cuda"))
        )
        if requested_cuda_kernel and not self.use_cuda_kernel:
            raise RuntimeError("TED-TTS requested the CUDA kernel, but upstream initialization disabled it")


def _load_cuda_kernel(source_dir, cache_name):
    name = "anti_alias_activation_cuda"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    if os.environ.get("TORCH_EXTENSIONS_DIR"):
        buildpath = Path(os.environ["TORCH_EXTENSIONS_DIR"]) / "ted-tts" / cache_name
    else:
        from speech_edit.runtime.paths import resolve_output_path
        buildpath = resolve_output_path(f"cache/ted-tts/{cache_name}", scope="shared")
    buildpath.mkdir(parents=True, exist_ok=True)
    sources = [source_dir / "anti_alias_activation.cpp", source_dir / "anti_alias_activation_cuda.cu"]
    environment_bin = str(Path(sys.executable).parent)
    if environment_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = os.pathsep.join([environment_bin, os.environ.get("PATH", "")])
    _, major, _ = _bigvgan_loader._get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    flags = ["-gencode", "arch=compute_80,code=sm_80"] if int(major) >= 11 else []
    extension = cpp_extension.load(
        name=name, sources=sources, build_directory=str(buildpath), extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3", "-gencode", "arch=compute_70,code=sm_70", "--use_fast_math",
            "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
            "--expt-relaxed-constexpr", "--expt-extended-lambda", *flags,
        ],
        verbose=True,
    )
    sys.modules[name] = extension
    return extension


def load_bigvgan_cuda():
    return _load_cuda_kernel(Path(_bigvgan_loader.__file__).parent, "bigvgan")


def load_s2mel_cuda():
    return _load_cuda_kernel(Path(_s2mel_loader.__file__).parent, "s2mel")


upstream_infer.UnifiedVoice = UnifiedVoice
upstream_gpt.GPT2InferenceModel = GPT2InferenceModel
_bigvgan_loader = importlib.import_module("indextts.BigVGAN.alias_free_activation.cuda.load")
_s2mel_loader = importlib.import_module("indextts.s2mel.modules.bigvgan.alias_free_activation.cuda.load")
_bigvgan_loader.load = load_bigvgan_cuda
_s2mel_loader.load = load_s2mel_cuda
