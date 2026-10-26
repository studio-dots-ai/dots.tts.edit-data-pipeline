"""IndexTTS service adaptations over the adjacent official source checkout."""
import os
from pathlib import Path
import random
import sys
import time
import warnings
from types import SimpleNamespace

UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
sys.path.insert(0, str(UPSTREAM_ROOT.parents[2]))
sys.path.insert(0, str(UPSTREAM_ROOT))
# Absolute interpreter launches also need the same environment's build tools.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

# Initialize Hub configuration before upstream's import-time environment write.
from huggingface_hub import constants as hub_constants
import numpy as np
import torch
import torchaudio
from torch.utils import cpp_extension

_hub_cache = os.environ.get("HF_HUB_CACHE")
try:
    from indextts import infer_v2 as upstream
finally:
    if _hub_cache is None:
        os.environ.pop("HF_HUB_CACHE", None)
    else:
        os.environ["HF_HUB_CACHE"] = _hub_cache

from indextts.gpt.model_v2 import UnifiedVoice as UpstreamUnifiedVoice
from indextts.s2mel.modules.bigvgan.alias_free_activation.cuda import load as cuda_load
from indextts.infer_v2 import find_most_similar_cosine


def load_cuda_kernel():
    """Build official CUDA sources in PyTorch's configured external cache."""
    source = Path(cuda_load.__file__).parent
    _, major, _ = cuda_load._get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)
    flags = ["-O3", "-gencode", "arch=compute_70,code=sm_70", "--use_fast_math",
             "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
             "--expt-relaxed-constexpr", "--expt-extended-lambda"]
    if int(major) >= 11:
        flags += ["-gencode", "arch=compute_80,code=sm_80"]
    from speech_edit.runtime.paths import get_artifact_paths
    cache = Path(os.environ["TORCH_EXTENSIONS_DIR"]) if os.environ.get("TORCH_EXTENSIONS_DIR") else get_artifact_paths().shared / "torch_extensions"
    build = cache / "index_tts" / f"torch-{torch.__version__}-cuda-{torch.version.cuda}"
    build.mkdir(parents=True, exist_ok=True)
    return cpp_extension.load(
        name="anti_alias_activation_cuda",
        sources=[str(source / "anti_alias_activation.cpp"),
                 str(source / "anti_alias_activation_cuda.cu")],
        build_directory=str(build),
        extra_cflags=["-O3"], extra_cuda_cflags=flags, verbose=True,
    )


class UnifiedVoice(UpstreamUnifiedVoice):
    def inference_speech(
        self, speech_condition, text_inputs, emo_speech_condition=None,
        cond_lengths=None, emo_cond_lengths=None, emo_vec=None, use_speed=False,
        input_tokens=None, num_return_sequences=1, max_generate_length=None,
        typical_sampling=False, typical_mass=.9, **hf_generate_kwargs,
    ):
        # The acceleration engine cannot consume continuation tokens. The
        # service's inference lock covers this temporary per-model routing.
        engine = self.accel_engine
        if input_tokens is not None:
            self.accel_engine = None
        try:
            return super().inference_speech(
                speech_condition, text_inputs, emo_speech_condition,
                cond_lengths, emo_cond_lengths, emo_vec, use_speed, input_tokens,
                num_return_sequences, max_generate_length, typical_sampling,
                typical_mass, **hf_generate_kwargs,
            )
        finally:
            self.accel_engine = engine


_load_causal_model = upstream.AutoModelForCausalLM.from_pretrained


def load_emotion_model(*args, **kwargs):
    """Keep the emotion model on the service's first visible device."""
    return _load_causal_model(*args, **{**kwargs, "device_map": 0})


# Install only the constructor dependencies used by the inherited initializer.
upstream.UnifiedVoice = UnifiedVoice
upstream.AutoModelForCausalLM = SimpleNamespace(from_pretrained=load_emotion_model)
cuda_load.load = load_cuda_kernel


class IndexTTS2(upstream.IndexTTS2):
    def _join_continuation_text(self, prefix_text: str, text: str) -> str:
        """Join continuation transcript pieces without adding fake CJK spaces."""
        prefix = prefix_text.strip()
        suffix = text.strip()
        if not prefix:
            return suffix
        if not suffix:
            return prefix

        # Match the front-end language heuristic: if either side is treated as
        # Chinese/CJK-dominant, preserve direct concatenation instead of forcing
        # a Latin-style word boundary that can surface as an unnecessary pause.
        if self.normalizer and (
            self.normalizer.use_chinese(prefix[-1:]) or self.normalizer.use_chinese(suffix[:1])
        ):
            return prefix + suffix

        if suffix[0] in ",.;:!?)]}%}'\"-" or prefix[-1] in "([{/'\"-":
            return prefix + suffix
        return prefix + " " + suffix

    def _max_vocode_code_tokens(self) -> int:
        """Return the max semantic-code length GPT forward can vocode safely."""
        pos_emb = getattr(getattr(self, "gpt", None), "mel_pos_embedding", None)
        emb = getattr(pos_emb, "emb", None)
        if emb is not None and hasattr(emb, "num_embeddings"):
            # gpt.forward pads a stop token, then build_aligned_inputs adds a
            # start token before applying mel_pos_embedding, so full_codes can
            # use at most rows - 2 positions.
            return int(emb.num_embeddings) - 2
        gpt_max = getattr(getattr(self, "gpt", None), "max_mel_tokens", None)
        if gpt_max is not None:
            return int(gpt_max)
        return int(getattr(getattr(self.cfg, "gpt", {}), "max_mel_tokens", 1500))

    def _validate_continuation_text_tokens(
        self, num_text_tokens: int, max_text_tokens: int = 600
    ) -> None:
        """Reject continuation requests that exceed IndexTTS2 text capacity."""
        if num_text_tokens > max_text_tokens:
            raise ValueError(
                "Continuation full text is too long for IndexTTS2: "
                f"{num_text_tokens} BPE tokens > max_text_tokens={max_text_tokens}. "
                "Reduce Stage I text length or split the sample."
            )

    def _cap_generation_tokens_for_vocode(
        self,
        requested_max_mel_tokens: int,
        num_prefix_tokens: int,
    ) -> int:
        """Cap new tokens so prefix + continuation fits mel position embeddings."""
        max_full_codes = self._max_vocode_code_tokens()
        available_new_tokens = max_full_codes - num_prefix_tokens
        if available_new_tokens <= 0:
            raise ValueError(
                "Continuation prefix is too long for IndexTTS2 vocoder: "
                f"prefix_tokens={num_prefix_tokens}, "
                f"max_safe_full_codes={max_full_codes}. "
                "Reduce segment length or synthesize this item without token continuation."
            )
        if requested_max_mel_tokens > available_new_tokens:
            warnings.warn(
                "Reducing max_mel_tokens from "
                f"{requested_max_mel_tokens} to {available_new_tokens} so "
                f"prefix_tokens={num_prefix_tokens} plus generated tokens fits "
                f"max_safe_full_codes={max_full_codes}.",
                category=RuntimeWarning,
            )
            return available_new_tokens
        return requested_max_mel_tokens

    @torch.no_grad()
    def compute_speaker_conditioning(
        self, spk_audio_prompt: str, verbose: bool = False
    ) -> dict:
        """Compute speaker conditioning tensors from audio file.

        Returns a dict with keys:
            spk_cond  – w2v-bert embedding (tensor on self.device)
            s2mel_style – CAM++ global style vector
            s2mel_prompt – length-regulated prompt condition
            mel – reference mel spectrogram
        """
        audio, sr = self._load_and_cut_audio(spk_audio_prompt, 15, verbose)
        audio_22k = torchaudio.transforms.Resample(sr, 22050)(audio)
        audio_16k = torchaudio.transforms.Resample(sr, 16000)(audio)

        inputs = self.extract_features(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        spk_cond_emb = self.get_emb(input_features, attention_mask)

        _, S_ref = self.semantic_codec.quantize(spk_cond_emb)
        ref_mel = self.mel_fn(audio_22k.to(spk_cond_emb.device).float())
        ref_target_lengths = torch.LongTensor([ref_mel.size(2)]).to(ref_mel.device)
        feat = torchaudio.compliance.kaldi.fbank(
            audio_16k.to(ref_mel.device),
            num_mel_bins=80,
            dither=0,
            sample_frequency=16000,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        style = self.campplus_model(feat.unsqueeze(0))

        prompt_condition = self.s2mel.models["length_regulator"](
            S_ref, ylens=ref_target_lengths, n_quantizers=3, f0=None
        )[0]

        return {
            "spk_cond": spk_cond_emb,
            "s2mel_style": style,
            "s2mel_prompt": prompt_condition,
            "mel": ref_mel,
        }

    @torch.no_grad()
    def compute_emotion_conditioning(
        self, emo_audio_prompt: str, verbose: bool = False
    ) -> dict:
        """Compute emotion conditioning tensor from audio file.

        Returns a dict with key:
            emo_cond – w2v-bert emotion embedding (tensor on self.device)
        """
        emo_audio, _ = self._load_and_cut_audio(emo_audio_prompt, 15, verbose, sr=16000)
        emo_inputs = self.extract_features(
            emo_audio, sampling_rate=16000, return_tensors="pt"
        )
        emo_input_features = emo_inputs["input_features"].to(self.device)
        emo_attention_mask = emo_inputs["attention_mask"].to(self.device)
        emo_cond_emb = self.get_emb(emo_input_features, emo_attention_mask)
        return {"emo_cond": emo_cond_emb}

    def _prepare_emotion_preamble(
        self,
        emo_audio_prompt,
        spk_audio_prompt,
        emo_vector,
        emo_alpha,
        use_emo_text=False,
        emo_text=None,
        text=None,
    ):
        """Resolve emotion parameters into final (emo_audio_prompt, emo_vector, emo_alpha).

        Handles:
        - use_emo_text → QwenEmotion.inference → emo_vector
        - emo_vector scaling by emo_alpha
        - Fallback emo_audio_prompt to spk_audio_prompt when not using external reference
        """
        if use_emo_text or emo_vector is not None:
            # we're using a text or emotion vector guidance; so we must remove
            # "emotion reference voice", to ensure we use correct emotion mixing!
            emo_audio_prompt = None

        if use_emo_text:
            # automatically generate emotion vectors from text prompt
            if emo_text is None:
                emo_text = text  # use main text prompt
            emo_dict = self.qwen_emo.inference(emo_text)
            print(f"detected emotion vectors from text: {emo_dict}")
            # convert ordered dict to list of vectors; the order is VERY important!
            emo_vector = list(emo_dict.values())

        if emo_vector is not None:
            # we have emotion vectors; they can't be blended via alpha mixing
            # in the main inference process later, so we must pre-calculate
            # their new strengths here based on the alpha instead!
            emo_vector_scale = max(0.0, min(1.0, emo_alpha))
            if emo_vector_scale != 1.0:
                # scale each vector and truncate to 4 decimals (for nicer printing)
                emo_vector = [
                    int(x * emo_vector_scale * 10000) / 10000 for x in emo_vector
                ]
                print(f"scaled emotion vectors to {emo_vector_scale}x: {emo_vector}")

        if emo_audio_prompt is None:
            # we are not using any external "emotion reference voice"; use
            # speaker's voice as the main emotion reference audio.
            emo_audio_prompt = spk_audio_prompt
            # must always use alpha=1.0 when we don't have an external reference voice
            emo_alpha = 1.0

        return emo_audio_prompt, emo_vector, emo_alpha

    def _compute_emovec(
        self,
        spk_cond_emb,
        emo_cond_emb,
        emo_alpha,
        emo_vector=None,
        style=None,
        use_random=False,
    ):
        """Compute merged emotion vector from speaker and emotion conditioning.

        Returns emovec tensor.  If emo_vector is provided, also incorporates the
        emotion-matrix lookup.
        """
        with torch.amp.autocast(
            self.device, enabled=self.dtype is not None, dtype=self.dtype
        ):
            emovec = self.gpt.merge_emovec(
                spk_cond_emb,
                emo_cond_emb,
                torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                torch.tensor([emo_cond_emb.shape[-1]], device=self.device),
                alpha=emo_alpha,
            )
            if emo_vector is not None:
                weight_vector = torch.tensor(emo_vector, device=self.device)
                if use_random:
                    random_index = [random.randint(0, x - 1) for x in self.emo_num]
                else:
                    random_index = [
                        find_most_similar_cosine(style, tmp) for tmp in self.spk_matrix
                    ]
                emo_matrix = [
                    tmp[index].unsqueeze(0)
                    for index, tmp in zip(random_index, self.emo_matrix)
                ]
                emo_matrix_cat = torch.cat(emo_matrix, 0)
                emovec_mat = weight_vector.unsqueeze(1) * emo_matrix_cat
                emovec_mat = torch.sum(emovec_mat, 0).unsqueeze(0)
                emovec = emovec_mat + (1 - torch.sum(weight_vector)) * emovec
        return emovec

    def _ar_generate(
        self,
        text_tokens,
        spk_cond_emb,
        emo_cond_emb,
        emovec,
        prefix_tokens=None,
        max_mel_tokens=1500,
        **generation_kwargs,
    ):
        """AR text-to-semantic generation with optional continuation prefix.

        Args:
            text_tokens: [1, T_text] BPE text token IDs
            spk_cond_emb: speaker conditioning embedding
            emo_cond_emb: emotion conditioning embedding
            emovec: merged emotion vector (from _compute_emovec)
            prefix_tokens: optional [1, T_prefix] continuation prefix codes (0-8191)
            max_mel_tokens: max generation length
            **generation_kwargs: temperature, top_p, top_k, etc.

        Returns:
            new_codes: [1, T_new] newly generated codes (no SOS/EOS)
            speech_conditioning_latent: for subsequent GPT forward pass
        """
        do_sample = generation_kwargs.pop("do_sample", True)
        top_p = generation_kwargs.pop("top_p", 0.8)
        top_k = generation_kwargs.pop("top_k", 30)
        temperature = generation_kwargs.pop("temperature", 0.8)
        length_penalty = generation_kwargs.pop("length_penalty", 0.0)
        num_beams = generation_kwargs.pop("num_beams", 3)
        repetition_penalty = generation_kwargs.pop("repetition_penalty", 10.0)
        num_return_sequences = generation_kwargs.pop("num_return_sequences", 1)

        with torch.amp.autocast(
            self.device, enabled=self.dtype is not None, dtype=self.dtype
        ):
            codes, speech_conditioning_latent = self.gpt.inference_speech(
                spk_cond_emb,
                text_tokens,
                emo_cond_emb,
                cond_lengths=torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
                emo_cond_lengths=torch.tensor(
                    [emo_cond_emb.shape[-1]], device=self.device
                ),
                emo_vec=emovec,
                input_tokens=prefix_tokens,
                do_sample=do_sample,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                max_generate_length=max_mel_tokens,
                **generation_kwargs,
            )

        # Trim stop tokens
        code_lens = []
        max_code_len = 0
        for code in codes:
            if self.stop_mel_token not in code:
                code_len = len(code)
            else:
                len_ = (code == self.stop_mel_token).nonzero(as_tuple=False)[0]
                code_len = len_[0].item() if len_.numel() > 0 else len(code)
            code_lens.append(code_len)
            max_code_len = max(max_code_len, code_len)
        codes = codes[:, :max_code_len]
        code_lens = torch.LongTensor(code_lens).to(self.device)

        return codes, code_lens, speech_conditioning_latent

    def _vocode(
        self,
        codes,
        code_lens,
        text_tokens,
        speech_cond_latent,
        spk_cond_emb,
        emo_cond_emb,
        emovec,
        speaker_cache,
        use_speed=None,
    ):
        """Codes -> waveform via GPT forward + S2Mel CFM + BigVGAN.

        Args:
            codes: [1, T] semantic codes (0-8191)
            code_lens: [1] code lengths
            text_tokens: [1, T_text] BPE text token IDs
            speech_cond_latent: from gpt.get_conditioning or _ar_generate
            spk_cond_emb: speaker conditioning embedding (for cond_mel_lengths)
            emo_cond_emb: emotion conditioning embedding
            emovec: merged emotion vector
            speaker_cache: dict with keys s2mel_style, s2mel_prompt, mel
            use_speed: speed control tensor (optional)

        Returns:
            wav: waveform tensor [1, T_audio] on CPU
            timing: dict with gpt_forward_time, s2mel_time, bigvgan_time
        """
        prompt_condition = speaker_cache["s2mel_prompt"]
        ref_mel = speaker_cache["mel"]
        style = speaker_cache["s2mel_style"]

        if use_speed is None:
            use_speed = torch.zeros(spk_cond_emb.size(0)).to(self.device).long()

        m_start_time = time.perf_counter()
        with torch.amp.autocast(
            self.device, enabled=self.dtype is not None, dtype=self.dtype
        ):
            latent = self.gpt(
                speech_cond_latent,
                text_tokens,
                torch.tensor([text_tokens.shape[-1]], device=self.device),
                codes,
                code_lens,
                emo_cond_emb,
                cond_mel_lengths=torch.tensor(
                    [spk_cond_emb.shape[-1]], device=self.device
                ),
                emo_cond_mel_lengths=torch.tensor(
                    [emo_cond_emb.shape[-1]], device=self.device
                ),
                emo_vec=emovec,
                use_speed=use_speed,
            )
        gpt_forward_time = time.perf_counter() - m_start_time

        m_start_time = time.perf_counter()
        dtype = None
        with torch.amp.autocast(self.device, enabled=dtype is not None, dtype=dtype):
            latent = self.s2mel.models["gpt_layer"](latent)
            S_infer = self.semantic_codec.quantizer.vq2emb(codes.unsqueeze(1))
            S_infer = S_infer.transpose(1, 2) + latent
            target_lengths = (code_lens * 1.72).long()

            cond = self.s2mel.models["length_regulator"](
                S_infer, ylens=target_lengths, n_quantizers=3, f0=None
            )[0]
            cat_condition = torch.cat([prompt_condition, cond], dim=1)
            vc_target = self.s2mel.models["cfm"].inference(
                cat_condition,
                torch.LongTensor([cat_condition.size(1)]).to(cond.device),
                ref_mel,
                style,
                None,
                25,
                inference_cfg_rate=0.7,
            )
            vc_target = vc_target[:, :, ref_mel.size(-1) :]
        s2mel_time = time.perf_counter() - m_start_time

        m_start_time = time.perf_counter()
        with torch.amp.autocast(self.device, enabled=dtype is not None, dtype=dtype):
            wav = self.bigvgan(vc_target.float()).squeeze().unsqueeze(0)
            wav = wav.squeeze(1)
        bigvgan_time = time.perf_counter() - m_start_time

        wav = torch.clamp(32767 * wav, -32767.0, 32767.0)

        return wav, {
            "gpt_forward_time": gpt_forward_time,
            "s2mel_time": s2mel_time,
            "bigvgan_time": bigvgan_time,
        }

    def infer(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        speaker_cache=None,
        emotion_cache=None,
        return_cache=False,
        prefix_tokens=None,
        prefix_text=None,
        return_tokens=False,
        skip_vocode=False,
        **generation_kwargs,
    ):
        """Synthesize speech from text.

        Args:
            speaker_cache: Pre-computed speaker conditioning dict with keys
                spk_cond, s2mel_style, s2mel_prompt, mel (torch tensors).
                If provided, skips speaker conditioning computation.
            emotion_cache: Pre-computed emotion conditioning dict with key
                emo_cond (torch tensor). If provided, skips emotion
                conditioning computation.
            return_cache: If True, return a third element containing the
                conditioning cache dict that can be reused in future calls.
            prefix_tokens: Optional 1-D int64 numpy array or [1, T] tensor of
                continuation prefix codes (0-8191). When provided, AR generation
                uses these as prefix and text segmentation is skipped.
            prefix_text: Transcript of the prefix tokens. Required when
                prefix_tokens is provided. The full text for GPT will be
                ``prefix_text + " " + text``.
            return_tokens: If True, return generated semantic tokens alongside
                audio. When prefix_tokens is given, the returned tokens include
                the prefix.
            skip_vocode: If True, skip S2Mel + BigVGAN and return only tokens.
                Implies return_tokens=True. Also skips CAMPPlus/mel computation
                for speaker conditioning.
        """
        if skip_vocode:
            return_tokens = True

        # When return_cache is requested, pre-compute the cache so we can
        # pass it into the generator AND return it to the caller.
        _cache_out = None
        if return_cache:
            # Resolve emotion audio the same way the generator does:
            _emo_audio = emo_audio_prompt
            if use_emo_text or emo_vector is not None:
                _emo_audio = None
            if _emo_audio is None:
                _emo_audio = spk_audio_prompt

            if speaker_cache is None:
                speaker_cache = self.compute_speaker_conditioning(
                    spk_audio_prompt, verbose
                )
            if emotion_cache is None:
                emotion_cache = self.compute_emotion_conditioning(_emo_audio, verbose)

            _cache_out = {"speaker": speaker_cache, "emotion": emotion_cache}

        result = list(
            self.infer_generator(
                spk_audio_prompt,
                text,
                output_path,
                emo_audio_prompt,
                emo_alpha,
                emo_vector,
                use_emo_text,
                emo_text,
                use_random,
                interval_silence,
                verbose,
                max_text_tokens_per_segment,
                speaker_cache=speaker_cache,
                emotion_cache=emotion_cache,
                prefix_tokens=prefix_tokens,
                prefix_text=prefix_text,
                return_tokens=return_tokens,
                skip_vocode=skip_vocode,
                **generation_kwargs,
            )
        )[0]

        if return_cache:
            # Add continuation data to cache if tokens were returned
            if return_tokens and result is not None:
                if isinstance(result, tuple) and len(result) == 2:
                    _audio_result, _tokens_out = result
                elif isinstance(result, dict) and "tokens" in result:
                    # skip_vocode mode: result is the tokens dict itself
                    _tokens_out = result
                else:
                    _tokens_out = None
                if (
                    _tokens_out is not None
                    and _tokens_out.get("tokens") is not None
                ):
                    # The continuation text is the full text that was synthesized
                    cont_text = text
                    if prefix_text is not None:
                        cont_text = self._join_continuation_text(prefix_text, text)
                    _cache_out["continuation"] = {
                        "tokens": _tokens_out["tokens"],
                        "text": cont_text,
                    }
            return result, _cache_out
        return result

    def infer_generator(
        self,
        spk_audio_prompt,
        text,
        output_path,
        emo_audio_prompt=None,
        emo_alpha=1.0,
        emo_vector=None,
        use_emo_text=False,
        emo_text=None,
        use_random=False,
        interval_silence=200,
        verbose=False,
        max_text_tokens_per_segment=120,
        speaker_cache=None,
        emotion_cache=None,
        return_cache=False,
        prefix_tokens=None,
        prefix_text=None,
        return_tokens=False,
        skip_vocode=False,
        **generation_kwargs,
    ):
        print(">> starting inference...")
        self._set_gr_progress(0, "starting inference...")
        if verbose:
            print(
                f"origin text:{text}, spk_audio_prompt:{spk_audio_prompt}, "
                f"emo_audio_prompt:{emo_audio_prompt}, emo_alpha:{emo_alpha}, "
                f"emo_vector:{emo_vector}, use_emo_text:{use_emo_text}, "
                f"emo_text:{emo_text}"
            )
        start_time = time.perf_counter()

        if skip_vocode:
            return_tokens = True

        # --- Continuation mode: prepare prefix tokens tensor ---
        continuation_mode = prefix_tokens is not None
        prefix_tokens_tensor = None
        num_prefix_tokens = 0
        if continuation_mode:
            if prefix_text is None:
                raise ValueError(
                    "prefix_text is required when prefix_tokens is provided"
                )
            if isinstance(prefix_tokens, np.ndarray):
                prefix_tokens_tensor = torch.tensor(
                    prefix_tokens, dtype=torch.long, device=self.device
                ).unsqueeze(0)
            elif isinstance(prefix_tokens, torch.Tensor):
                prefix_tokens_tensor = prefix_tokens.to(
                    dtype=torch.long, device=self.device
                )
                if prefix_tokens_tensor.ndim == 1:
                    prefix_tokens_tensor = prefix_tokens_tensor.unsqueeze(0)
            else:
                raise TypeError(
                    f"prefix_tokens must be numpy array or torch tensor, got {type(prefix_tokens)}"
                )
            num_prefix_tokens = prefix_tokens_tensor.shape[1]
            print(f">> continuation mode: {num_prefix_tokens} prefix tokens")

        # --- Emotion preamble ---
        _emo_text_input = text
        if continuation_mode:
            _emo_text_input = self._join_continuation_text(prefix_text, text)
        emo_audio_prompt, emo_vector, emo_alpha = self._prepare_emotion_preamble(
            emo_audio_prompt,
            spk_audio_prompt,
            emo_vector,
            emo_alpha,
            use_emo_text,
            emo_text,
            _emo_text_input,
        )

        # --- Speaker conditioning ---
        if speaker_cache is not None:
            spk_cache = speaker_cache
        else:
            # TODO: when skip_vocode=True and no emo_vector, we could skip
            # CAMPPlus/mel/s2mel_prompt and only compute spk_cond.
            spk_cache = self.compute_speaker_conditioning(spk_audio_prompt, verbose)
        spk_cond_emb = spk_cache["spk_cond"]

        # --- Emotion conditioning ---
        if emotion_cache is not None:
            emo_cond_emb = emotion_cache["emo_cond"]
        else:
            emo_computed = self.compute_emotion_conditioning(emo_audio_prompt, verbose)
            emo_cond_emb = emo_computed["emo_cond"]

        # --- Emovec (computed once, reused per segment) ---
        emovec = self._compute_emovec(
            spk_cond_emb,
            emo_cond_emb,
            emo_alpha,
            emo_vector=emo_vector,
            style=spk_cache["s2mel_style"],
            use_random=use_random,
        )

        # --- Text processing ---
        self._set_gr_progress(0.1, "text processing...")
        if continuation_mode:
            # In continuation mode, use full text (prefix_text + text) without segmentation
            full_text = self._join_continuation_text(prefix_text, text)
            text_tokens_list = self.tokenizer.tokenize(full_text)
            text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
            num_text_tokens = len(text_token_ids)
            if verbose:
                print(f">> full text (continuation): '{full_text}'")
                print(f">> text tokens: {num_text_tokens}")
            self._validate_continuation_text_tokens(num_text_tokens)
            # Single segment containing all text tokens
            segments = [self.tokenizer.convert_ids_to_tokens(text_token_ids)]
        else:
            text_tokens_list = self.tokenizer.tokenize(text)
            segments = self.tokenizer.split_segments(
                text_tokens_list,
                max_text_tokens_per_segment,
                quick_streaming_tokens=0,
            )
            text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        segments_count = len(segments)

        if not continuation_mode and self.tokenizer.unk_token_id in text_token_ids:
            print(
                f"  >> Warning: input text contains {text_token_ids.count(self.tokenizer.unk_token_id)} unknown tokens (id={self.tokenizer.unk_token_id}):"
            )
            print(
                "     Tokens which can't be encoded: ",
                [
                    t
                    for t, id in zip(text_tokens_list, text_token_ids)
                    if id == self.tokenizer.unk_token_id
                ],
            )
            print(
                f"     Consider updating the BPE model or modifying the text to avoid unknown tokens."
            )

        if verbose:
            print(
                "text_tokens_list:",
                text_tokens_list if not continuation_mode else "(continuation mode)",
            )
            print("segments count:", segments_count)
            print("max_text_tokens_per_segment:", max_text_tokens_per_segment)
            print(*segments, sep="\n")

        # Extract generation params once (they're re-passed per segment)
        gen_params = {
            "do_sample": generation_kwargs.pop("do_sample", True),
            "top_p": generation_kwargs.pop("top_p", 0.8),
            "top_k": generation_kwargs.pop("top_k", 30),
            "temperature": generation_kwargs.pop("temperature", 0.8),
            "length_penalty": generation_kwargs.pop("length_penalty", 0.0),
            "num_beams": generation_kwargs.pop("num_beams", 3),
            "repetition_penalty": generation_kwargs.pop("repetition_penalty", 10.0),
            "num_return_sequences": 1,
        }
        max_mel_tokens = generation_kwargs.pop("max_mel_tokens", 1500)
        max_mel_tokens = self._cap_generation_tokens_for_vocode(
            max_mel_tokens,
            num_prefix_tokens if continuation_mode else 0,
        )
        sampling_rate = 22050

        wavs = []
        all_codes_list = []  # for return_tokens
        gpt_gen_time = 0
        gpt_forward_time = 0
        s2mel_time = 0
        bigvgan_time = 0
        has_warned = False
        for seg_idx, sent in enumerate(segments):
            self._set_gr_progress(
                0.2 + 0.7 * seg_idx / segments_count,
                f"speech synthesis {seg_idx + 1}/{segments_count}...",
            )

            text_tokens = self.tokenizer.convert_tokens_to_ids(sent)
            text_tokens = torch.tensor(
                text_tokens, dtype=torch.int32, device=self.device
            ).unsqueeze(0)
            if verbose:
                print(text_tokens)
                print(
                    f"text_tokens shape: {text_tokens.shape}, text_tokens type: {text_tokens.dtype}"
                )
                text_token_syms = self.tokenizer.convert_ids_to_tokens(
                    text_tokens[0].tolist()
                )
                print(
                    "text_token_syms is same as segment tokens", text_token_syms == sent
                )

            # In continuation mode, pass prefix_tokens for the (single) segment
            seg_prefix = prefix_tokens_tensor if continuation_mode else None

            m_start_time = time.perf_counter()
            with torch.no_grad():
                codes, code_lens, speech_conditioning_latent = self._ar_generate(
                    text_tokens,
                    spk_cond_emb,
                    emo_cond_emb,
                    emovec,
                    prefix_tokens=seg_prefix,
                    max_mel_tokens=max_mel_tokens,
                    **gen_params,
                    **generation_kwargs,
                )
                gpt_gen_time += time.perf_counter() - m_start_time

                if not has_warned and (codes[:, -1] == self.stop_mel_token).any():
                    # _ar_generate trims at stop token, so if stop token is still
                    # the last value it means max_mel_tokens was NOT reached.
                    pass
                # Check if max tokens was reached (no stop token found)
                if not has_warned and code_lens[0].item() >= max_mel_tokens:
                    if not continuation_mode:
                        warnings.warn(
                            "IndexTTS2 generation reached max_mel_tokens before "
                            f"emitting a stop token: max_mel_tokens={max_mel_tokens}, "
                            f"input_text_tokens={text_tokens.shape[1]}. "
                            "Output may be truncated.",
                            category=RuntimeWarning,
                        )
                        has_warned = True
                    else:
                        raise ValueError(
                            "IndexTTS2 generation reached max_mel_tokens before "
                            f"emitting a stop token: max_mel_tokens={max_mel_tokens}, "
                            f"input_text_tokens={text_tokens.shape[1]}. "
                            "Reduce text length or split the sample."
                        )

                # In continuation mode, build full codes (prefix + continuation)
                if continuation_mode:
                    new_tokens_count = code_lens[0].item()
                    new_codes = codes[:, :new_tokens_count]
                    full_codes = torch.cat([prefix_tokens_tensor, new_codes], dim=1)
                    max_full_codes = self._max_vocode_code_tokens()
                    if full_codes.shape[1] > max_full_codes:
                        raise ValueError(
                            "Continuation token length exceeds IndexTTS2 vocoder "
                            f"limit: prefix_tokens={num_prefix_tokens}, "
                            f"new_tokens={new_tokens_count}, "
                            f"full_codes={full_codes.shape[1]}, "
                            f"max_safe_full_codes={max_full_codes}."
                        )
                    full_code_lens = torch.tensor(
                        [full_codes.shape[1]], device=self.device, dtype=torch.long
                    )
                    # Use full codes for vocode and token output
                    vocode_codes = full_codes
                    vocode_code_lens = full_code_lens
                    if return_tokens:
                        all_codes_list.append(full_codes[0].cpu().numpy())
                else:
                    vocode_codes = codes
                    vocode_code_lens = code_lens
                    if return_tokens:
                        trimmed = codes[0, : code_lens[0].item()]
                        all_codes_list.append(trimmed.cpu().numpy())

                if verbose:
                    print(codes, type(codes))
                    print(f"fix codes shape: {codes.shape}, codes type: {codes.dtype}")
                    print(f"code len: {code_lens}")

                if not skip_vocode:
                    wav, timing = self._vocode(
                        vocode_codes,
                        vocode_code_lens,
                        text_tokens,
                        speech_conditioning_latent,
                        spk_cond_emb,
                        emo_cond_emb,
                        emovec,
                        spk_cache,
                    )
                    gpt_forward_time += timing["gpt_forward_time"]
                    s2mel_time += timing["s2mel_time"]
                    bigvgan_time += timing["bigvgan_time"]

                    if verbose:
                        print(
                            f"wav shape: {wav.shape}",
                            "min:",
                            wav.min(),
                            "max:",
                            wav.max(),
                        )
                    wavs.append(wav.cpu())
        end_time = time.perf_counter()

        if skip_vocode:
            # Return only tokens, no audio
            all_tokens = (
                np.concatenate(all_codes_list, axis=0)
                if all_codes_list
                else np.array([], dtype=np.int64)
            )
            print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
            print(f">> Total inference time: {end_time - start_time:.2f} seconds")
            print(f">> Total tokens: {len(all_tokens)}")
            yield {"tokens": all_tokens, "token_rate": 50.0}
            return

        self._set_gr_progress(0.9, "saving audio...")
        wavs = self.insert_interval_silence(
            wavs, sampling_rate=sampling_rate, interval_silence=interval_silence
        )
        wav = torch.cat(wavs, dim=1)
        wav_length = wav.shape[-1] / sampling_rate
        print(f">> gpt_gen_time: {gpt_gen_time:.2f} seconds")
        print(f">> gpt_forward_time: {gpt_forward_time:.2f} seconds")
        print(f">> s2mel_time: {s2mel_time:.2f} seconds")
        print(f">> bigvgan_time: {bigvgan_time:.2f} seconds")
        print(f">> Total inference time: {end_time - start_time:.2f} seconds")
        print(f">> Generated audio length: {wav_length:.2f} seconds")
        print(f">> RTF: {(end_time - start_time) / wav_length:.4f}")

        # Build tokens output if requested
        tokens_out = None
        if return_tokens:
            tokens_out = {
                "tokens": np.concatenate(all_codes_list, axis=0)
                if all_codes_list
                else np.array([], dtype=np.int64),
                "token_rate": 50.0,
            }
            if continuation_mode:
                tokens_out["num_prefix_tokens"] = num_prefix_tokens

        # save audio
        wav = wav.cpu()  # to cpu
        if output_path:
            if os.path.isfile(output_path):
                os.remove(output_path)
                print(">> remove old wav file:", output_path)
            if os.path.dirname(output_path) != "":
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            torchaudio.save(output_path, wav.type(torch.int16), sampling_rate)
            print(">> wav file saved to:", output_path)
            if return_tokens:
                yield output_path, tokens_out
            else:
                yield output_path
        else:
            wav_data = wav.type(torch.int16)
            wav_data = wav_data.numpy().T
            if return_tokens:
                yield (sampling_rate, wav_data), tokens_out
            else:
                yield (sampling_rate, wav_data)

    @torch.no_grad()
    def extract_semantic_tokens(self, audio_path: str) -> tuple:
        """
        Extract 50Hz semantic tokens from audio.

        Args:
            audio_path: path to audio file.

        Returns:
            (codes, token_rate) where codes is a 1-D numpy int64 array and
            token_rate is approximately 50.
        """
        audio, sr = self._load_and_cut_audio(audio_path, 30, verbose=False, sr=16000)
        audio_16k = audio  # already 16kHz

        inputs = self.extract_features(
            audio_16k, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs["input_features"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        emb = self.get_emb(input_features, attention_mask)
        codes, _ = self.semantic_codec.quantize(emb)  # codes: (1, T) scalar indices

        codes_np = codes.squeeze(0).cpu().numpy().astype(np.int64)
        token_rate = 50.0  # ≈ input_sr / hop_size for w2v-bert 20ms frames
        return codes_np, token_rate

    @torch.no_grad()
    def synthesize_from_tokens(
        self,
        codes_np,
        text: str,
        spk_audio_prompt: str,
        emo_audio_prompt: str = None,
        emo_alpha: float = 1.0,
        emo_vector=None,
        use_emo_text: bool = False,
        emo_text: str = None,
        max_text_tokens_per_segment: int = 120,
        interval_silence: int = 200,
        speaker_cache=None,
        emotion_cache=None,
        return_cache=False,
        **generation_kwargs,
    ):
        """
        Synthesize audio from pre-computed semantic codes.

        Unlike the normal infer path, the GPT autoregressive generation is
        replaced by the provided codes. We still need GPT's forward pass to get
        the latent representation, plus s2mel + BigVGAN for audio.

        Args:
            codes_np: 1-D int64 numpy array of semantic codes.
            text: corresponding text (for GPT forward pass).
            spk_audio_prompt: speaker reference audio path.
            emo_audio_prompt: emotion reference audio path (optional).
            emo_alpha: emotion blending strength.
            emo_vector: 8-dim emotion vector (optional).
            use_emo_text: use emo_text for emotion detection.
            emo_text: text for emotion detection.
            max_text_tokens_per_segment: text segmentation limit.
            interval_silence: silence between segments (ms).
            speaker_cache: Pre-computed speaker conditioning dict (optional).
            emotion_cache: Pre-computed emotion conditioning dict (optional).
            return_cache: If True, return cache as third element.

        Returns:
            (sample_rate, wav_data) with wav_data as int16 numpy array.
            If return_cache=True: ((sample_rate, wav_data), cache_dict).
        """
        sampling_rate = 22050

        # --- Emotion preamble ---
        emo_audio_prompt, emo_vector, emo_alpha = self._prepare_emotion_preamble(
            emo_audio_prompt,
            spk_audio_prompt,
            emo_vector,
            emo_alpha,
            use_emo_text,
            emo_text,
            text,
        )

        # --- Speaker conditioning ---
        if speaker_cache is not None:
            spk_cache = speaker_cache
        else:
            spk_cache = self.compute_speaker_conditioning(spk_audio_prompt)
        spk_cond_emb = spk_cache["spk_cond"]

        # --- Emotion conditioning ---
        if emotion_cache is not None:
            emo_cond_emb = emotion_cache["emo_cond"]
        else:
            emo_computed = self.compute_emotion_conditioning(emo_audio_prompt)
            emo_cond_emb = emo_computed["emo_cond"]

        # --- Emovec ---
        emovec = self._compute_emovec(
            spk_cond_emb,
            emo_cond_emb,
            emo_alpha,
            emo_vector=emo_vector,
            style=spk_cache["s2mel_style"],
        )

        # --- Text tokenization ---
        text_tokens_list = self.tokenizer.tokenize(text)
        all_text_token_ids = self.tokenizer.convert_tokens_to_ids(text_tokens_list)
        text_tokens_tensor = torch.tensor(
            all_text_token_ids, dtype=torch.int32, device=self.device
        ).unsqueeze(0)

        # --- Provided codes ---
        codes = torch.tensor(codes_np, dtype=torch.long, device=self.device).unsqueeze(
            0
        )
        code_lens = torch.tensor([codes.shape[1]], device=self.device, dtype=torch.long)

        # --- Get speech_conditioning_latent (no AR generation) ---
        with torch.amp.autocast(
            self.device, enabled=self.dtype is not None, dtype=self.dtype
        ):
            speech_conditioning_latent = self.gpt.get_conditioning(
                spk_cond_emb.transpose(1, 2),
                torch.tensor([spk_cond_emb.shape[-1]], device=self.device),
            )

        # --- Vocode (GPT forward + S2Mel + BigVGAN) ---
        wav, _timing = self._vocode(
            codes,
            code_lens,
            text_tokens_tensor,
            speech_conditioning_latent,
            spk_cond_emb,
            emo_cond_emb,
            emovec,
            spk_cache,
        )

        wav_data = wav.cpu().type(torch.int16).numpy().T

        if return_cache:
            _spk_cache = speaker_cache if speaker_cache is not None else spk_cache
            _emo_cache = emotion_cache if emotion_cache is not None else emo_computed
            cache_out = {"speaker": _spk_cache, "emotion": _emo_cache}
            return (sampling_rate, wav_data), cache_out
        return (sampling_rate, wav_data)
