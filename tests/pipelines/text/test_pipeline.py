from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/text_edit/defaults")

import base64
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from speech_edit.pipelines.text.audio_timeline import (
    mono_float_to_wav_bytes,
)
from speech_edit.pipelines.text import pipeline as pipeline_module
from speech_edit.pipelines.text.pipeline import (
    PipelineConfig,
    GenerationSkipped,
    SampleSkipped,
    build_tagged_plan_from_st,
    execute_tagged_plan,
    _alignment_language,
    generate_tagged_generations_attempt,
    split_tagged_instruction_into_single_op_steps,
    synthesize_target_audio,
)
from speech_edit.pipelines.text.parquet_rows import (
    text_edit_rows,
)
from speech_edit.pipelines.text.generation_plan import (
    GenerationSpec,
)
from speech_edit.pipelines.text.span_planner import SpanPlan
from speech_edit.data.text_markup import (
    render_source,
    render_target,
)


def configured_pipeline(**overrides):
    values = {name: PARAMETERS[name] for name in PipelineConfig.__dataclass_fields__ if name != "output_dir"}
    return PipelineConfig(**(values | overrides))


NL_PAYLOAD = {
    "simple_en": ["Replace 'brave' with 'kind'.", "Change 'brave' to 'kind'."],
    "simple_zh": ["把'brave'改成'kind'。", "将'brave'替换为'kind'。"],
    "hard_en": [
        "Please make the wording say 'kind' instead of 'brave'.",
        "Swap the word 'brave' for 'kind' in the sentence.",
    ],
    "hard_zh": [
        "请把句子里的'brave'换成'kind'。",
        "把原来的'brave'调整为'kind'，意思要保持清楚。",
    ],
}


REVERSE_NL_PAYLOAD = {
    "simple_en": ["Replace 'kind' with 'brave'.", "Change 'kind' to 'brave'."],
    "simple_zh": ["把'kind'改成'brave'。", "将'kind'替换为'brave'。"],
    "hard_en": [
        "Please restore the word 'brave' where it currently says 'kind'.",
        "Change the edited wording back from 'kind' to 'brave'.",
    ],
    "hard_zh": [
        "请把'kind'恢复成'brave'。",
        "将修改后的'kind'换回原来的'brave'。",
    ],
}


class FakeLLM:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(args[0])
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</sub> world",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )
        return json.dumps(
            {
                "instructions": {
                    "000": {
                        "nl": REVERSE_NL_PAYLOAD,
                    }
                }
            }
        )


class InvalidLLM:
    def __init__(self):
        self.calls = []

    def generate(self, *args, **kwargs):
        self.calls.append(args[0])
        return json.dumps({"generations": {"000": {"st": "<sub targ=\"x\">missing</sub>", "nl": NL_PAYLOAD}}})


def test_llm_json_call_uses_service_default_backend():
    class CapturingLLM:
        def __init__(self):
            self.kwargs = None

        def generate(self, *args, **kwargs):
            self.kwargs = kwargs
            return "{}"

    llm = CapturingLLM()

    assert (
        pipeline_module._call_llm_json(
            "hello",
            llm_client=llm,
            config=configured_pipeline(output_dir=".", evaluate=False),
        )
        == "{}"
    )
    assert "backend" not in llm.kwargs
    assert "model" not in llm.kwargs
    assert llm.kwargs["json_mode"] is True


class FakeAligner:
    def __init__(self):
        self.calls = []

    def align(self, audio_bytes, text, **kwargs):
        call = {
            "text": text,
            "queries": kwargs.get("queries", []),
            "word_level": kwargs.get("word_level", False),
            "backend": kwargs.get("backend"),
        }
        self.calls.append(call)
        words = [
            {
                "word": word,
                "text": word,
                "start": round(index * 0.1, 3),
                "end": round(index * 0.1 + 0.08, 3),
            }
            for index, word in enumerate(text.split())
        ]
        query_count = len(kwargs.get("queries") or [])
        if query_count:
            base_start, base_end = (0.2, 0.4) if text == "hello brave world" else (0.22, 0.48)
            answers = [
                {"start": round(base_start + index * 0.2, 3), "end": round(base_end + index * 0.2, 3)}
                for index in range(query_count)
            ]
        else:
            answers = [{"start": 0.2, "end": 0.4}] if text == "hello brave world" else [{"start": 0.22, "end": 0.48}]
        if kwargs.get("queries") is not None:
            response = {"answers": answers}
            if kwargs.get("word_level"):
                response["alignments"] = words
                response["backend"] = kwargs.get("backend")
            return response
        if kwargs.get("word_level"):
            return words
        return {"answers": answers}


class FakeF5:
    def __init__(self):
        self.calls = []

    def sample(self, cond_waveform, text, duration=None, **kwargs):
        self.calls.append(
            {
                "cond_shape": tuple(cond_waveform.shape),
                "text": text,
                "duration": duration,
                "kwargs": kwargs,
            }
        )
        waveform = np.asarray(cond_waveform, dtype=np.float32).squeeze()
        return {
            "waveform_tensor": {
                "data": base64.b64encode(waveform.tobytes()).decode("utf-8"),
                "shape": list(waveform.shape),
                "dtype": str(waveform.dtype),
            },
            "sample_rate": 24000,
            "metadata": {"duration_source": "cond_waveform"},
        }


class FakeWER:
    def compute(self, reference, hypothesis, *, language="auto"):
        return {"error_rate": 0.125, "metric": "wer"}


class FakeASR:
    def transcribe(self, audio_bytes, **kwargs):
        return {"text": "hello kind world", "language": kwargs.get("language")}


class FakeUTMOS:
    def predict(self, audio_bytes, **kwargs):
        return {"score": 4.25}


class FakeSpkSim:
    def similarity(self, audio_a, audio_b, **kwargs):
        return {"similarity": 0.875}


class FakeWDTW:
    def __init__(self):
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "wdtw": 0.25,
            "distance": 1.0,
            "normalizer": 4.0,
            "mode": "preserved",
            "status": "ok",
        }


def test_execute_tagged_plan_writes_alignment_into_text_edit_rows(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )
    config = configured_pipeline(
        output_dir=str(tmp_path),
        span_level_weights={"word": 1.0},
        op_count_weights={1: 1.0},
        require_del_ins_prob=0.0,
        evaluate=False,
        expand_radius=0.0,
        merge_gap=0.0,
    )

    execute_tagged_plan(
        segment=segment,
        sample_id="sample_000000",
        plan=build_tagged_plan_from_st(
            source_text="hello brave world",
            st="hello <sub targ=\"kind\">brave</sub> world",
        ),
        config=config,
        aligner=FakeAligner(),
        f5=FakeF5(),
        include_artifacts=True,
    )

    sample_dir = tmp_path / "sample_000000"
    forward = json.loads(
        (sample_dir / "edit_pairs" / "original_to_000" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    reverse = json.loads(
        (sample_dir / "edit_pairs" / "000_to_original" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    sample_meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "align" not in forward["segments"]["src"]
    assert forward["segments"]["src"]["alignment"]["level"] == "word"
    assert forward["segments"]["src"]["alignment"]["items"][1]["text"] == "brave"
    assert forward["segments"]["tgt"]["alignment"]["items"][1]["text"] == "kind"
    assert reverse["segments"]["src"]["alignment"] == forward["segments"]["tgt"]["alignment"]
    assert reverse["segments"]["tgt"]["alignment"] == forward["segments"]["src"]["alignment"]

    rows = text_edit_rows(
        sample_id="sample_000000",
        original_audio_bytes=source_audio,
        generated_audio_bytes={"000": source_audio},
        pair_results=[forward, reverse],
        metadata=sample_meta,
        dataset_name="test",
        sample_rate=24000,
    )
    row_segments = json.loads(rows[0]["segments"])
    assert row_segments["src"]["alignment"] == forward["segments"]["src"]["alignment"]
    assert row_segments["tgt"]["alignment"] == forward["segments"]["tgt"]["alignment"]


def test_split_tagged_instruction_into_single_op_steps_preserves_progression():
    st = (
        "<sub targ=\"After work\">After dinner</sub>, "
        "I read<ins> quietly</ins> and <del>slept</del>."
    )

    steps = split_tagged_instruction_into_single_op_steps(st)

    assert steps == [
        "<sub targ=\"After work\">After dinner</sub>, I read and slept.",
        "After work, I read<ins> quietly</ins> and slept.",
        "After work, I read quietly and <del>slept</del>.",
    ]
    for left, right in zip(steps, steps[1:]):
        assert render_target(left) == render_source(right)
    assert render_target(steps[-1]) == "After work, I read quietly and."


def test_execute_plan_per_op_calls_f5_once_per_edit(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )

    class MultiAligner(FakeAligner):
        def align(self, audio_bytes, text, **kwargs):
            if kwargs.get("word_level"):
                return super().align(audio_bytes, text, **kwargs)
            self.calls.append({"text": text, "queries": kwargs.get("queries", [])})
            answers = []
            for index, _ in enumerate(kwargs.get("queries", [])):
                answers.append({"start": 0.2 + index * 0.2, "end": 0.3 + index * 0.2})
            return {"answers": answers}

    f5 = FakeF5()
    plan = build_tagged_plan_from_st(
        source_text="hello brave world",
        st="hello <sub targ=\"kind\">brave</sub><ins> wide</ins> world",
    )
    execute_tagged_plan(
        segment=segment,
        sample_id="sample",
        plan=plan,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            synthesis_mode="per_op",
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        aligner=MultiAligner(),
        f5=f5,
        persist=False,
        include_artifacts=True,
    )

    assert len(f5.calls) == 2


def test_execute_plan_one_shot_uses_single_f5_call(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )

    class MultiAligner(FakeAligner):
        def align(self, audio_bytes, text, **kwargs):
            if kwargs.get("word_level"):
                return super().align(audio_bytes, text, **kwargs)
            self.calls.append({"text": text, "queries": kwargs.get("queries", [])})
            return {
                "answers": [
                    {"start": 0.2 + index * 0.2, "end": 0.3 + index * 0.2}
                    for index, _ in enumerate(kwargs.get("queries", []))
                ]
            }

    f5 = FakeF5()
    plan = build_tagged_plan_from_st(
        source_text="hello brave world",
        st="hello <sub targ=\"kind\">brave</sub><ins> wide</ins> world",
    )
    execute_tagged_plan(
        segment=segment,
        sample_id="sample",
        plan=plan,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            synthesis_mode="one_shot",
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        aligner=MultiAligner(),
        f5=f5,
        persist=False,
        include_artifacts=True,
    )

    assert len(f5.calls) == 1
    assert len(f5.calls[0]["kwargs"]["edit_spans_sec"]) == 2


def test_execute_tagged_plan_from_external_st_skips_llm_and_can_skip_persistence(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )
    plan = build_tagged_plan_from_st(
        source_text="hello brave world",
        st="hello <sub targ=\"kind\">brave</sub> world",
        forward_nl={"simple": ["Replace brave."], "hard": []},
    )
    fake_f5 = FakeF5()

    result = execute_tagged_plan(
        segment=segment,
        sample_id="external_000",
        plan=plan,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        aligner=FakeAligner(),
        f5=fake_f5,
        persist=False,
        include_artifacts=True,
    )

    assert result["generations"] == 1
    assert result["llm_forward_attempts"] == 0
    assert result["metadata"]["generations"]["000"]["target_text"] == "hello kind world"
    assert result["pair_results"][1]["instructions"]["nl"]["simple"] == [
        "Reverse the edit: replace 'kind' with 'brave'.",
        "Change the generated text back by applying this reverse edit: replace 'kind' with 'brave'.",
        "反向修改：把'kind'改成'brave'。",
        "把生成文本改回原文：把'kind'改成'brave'。",
    ]
    assert result["generated_audio_bytes"]["000"]
    assert not (tmp_path / "external_000").exists()


def test_execute_tagged_plan_computes_pair_wdtw_once_and_shares_result(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )
    plan = build_tagged_plan_from_st(
        source_text="hello brave world",
        st="hello <sub targ=\"kind\">brave</sub> world",
    )
    fake_wdtw = FakeWDTW()

    result = execute_tagged_plan(
        segment=segment,
        sample_id="external_eval_000",
        plan=plan,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=True,
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        aligner=FakeAligner(),
        f5=FakeF5(),
        asr=FakeASR(),
        wer=FakeWER(),
        wdtw=fake_wdtw,
        persist=False,
        include_artifacts=True,
    )

    forward, reverse = result["pair_results"]
    alignment_cache = result["alignment_cache"]
    assert len(fake_wdtw.calls) == 1
    assert fake_wdtw.calls[0]["language"] == "eng"
    assert fake_wdtw.calls[0]["source_alignments"] == alignment_cache["source_word_alignments"]
    assert fake_wdtw.calls[0]["target_alignments"] == alignment_cache["generated_word_alignments_by_generation"]["000"]
    assert "aligner_url" not in fake_wdtw.calls[0]
    assert alignment_cache["aligner_backend"] == "qwen3"
    assert forward["eval_results"]["wdtw_result"] is reverse["eval_results"]["wdtw_result"]
    assert forward["eval_results"]["wdtw"] == 0.25
    assert "wer_result" not in forward["eval_results"]
    assert "local_wer_result" in forward["eval_results"]


def test_evaluate_synthesized_result_uses_cached_alignments_without_eval_aligner(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )
    plan = build_tagged_plan_from_st(
        source_text="hello brave world",
        st="hello <sub targ=\"kind\">brave</sub> world",
    )
    synth_result = execute_tagged_plan(
        segment=segment,
        sample_id="external_eval_cached_000",
        plan=plan,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        aligner=FakeAligner(),
        f5=FakeF5(),
        persist=False,
        include_artifacts=True,
    )

    fake_wdtw = FakeWDTW()
    item = pipeline_module.SynthesizedWorkItem(
        sample_id="external_eval_cached_000",
        segment=segment,
        attempt_index=0,
        result=synth_result,
    )
    evaluated = pipeline_module._evaluate_synthesized_result(
        item,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=True,
            expand_radius=0.0,
            merge_gap=0.0,
        ),
        asr=FakeASR(),
        wer=FakeWER(),
        wdtw=fake_wdtw,
    )

    forward, reverse = evaluated.result["pair_results"]
    assert len(fake_wdtw.calls) == 1
    assert fake_wdtw.calls[0]["source_alignments"] == synth_result["alignment_cache"]["source_word_alignments"]
    assert fake_wdtw.calls[0]["target_alignments"] == synth_result["alignment_cache"]["generated_word_alignments_by_generation"]["000"]
    assert "aligner_url" not in fake_wdtw.calls[0]
    assert forward["eval_results"]["wdtw_result"] is reverse["eval_results"]["wdtw_result"]
    assert forward["eval_results"]["wdtw"] == 0.25


def test_prompt_contracts_cover_common_llm_failure_modes():
    from speech_edit.pipelines.text.prompts import (
        build_forward_generation_prompt,
        build_reverse_instruction_prompt,
    )

    prompt = build_forward_generation_prompt(
        "他今天来到学校,然后开始上课。",
        specs=[GenerationSpec("000", "phrase", 2)],
    )
    assert "Start from the canonical source text copied character-for-character" in prompt
    assert "Every <del>...</del>, <ins>...</ins>, and <sub" in prompt
    assert "Do not express one replacement as <del>old</del><ins>new</ins>" in prompt
    assert "The language rule applies to the instruction wording" in prompt
    assert "render_source(st)" not in prompt
    assert "Valid Forward Examples" in prompt
    assert '<sub targ=\\"kind\\">' in prompt
    assert "exactly these generation ids" in prompt

    reverse_prompt = build_reverse_instruction_prompt(
        [
            {
                "generation_id": "000",
                "source_text": "hello kind world",
                "target_text": "hello brave world",
                "st": "hello <sub targ=\"brave\">kind</sub> world",
                "tag_diff": [{"kind": "sub", "source_text": "kind", "target_text": "brave"}],
            }
        ]
    )
    assert "Valid Reverse Instruction Example" in reverse_prompt
    assert '<sub targ=\\"brave\\">' in reverse_prompt
    assert "Do not output or rewrite `st`" in reverse_prompt
    assert "Instruction field names select the instruction wording language" in reverse_prompt
    assert "render_source(st)" not in reverse_prompt


def test_prompt_includes_del_ins_requirement_only_when_requested():
    from speech_edit.pipelines.text.prompts import (
        build_forward_generation_prompt,
    )

    normal = build_forward_generation_prompt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
    )
    required = build_forward_generation_prompt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1, require_del_ins=True)],
    )

    assert "Include at least one <ins> or <del> edit." not in normal
    assert "Include at least one <ins> or <del> edit." in required
    assert '"require_del_ins": true' in required


def test_prompt_includes_large_edit_block_only_when_configured():
    from speech_edit.pipelines.text.prompts import (
        build_forward_generation_prompt,
    )

    normal = build_forward_generation_prompt(
        "hello brave world",
        specs=[GenerationSpec("000", "arbitrary", 2)],
    )
    large = build_forward_generation_prompt(
        "hello brave world",
        specs=[GenerationSpec("000", "arbitrary", 2)],
        min_edit_distance_pct=0.75,
    )

    assert "Large-Edit Requirement" not in normal
    assert "Large-Edit Requirement" in large
    assert "substantially different from the source" in large
    assert "Keep part of the source outside edit tags" in large
    assert "at least 75.0% of the source or target text" in large
    assert "if set" not in large.lower()


def test_required_del_ins_rejects_sub_only_generation(tmp_path: Path):
    class SubOnlyLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</sub> world",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    llm = SubOnlyLLM()
    with pytest.raises(SampleSkipped) as exc_info:
        generate_tagged_generations_attempt(
            "hello brave world",
            specs=[GenerationSpec("000", "word", 1, require_del_ins=True)],
            llm_client=llm,
            config=configured_pipeline(
                output_dir=str(tmp_path),
                evaluate=False,
                llm_max_retries=1,
            ),
        )

    assert exc_info.value.stage == "forward_llm"
    assert any("del or ins" in error for error in exc_info.value.validation_errors)


def test_required_del_ins_accepts_delete_generation(tmp_path: Path):
    class DeleteLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <del>brave</del> world",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    generations, stats = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1, require_del_ins=True)],
        llm_client=DeleteLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            llm_max_retries=1,
        ),
    )

    assert [generation.st for generation in generations] == ["hello <del>brave</del> world"]
    assert stats["forward_attempts"] == 1


def test_min_target_edit_distance_rejects_small_edits(tmp_path: Path):
    class SmallEditLLM:
        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</sub> world",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    with pytest.raises(SampleSkipped) as exc_info:
        generate_tagged_generations_attempt(
            "hello brave world",
            specs=[GenerationSpec("000", "arbitrary", 1)],
            llm_client=SmallEditLLM(),
            config=configured_pipeline(
                output_dir=str(tmp_path),
                evaluate=False,
                min_edit_distance_pct=0.75,
                llm_max_retries=1,
            ),
        )

    assert any(
        "below required 75.0% on either side" in error
        for error in exc_info.value.validation_errors
    )


def test_min_target_edit_distance_accepts_large_multi_edit(tmp_path: Path):
    class LargeEditLLM:
        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": (
                                "hello <sub targ=\"entirely different target\">brave</sub> "
                                "<sub targ=\"sentence\">world</sub>"
                            ),
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    generations, stats = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "arbitrary", 2)],
        llm_client=LargeEditLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            min_edit_distance_pct=0.75,
            min_preserved_source_ratio=0.10,
            llm_max_retries=1,
        ),
    )

    assert render_target(generations[0].st) == "hello entirely different target sentence"
    assert stats["forward_attempts"] == 1


def test_min_target_edit_distance_accepts_target_side_threshold(tmp_path: Path):
    class TargetSideLargeEditLLM:
        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "a b c d e <del>f g h i j</del>",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    generations, _ = generate_tagged_generations_attempt(
        "a b c d e f g h i j",
        specs=[GenerationSpec("000", "arbitrary", 1)],
        llm_client=TargetSideLargeEditLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            min_edit_distance_pct=0.75,
            min_preserved_source_ratio=0.10,
            llm_max_retries=1,
        ),
    )

    assert render_target(generations[0].st) == "a b c d e"


def test_min_preserved_source_ratio_rejects_full_non_sentence_rewrite(tmp_path: Path):
    class FullRewriteLLM:
        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "<sub targ=\"entirely different target sentence\">hello brave world</sub>",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    with pytest.raises(SampleSkipped) as exc_info:
        generate_tagged_generations_attempt(
            "hello brave world",
            specs=[GenerationSpec("000", "arbitrary", 1)],
            llm_client=FullRewriteLLM(),
            config=configured_pipeline(
                output_dir=str(tmp_path),
                evaluate=False,
                min_edit_distance_pct=0.75,
                min_preserved_source_ratio=0.10,
                llm_max_retries=1,
            ),
        )

    assert any(
        "preserved source ratio" in error
        for error in exc_info.value.validation_errors
    )


def test_min_preserved_source_ratio_does_not_reject_sentence_level(tmp_path: Path):
    class SentenceRewriteLLM:
        def generate(self, *args, **kwargs):
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "<sub targ=\"entirely different target sentence\">hello brave world</sub>",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    generations, _ = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "sentence", 1)],
        llm_client=SentenceRewriteLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            min_edit_distance_pct=0.75,
            min_preserved_source_ratio=0.10,
            llm_max_retries=1,
        ),
    )

    assert render_source(generations[0].st) == "hello brave world"


def test_forward_nl_template_ignores_invalid_llm_payload(tmp_path: Path):
    class InvalidForwardNlLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</sub> world",
                            "nl": {"simple_en": ["bad"]},
                        }
                    }
                }
            )

    generations, stats = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=InvalidForwardNlLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            llm_max_retries=1,
            forward_nl_mode="template",
        ),
    )

    assert stats["forward_nl_source"] == "template"
    assert generations[0].forward_nl["simple"] == [
        "Edit the text: replace 'brave' with 'kind'.",
        "Apply this text edit: replace 'brave' with 'kind'.",
        "修改文本：把'brave'改成'kind'。",
        "按要求编辑：把'brave'改成'kind'。",
    ]


def test_forward_nl_llm_mode_preserves_llm_payload(tmp_path: Path):
    generations, stats = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=FakeLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            llm_max_retries=1,
            forward_nl_mode="llm",
        ),
    )

    assert stats["forward_nl_source"] == "llm"
    assert generations[0].forward_nl == {
        "simple": [
            "Replace 'brave' with 'kind'.",
            "Change 'brave' to 'kind'.",
            "把'brave'改成'kind'。",
            "将'brave'替换为'kind'。",
        ],
        "hard": [
            "Please make the wording say 'kind' instead of 'brave'.",
            "Swap the word 'brave' for 'kind' in the sentence.",
            "请把句子里的'brave'换成'kind'。",
            "把原来的'brave'调整为'kind'，意思要保持清楚。",
        ],
    }


def test_forward_nl_mode_rejects_unknown_value(tmp_path: Path):
    with pytest.raises(SampleSkipped) as exc_info:
        generate_tagged_generations_attempt(
            "hello brave world",
            specs=[GenerationSpec("000", "word", 1)],
            llm_client=FakeLLM(),
            config=configured_pipeline(
                output_dir=str(tmp_path),
                evaluate=False,
                llm_max_retries=1,
                forward_nl_mode="mystery",
            ),
        )
    assert "forward_nl_mode" in " ".join(exc_info.value.validation_errors)


def test_forward_attempt_rejects_partial_valid_generations(tmp_path: Path):
    class AlternatingPartialLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            if len(self.calls) == 1:
                payload = {
                    "000": {
                        "st": "hello <sub targ=\"kind\">brave</sub> world",
                        "nl": NL_PAYLOAD,
                    },
                    "001": {
                        "st": "hello <sub targ=\"oops\">missing</sub> world",
                        "nl": NL_PAYLOAD,
                    },
                }
            elif len(self.calls) == 2:
                payload = {
                    "000": {
                        "st": "hello <sub targ=\"oops\">missing</sub> world",
                        "nl": NL_PAYLOAD,
                    },
                    "001": {
                        "st": "hello <del>brave </del>world",
                        "nl": NL_PAYLOAD,
                    },
                }
            else:
                payload = {
                    "000": {
                        "st": "hello <sub targ=\"oops\">missing</sub> world",
                        "nl": NL_PAYLOAD,
                    },
                    "001": {
                        "st": "hello <sub targ=\"oops\">missing</sub> world",
                        "nl": NL_PAYLOAD,
                    },
                }
            return json.dumps({"generations": payload})

    llm = AlternatingPartialLLM()
    specs = [
        GenerationSpec("000", "word", 1),
        GenerationSpec("001", "word", 1),
    ]
    try:
        from speech_edit.pipelines.text.pipeline import (
            generate_tagged_generations_attempt,
        )

        generate_tagged_generations_attempt(
            "hello brave world",
            specs=specs,
            llm_client=llm,
            config=configured_pipeline(
                output_dir=str(tmp_path),
                evaluate=False,
                llm_max_retries=3,
            ),
        )
    except SampleSkipped as exc:
        assert exc.stage == "forward_llm"
        assert exc.forward_attempts == 1
    else:
        raise AssertionError("partial valid generations must not be merged across attempts")
    assert len(llm.calls) == 1


def test_template_reverse_nl_avoids_reverse_llm_call(tmp_path: Path):
    llm = FakeLLM()
    plan = pipeline_module.generate_tagged_edit_plan_with_specs_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=llm,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            require_del_ins_prob=0.0,
        ),
    )

    assert len(llm.calls) == 1
    assert plan.reverse_llm_stats["source"] == "template"
    assert plan.reverse_llm_stats["reverse_attempts"] == 0
    reverse_nl = plan.reverse_nl_by_generation["000"]
    assert len(reverse_nl["simple"]) == 4
    assert len(reverse_nl["hard"]) == 4
    assert all("<sub" not in text for text in reverse_nl["simple"] + reverse_nl["hard"])
    assert "'kind'" in reverse_nl["simple"][0]
    assert "'brave'" in reverse_nl["simple"][0]


def test_plan_attempt_uses_single_generate_not_generate_batch(tmp_path: Path):
    class NoBatchLLM(FakeLLM):
        def generate_batch(self, *args, **kwargs):
            raise AssertionError("plan buffering must use single generate calls")

    llm = NoBatchLLM()
    plan = pipeline_module.generate_tagged_edit_plan_with_specs_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=llm,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            require_del_ins_prob=0.0,
        ),
    )

    assert len(llm.calls) == 1
    assert plan.generations[0].st == "hello <sub targ=\"kind\">brave</sub> world"
    assert plan.forward_llm_stats["forward_attempts"] == 1


def test_alignment_language_maps_common_codes():
    assert _alignment_language("hello", None) == "eng"
    assert _alignment_language("你好", None) == "zho"
    assert _alignment_language("hello", "en") == "eng"
    assert _alignment_language("你好", "zh") == "zho"
    assert _alignment_language("你好", "zho") == "zho"
    assert _alignment_language("你好", "cmn") == "zho"


def test_synthesize_uses_prepend_context_when_under_f5_limit():
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    f5 = FakeF5()
    span_plan = SpanPlan(
        source_spans=[[0.2, 0.4]],
        target_spans=[[0.2, 0.4]],
        planned_spans=[],
        target_duration=1.0,
        duration_mode="estimate",
    )
    wav_bytes, metadata = synthesize_target_audio(
        st="hello <sub targ=\"kind\">brave</sub> world",
        source_audio_bytes=source_audio,
        span_plan=span_plan,
        f5=f5,
        config=configured_pipeline(output_dir=".", evaluate=False),
        trace_id="trace",
        span_level="word",
    )
    assert wav_bytes
    assert metadata["prepend_context_used"] is True
    assert f5.calls[0]["duration"] is None
    assert "duration_sec" not in f5.calls[0]["kwargs"]
    assert f5.calls[0]["kwargs"]["edit_spans_sec"] == [[1.2, 1.4]]
    assert f5.calls[0]["text"] == ["hello brave world hello kind world"]
    assert f5.calls[0]["kwargs"]["output_format"] == "both"
    assert f5.calls[0]["kwargs"]["seed"] == 42
    assert f5.calls[0]["kwargs"]["cfg_strength"] == 2.0
    assert f5.calls[0]["kwargs"]["source_crossfade_sec"] == 0.02
    assert metadata["f5_payload_hash"]
    assert metadata["generated_wav_sha256"]


def test_synthesize_decodes_f5_both_waveform_response():
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    f5 = FakeF5()
    span_plan = SpanPlan(
        source_spans=[[0.2, 0.4]],
        target_spans=[[0.2, 0.4]],
        planned_spans=[],
        target_duration=1.0,
        duration_mode="estimate",
    )

    wav_bytes, metadata = synthesize_target_audio(
        st="hello <sub targ=\"kind\">brave</sub> world",
        source_audio_bytes=source_audio,
        span_plan=span_plan,
        f5=f5,
        config=configured_pipeline(output_dir=".", evaluate=False),
        trace_id="trace",
        span_level="word",
    )

    assert wav_bytes
    assert metadata["f5_output_format"] == "both"
    assert metadata["f5_service_metadata"] == {"duration_source": "cond_waveform"}


def test_synthesize_disables_prepend_when_allowed_over_f5_limit():
    source_audio = mono_float_to_wav_bytes(np.zeros(29 * 24000, dtype=np.float32), 24000)
    f5 = FakeF5()
    span_plan = SpanPlan(
        source_spans=[[25.0, 29.0]],
        target_spans=[[25.0, 28.0]],
        planned_spans=[],
        target_duration=28.0,
        duration_mode="estimate",
    )
    _, metadata = synthesize_target_audio(
        st="hello <sub targ=\"kind\">brave</sub> world",
        source_audio_bytes=source_audio,
        span_plan=span_plan,
        f5=f5,
        config=configured_pipeline(
            output_dir=".",
            evaluate=False,
            min_unmasked_context_sec=0.1,
            min_unmasked_context_ratio=0.01,
        ),
        trace_id="trace",
        span_level="word",
    )
    assert metadata["prepend_context_used"] is False
    assert metadata["prepend_context_disabled_reason"] == "duration_limit"
    assert f5.calls[0]["kwargs"]["edit_spans_sec"] == [[25.0, 28.0]]


def test_synthesize_fails_when_unmasked_context_is_too_small_and_prepend_exceeds_limit():
    source_audio = mono_float_to_wav_bytes(np.zeros(29 * 24000, dtype=np.float32), 24000)
    span_plan = SpanPlan(
        source_spans=[[0.0, 28.5]],
        target_spans=[[0.0, 28.5]],
        planned_spans=[],
        target_duration=29.0,
        duration_mode="estimate",
    )
    try:
        synthesize_target_audio(
            st="hello <sub targ=\"kind\">brave</sub> world",
            source_audio_bytes=source_audio,
            span_plan=span_plan,
            f5=FakeF5(),
            config=configured_pipeline(output_dir=".", evaluate=False),
            trace_id="trace",
            span_level="word",
        )
    except GenerationSkipped as exc:
        assert "prepend context is required" in str(exc)
    else:
        raise AssertionError("low unmasked context should require prepend and fail")


def test_synthesize_fails_sentence_when_prepend_exceeds_f5_limit():
    source_audio = mono_float_to_wav_bytes(np.zeros(29 * 24000, dtype=np.float32), 24000)
    span_plan = SpanPlan(
        source_spans=[[0.0, 28.0]],
        target_spans=[[0.0, 28.0]],
        planned_spans=[],
        target_duration=29.0,
        duration_mode="estimate",
    )
    try:
        synthesize_target_audio(
            st="<sub targ=\"new sentence\">old sentence</sub>",
            source_audio_bytes=source_audio,
            span_plan=span_plan,
            f5=FakeF5(),
            config=configured_pipeline(output_dir=".", evaluate=False),
            trace_id="trace",
            span_level="sentence",
        )
    except GenerationSkipped as exc:
        assert "prepend context is required" in str(exc)
    else:
        raise AssertionError("sentence generation should fail when prepend is required")


def test_synthesize_fails_sentence_when_prepend_is_disabled():
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    span_plan = SpanPlan(
        source_spans=[[0.0, 1.0]],
        target_spans=[[0.0, 1.0]],
        planned_spans=[],
        target_duration=1.0,
        duration_mode="estimate",
    )
    try:
        synthesize_target_audio(
            st="<sub targ=\"new sentence\">old sentence</sub>",
            source_audio_bytes=source_audio,
            span_plan=span_plan,
            f5=FakeF5(),
            config=configured_pipeline(
                output_dir=".",
                evaluate=False,
                prepend_context_enabled=False,
            ),
            trace_id="trace",
            span_level="sentence",
        )
    except GenerationSkipped as exc:
        assert "disabled by config" in str(exc)
    else:
        raise AssertionError("sentence generation should fail when prepend is disabled")


def test_synthesize_fails_low_context_when_prepend_is_disabled():
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    span_plan = SpanPlan(
        source_spans=[[0.0, 0.9]],
        target_spans=[[0.0, 0.95]],
        planned_spans=[],
        target_duration=1.0,
        duration_mode="estimate",
    )
    try:
        synthesize_target_audio(
            st="<sub targ=\"kind\">brave</sub> world",
            source_audio_bytes=source_audio,
            span_plan=span_plan,
            f5=FakeF5(),
            config=configured_pipeline(
                output_dir=".",
                evaluate=False,
                prepend_context_enabled=False,
            ),
            trace_id="trace",
            span_level="word",
        )
    except GenerationSkipped as exc:
        assert "disabled by config" in str(exc)
    else:
        raise AssertionError("low unmasked context should fail when prepend is disabled")


def test_synthesize_fails_when_target_timeline_exceeds_f5_limit():
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    span_plan = SpanPlan(
        source_spans=[[0.0, 1.0]],
        target_spans=[[0.0, 31.0]],
        planned_spans=[],
        target_duration=31.0,
        duration_mode="estimate",
    )
    try:
        synthesize_target_audio(
            st="<sub targ=\"new sentence\">old sentence</sub>",
            source_audio_bytes=source_audio,
            span_plan=span_plan,
            f5=FakeF5(),
            config=configured_pipeline(output_dir=".", evaluate=False),
            trace_id="trace",
            span_level="word",
        )
    except GenerationSkipped as exc:
        assert "target timeline exceeds" in str(exc)
    else:
        raise AssertionError("target timeline over 30s should fail")


def test_text_edit_parquet_rows_keep_eval_transcripts():
    audio = mono_float_to_wav_bytes(np.zeros(2400, dtype=np.float32), 24000)
    rows = text_edit_rows(
        sample_id="sample_1",
        original_audio_bytes=audio,
        generated_audio_bytes={"000": audio},
        pair_results=[
            {
                "source_id": "original",
                "target_id": "000",
                "pair_dir_name": "original_to_000",
                "instructions": {"st": "hello <sub targ=\"kind\">brave</sub> world"},
                "requested_span_level": "word",
                "requested_op_count": 1,
                "target_edit_distance": {
                    "source_pct": 33.33333333333333,
                    "target_pct": 33.33333333333333,
                    "max_pct": 33.33333333333333,
                    "metric": "mixed",
                },
                "segments": {
                    "src": {"label": "hello brave world"},
                    "tgt": {"label": "hello kind world"},
                },
                "eval_results": {
                    "src": {
                        "transcript": "hello brave world",
                        "wer": {
                            "error_rate": 0.0,
                            "metric": "wer",
                            "backend": "wer_service",
                        },
                        "utmos": {"score": 4.1},
                    },
                    "tgt": {
                        "transcript": "hello kind world",
                        "wer": {
                            "error_rate": 0.1,
                            "metric": "wer",
                            "backend": "wer_service",
                        },
                        "utmos": {"score": 4.2},
                    },
                    "delta": {"wer": 0.1, "wer_pct": 10.0},
                    "spk_sim": {"similarity": 0.875},
                    "local_wer_result": {
                        "edit_region_wer": {"error_rate": 0.0},
                        "non_edit_region_wer": {"error_rate": 0.1},
                    },
                    "wdtw": 0.25,
                    "wdtw_result": {
                        "wdtw": 0.25,
                        "distance": 1.0,
                        "normalizer": 4.0,
                        "mode": "preserved",
                    },
                },
            }
        ],
        metadata={"config": {}},
        dataset_name="Unit-TextEdit",
        sample_rate=24000,
    )

    eval_results = json.loads(rows[0]["eval_results"])
    edit_metadata = json.loads(rows[0]["edit_metadata"])
    assert eval_results["src"]["transcript"] == "hello brave world"
    assert eval_results["tgt"]["transcript"] == "hello kind world"
    assert "asr_text" not in eval_results["tgt"]
    assert "wer_result" not in eval_results
    assert "wer_result" not in eval_results["tgt"]
    assert "local_wer_result" not in eval_results["tgt"]
    assert eval_results["src"]["wer"]["error_rate"] == 0.0
    assert eval_results["src"]["utmos"]["score"] == 4.1
    assert eval_results["tgt"]["utmos"]["score"] == 4.2
    assert eval_results["delta"]["wer_pct"] == 10.0
    assert eval_results["spk_sim"]["similarity"] == 0.875
    assert eval_results["local_wer_result"]["edit_region_wer"]["error_rate"] == 0.0
    assert eval_results["wdtw"] == 0.25
    assert eval_results["wdtw_result"]["mode"] == "preserved"
    assert edit_metadata["requested_span_level"] == "word"
    assert edit_metadata["requested_op_count"] == 1
    assert edit_metadata["target_edit_distance_pct"] == pytest.approx(33.3333333)
    assert edit_metadata["target_edit_distance_source_pct"] == pytest.approx(33.3333333)
    assert edit_metadata["target_edit_distance_target_pct"] == pytest.approx(33.3333333)
    assert edit_metadata["target_edit_distance_metric"] == "mixed"
    assert edit_metadata["target_edit_distance_denominator"] == "source_or_target"


def test_evaluate_pair_if_enabled_keeps_spk_sim_result():
    audio = mono_float_to_wav_bytes(np.zeros(2400, dtype=np.float32), 24000)
    result = pipeline_module.evaluate_pair_if_enabled(
        source_audio_bytes=audio,
        target_audio_bytes=audio,
        source_text="hello brave world",
        target_text="hello kind world",
        sample_language="en",
        config=configured_pipeline(output_dir=".", evaluate=True),
        asr=FakeASR(),
        wer=FakeWER(),
        utmos=FakeUTMOS(),
        spk_sim=FakeSpkSim(),
        shared_wdtw_result={"wdtw": 0.25, "mode": "preserved"},
    )

    assert result["src"]["wer"]["error_rate"] == 0.125
    assert result["tgt"]["wer"]["error_rate"] == 0.125
    assert result["spk_sim"]["similarity"] == 0.875
    assert result["src"]["utmos"]["score"] == 4.25
    assert result["tgt"]["utmos"]["score"] == 4.25
    assert result["wdtw"] == 0.25
    assert result["wdtw_result"]["mode"] == "preserved"


def test_execute_tagged_plan_writes_bidirectional_tagged_layout(tmp_path: Path):
    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="hello brave world",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={"language": "en"},
    )
    fake_f5 = FakeF5()
    llm = FakeLLM()
    aligner = FakeAligner()
    config = configured_pipeline(
        output_dir=str(tmp_path),
        num_generations=1,
        span_level_weights={"word": 1.0},
        op_count_weights={1: 1.0},
        require_del_ins_prob=0.0,
        evaluate=False,
        expand_radius=0.0,
        merge_gap=0.0,
        forward_nl_mode="llm",
    )
    plan = pipeline_module.generate_tagged_edit_plan_with_specs_attempt(
        segment.text,
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=llm,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            num_generations=1,
            span_level_weights={"word": 1.0},
            op_count_weights={1: 1.0},
            require_del_ins_prob=0.0,
            evaluate=False,
            forward_nl_mode="llm",
        ),
    )

    result = execute_tagged_plan(
        segment=segment,
        sample_id="sample_000000",
        plan=plan,
        config=config,
        aligner=aligner,
        f5=fake_f5,
    )

    assert result["generations"] == 1
    sample_dir = tmp_path / "sample_000000"
    assert (sample_dir / "audios" / "original.wav").exists()
    assert (sample_dir / "audios" / "generated_000.wav").exists()

    forward_dir = sample_dir / "edit_pairs" / "original_to_000"
    reverse_dir = sample_dir / "edit_pairs" / "000_to_original"
    assert (forward_dir / "source.wav").is_symlink()
    assert (forward_dir / "target.wav").is_symlink()
    assert (reverse_dir / "source.wav").is_symlink()
    assert (reverse_dir / "target.wav").is_symlink()

    forward = json.loads((forward_dir / "metadata.json").read_text(encoding="utf-8"))
    reverse = json.loads((reverse_dir / "metadata.json").read_text(encoding="utf-8"))
    sample_meta = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))

    assert "wave_id" not in forward
    assert forward["generation_id"] == "000"
    assert "generated_waves" not in sample_meta
    assert "generations" in sample_meta
    assert sample_meta["config"]["num_generations"] == 1
    assert forward["instructions"]["st"] == "hello <sub targ=\"kind\">brave</sub> world"
    assert reverse["instructions"]["st"] == "hello <sub targ=\"brave\">kind</sub> world"
    assert len(forward["instructions"]["nl"]["simple"]) == 4
    assert len(forward["instructions"]["nl"]["hard"]) == 4
    assert len(reverse["instructions"]["nl"]["simple"]) == 4
    assert len(reverse["instructions"]["nl"]["hard"]) == 4
    assert "Replace 'brave' with 'kind'." in forward["instructions"]["nl"]["simple"]
    assert "Reverse the edit: replace 'kind' with 'brave'." in reverse["instructions"]["nl"]["simple"]
    assert forward["segments"]["src"]["label"] == "hello brave world"
    assert forward["segments"]["tgt"]["label"] == "hello kind world"
    assert forward["segments"]["src"]["edit_time_spans"] == [[0.2, 0.4]]
    assert forward["segments"]["tgt"]["edit_time_spans"][0][0] == 0.2
    assert reverse["segments"]["src"]["edit_time_spans"] == [[0.22, 0.48]]
    assert reverse["segments"]["tgt"]["edit_time_spans"] == [[0.2, 0.4]]

    assert fake_f5.calls[0]["kwargs"]["preserve_unmasked_waveform"] is True
    assert fake_f5.calls[0]["kwargs"]["edit_spans_sec"] == [[1.2, 1.46667]]
    assert sample_meta["generations"]["000"]["synthesis"]["target_edit_spans_sec"] == (
        forward["segments"]["tgt"]["edit_time_spans"]
    )



def test_forward_nl_language_content_is_not_a_hard_fail(tmp_path: Path):
    class ChineseOnlyEnglishSlotLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            bad_nl = dict(NL_PAYLOAD)
            bad_nl["simple_en"] = ["把'brave'改成'kind'。", "将'brave'替换为'kind'。"]
            bad_nl["hard_en"] = [
                "请把句子里的'brave'换成'kind'。",
                "把原来的'brave'调整为'kind'。",
            ]
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</sub> world",
                            "nl": bad_nl,
                        }
                    }
                }
            )

    llm = ChineseOnlyEnglishSlotLLM()
    generations, _ = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=llm,
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            span_level_weights={"word": 1.0},
            op_count_weights={1: 1.0},
            require_del_ins_prob=0.0,
            llm_max_retries=1,
            forward_nl_mode="llm",
        ),
    )
    assert [generation.st for generation in generations] == ['hello <sub targ="kind">brave</sub> world']
    assert len(llm.calls) == 1



def test_forward_nl_allows_unquoted_chinese_edit_literals(tmp_path: Path):
    class UnquotedChineseLiteralLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            nl = {
                "simple_en": ["Replace '朋友' with 世界.", "Change 朋友 to 世界."],
                "simple_zh": ["把'朋友'改成'世界'。", "将'朋友'替换为'世界'。"],
                "hard_en": [
                    "Please replace 朋友 with 世界 and keep the rest unchanged.",
                    "Update the wording by changing 朋友 into 世界.",
                ],
                "hard_zh": ["请把'朋友'改成'世界'。", "把原来的'朋友'调整为'世界'。"],
            }
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "你好<sub targ=\"世界\">朋友</sub>",
                            "nl": nl,
                        }
                    }
                }
            )

    generations, _ = generate_tagged_generations_attempt(
        "你好朋友",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=UnquotedChineseLiteralLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            span_level_weights={"word": 1.0},
            op_count_weights={1: 1.0},
            require_del_ins_prob=0.0,
            llm_max_retries=1,
            forward_nl_mode="llm",
        ),
    )
    assert [generation.st for generation in generations] == ['你好<sub targ="世界">朋友</sub>']



def test_forward_repairs_common_sub_closing_tag_typo(tmp_path: Path):
    class ClosingTypoLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            return json.dumps(
                {
                    "generations": {
                        "000": {
                            "st": "hello <sub targ=\"kind\">brave</a> world",
                            "nl": NL_PAYLOAD,
                        }
                    }
                }
            )

    generations, _ = generate_tagged_generations_attempt(
        "hello brave world",
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=ClosingTypoLLM(),
        config=configured_pipeline(
            output_dir=str(tmp_path),
            evaluate=False,
            span_level_weights={"word": 1.0},
            op_count_weights={1: 1.0},
            require_del_ins_prob=0.0,
            llm_max_retries=1,
            forward_nl_mode="llm",
        ),
    )
    assert [generation.st for generation in generations] == ['hello <sub targ="kind">brave</sub> world']



def test_generate_tagged_edit_plan_skips_before_audio_when_reverse_nl_fails(tmp_path: Path):
    class ReverseBadLLM(FakeLLM):
        def generate(self, *args, **kwargs):
            if not self.calls:
                return super().generate(*args, **kwargs)
            self.calls.append(args[0])
            return json.dumps({"instructions": {"000": {"nl": {"simple_en": []}}}})

    llm = ReverseBadLLM()

    try:
        pipeline_module.generate_tagged_edit_plan_with_specs_attempt(
            "hello brave world",
            specs=[GenerationSpec("000", "word", 1)],
            llm_client=llm,
            config=configured_pipeline(
                output_dir=str(tmp_path),
                num_generations=1,
                span_level_weights={"word": 1.0},
                op_count_weights={1: 1.0},
                require_del_ins_prob=0.0,
                evaluate=False,
                llm_max_retries=3,
                reverse_nl_mode="llm",
            ),
        )
    except SampleSkipped as exc:
        assert exc.stage == "reverse_llm"
        assert exc.reverse_attempts == 3
    else:
        raise AssertionError("reverse NL failures should skip the sample")



def test_canonical_llm_output_projects_to_surface_spaces(tmp_path: Path):
    class SpaceLLM:
        def __init__(self):
            self.calls = []

        def generate(self, *args, **kwargs):
            self.calls.append(args[0])
            if len(self.calls) == 1:
                return json.dumps(
                    {
                        "generations": {
                            "000": {
                                "st": "你好,<sub targ=\"world\">世界</sub>hello",
                                "nl": NL_PAYLOAD,
                            }
                        }
                    }
                )
            return json.dumps({"instructions": {"000": {"nl": REVERSE_NL_PAYLOAD}}})

    source_audio = mono_float_to_wav_bytes(np.zeros(24000, dtype=np.float32), 24000)
    segment = SimpleNamespace(
        clip_id="clip_1",
        text="  你好, 世界  hello ",
        audio_bytes=source_audio,
        duration=1.0,
        sample_rate=24000,
        metadata={
            "language": "zh",
            "source_parquet_path": "/data/source.parquet",
            "source_row_idx": 7,
            "source_file_index": 3,
        },
    )
    config = configured_pipeline(
        output_dir=str(tmp_path),
        num_generations=1,
        span_level_weights={"word": 1.0},
        op_count_weights={1: 1.0},
        require_del_ins_prob=0.0,
        evaluate=False,
        expand_radius=0.0,
        merge_gap=0.0,
    )
    plan = pipeline_module.generate_tagged_edit_plan_with_specs_attempt(
        segment.text,
        specs=[GenerationSpec("000", "word", 1)],
        llm_client=SpaceLLM(),
        config=config,
    )
    execute_tagged_plan(
        segment=segment,
        sample_id="sample_000000",
        plan=plan,
        config=config,
        aligner=FakeAligner(),
        f5=FakeF5(),
    )
    forward = json.loads(
        (tmp_path / "sample_000000" / "edit_pairs" / "original_to_000" / "metadata.json").read_text(
            encoding="utf-8"
        )
    )
    assert render_source(forward["instructions"]["st"]) == "  你好, 世界  hello "
    assert forward["instructions"]["st"] == (
        "  你好, <sub targ=\"world\">世界</sub>  hello "
    )
    sample_meta = json.loads(
        (tmp_path / "sample_000000" / "metadata.json").read_text(encoding="utf-8")
    )
    assert sample_meta["source_metadata"]["source_parquet_path"] == "/data/source.parquet"
    assert forward["source_metadata"]["source_row_idx"] == 7
