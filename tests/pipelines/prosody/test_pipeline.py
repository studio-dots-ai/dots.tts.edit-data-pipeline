from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/prosody_edit/defaults")

import base64
import copy
import io
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from speech_edit.pipelines.prosody.editor import (
    EditResult,
    apply_edits_from_service,
    numpy_to_wav,
    resolve_psola_indextts2_cco_effective_semitones,
    resolve_working_sample_rate,
    _splice_replacements,
    smooth_splice_boundaries,
    wav_to_numpy,
)
from speech_edit.evaluation.audio import (
    EvalMetrics,
    PairEvalResult,
)
from speech_edit.pipelines.prosody.effect_metrics import (
    compute_region_effect_metrics,
)
from speech_edit.pipelines.prosody.generation_policy import (
    run_all_filters,
    short_rate_risk,
    tiny_stopword_rate_risk,
)
from speech_edit.pipelines.prosody.input_generator import (
    LLMCallError,
    LLM_SERVICE_TIMEOUT_SEC,
    PipelineInput,
    Stage0Sample,
    generate_input_from_sampled_data,
    normalize_parameter_policy,
)
import speech_edit.pipelines.prosody.input_generator as input_generator
import speech_edit.pipelines.prosody.pipeline as pipeline_mod
from speech_edit.pipelines.prosody.prompts import build_prompt
from speech_edit.pipelines.prosody.sample_output import (
    build_sample_metadata,
    save_sample_outputs,
)
from speech_edit.pipelines.prosody.parquet_rows import (
    ProsodyParquetRowProcessor,
)
from speech_edit.pipelines.prosody.tag_utils import (
    ORIGINAL_WAVE_ID,
    WaveEditPair,
    WaveGroup,
    build_wave_group,
    invert_structural_instruction,
    pair_key,
    parse_tags,
    summarize_tagged_text,
    strip_tags,
    validate_tagged_text,
)
from speech_edit.runtime.models import (
    AdapterContext,
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
    StageSpec,
)
from tests.service_graph_fixtures import direct_services
from speech_edit.pipelines.prosody.stage_adapter import ProsodyStageAdapter


def test_prosody_yaml_parameters():
    assert PARAMETERS["pitch_method"] == "psola+indextts2_cco"
    assert PARAMETERS["psola_indextts2_cco_pitch_boost"] == "mul:1.6"
    assert PARAMETERS["rate_method"] == "wsola+indextts2"
    assert PARAMETERS["smooth_policy"] == "auto"
    assert PARAMETERS["crossfade_ms"] == 40.0
    assert PARAMETERS["smooth_margin_sec"] == 0.1


@dataclass
class FakeSegment:
    clip_id: str = "clip_001"
    text: str = "Hello beautiful world."
    audio_bytes: bytes = b"wav-bytes"
    sample_rate: int = 24000
    duration: float = 1.0


class FakeSampler:
    def __init__(self, segment=None):
        self._segment = segment or FakeSegment()
        self.calls = 0

    def __next__(self):
        self.calls += 1
        return self._segment


class FakeLLMClient:
    def __init__(self, responses=None, exc: Exception | None = None):
        self.responses = list(responses or [])
        self.exc = exc
        self.calls = 0
        self.kwargs = []
        self.prompts = []

    def generate(self, prompt, **kwargs):
        self.calls += 1
        self.prompts.append(prompt)
        self.kwargs.append(kwargs)
        if self.exc is not None:
            raise self.exc
        if self.responses:
            return self.responses.pop(0)
        raise AssertionError("No fake LLM response left")


class FakeDataSaver:
    def __init__(
        self,
        *,
        health_exc: Exception | None = None,
        save_errors: int = 0,
        always_fail_save: bool = False,
    ):
        self.health_exc = health_exc
        self.save_errors = save_errors
        self.always_fail_save = always_fail_save
        self.health_calls = 0
        self.save_calls = 0
        self.save_kwargs = []
        self.flush_calls = []

    def health(self):
        self.health_calls += 1
        if self.health_exc is not None:
            raise self.health_exc
        return {"status": "healthy"}

    def save(self, **kwargs):
        self.save_calls += 1
        self.save_kwargs.append(kwargs)
        if self.always_fail_save or self.save_calls <= self.save_errors:
            raise RuntimeError("data saver down")
        return {"rows_buffered": 2, "buffer_size": 2}

    def flush(self, **kwargs):
        self.flush_calls.append(kwargs)
        return {"status": "success"}


@pytest.fixture(autouse=True)
def stable_parameter_values(monkeypatch):
    pitch_pool = [2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 1, -1, 7, -7]
    rate_pool = [0.8, 0.7, 1.2, 1.25, 0.9, 1.1, 0.6, 1.5, 0.75, 1.33]

    def take(pool, n):
        return [pool[i % len(pool)] for i in range(n)]

    monkeypatch.setattr(
        input_generator,
        "sample_pitch_values",
        lambda n, policy=None: take(pitch_pool, n),
    )
    monkeypatch.setattr(
        input_generator,
        "sample_rate_values",
        lambda n, policy=None: take(rate_pool, n),
    )


def _reverse_st_template(tagged_text: str) -> str:
    inverse = invert_structural_instruction(tagged_text)
    return re.sub(r"<rate, factor=[^>]+>", "<rate, factor=[[RATE_VALUE_PH]]>", inverse)


def _reverse_precise_template(tagged_text: str) -> str:
    count = sum(1 for segment in parse_tags(tagged_text) if segment.factor is not None)
    if count:
        return "Restore rate to " + " and ".join(
            ["[[RATE_VALUE_PH]]x"] * count
        )
    return "Reverse the marked pitch edit."


def _pair_instructions(wave_id: str, tagged_text: str) -> dict[str, dict]:
    return {
        pair_key(ORIGINAL_WAVE_ID, wave_id): {
            "nl": {
                "precise": "Edit the marked span.",
                "vague": "Change the delivery.",
            },
        },
        pair_key(wave_id, ORIGINAL_WAVE_ID): {
            "nl": {
                "precise": _reverse_precise_template(tagged_text),
                "vague": "Restore the original delivery.",
            },
        },
    }


def _llm_multi_response(normed_text: str, tagged_by_wave: dict[str, str]) -> str:
    return json.dumps(
        {
            "normed_text": normed_text,
            "waves": {
                wave_id: {
                    "tagged_text": tagged_text,
                    "pair_instructions": _pair_instructions(wave_id, tagged_text),
                }
                for wave_id, tagged_text in tagged_by_wave.items()
            },
        }
    )


def _llm_response(normed_text: str, tagged_text: str) -> str:
    return _llm_multi_response(normed_text, {"1": tagged_text})


def _reverse_llm_response(wave_id: str, precise: str, vague: str = "Restore delivery.") -> str:
    return json.dumps(
        {
            "reverse_instructions": {
                wave_id: {
                    "nl": {
                        "precise": precise,
                        "vague": vague,
                    }
                }
            }
        }
    )


def test_parse_tags_rejects_nested_and_crossing_edits():
    valid = (
        "<pitch, semitones=2>Hello</pitch> beautiful "
        "<rate, factor=0.8>world</rate>"
    )
    segments = parse_tags(valid)
    assert summarize_tagged_text(valid) == {
        "num_regions": 2,
        "num_pitch_regions": 1,
        "num_rate_regions": 1,
    }
    assert [seg.tag_type for seg in segments if seg.is_tagged] == ["pitch", "rate"]

    nested = (
        "<pitch, semitones=2><rate, factor=0.8>Hello</rate></pitch> beautiful world"
    )
    crossing = (
        "<pitch, semitones=2>Hello <rate, factor=0.8>beautiful</pitch> world</rate>"
    )
    assert validate_tagged_text(nested, "Hello beautiful world")[0] is False
    assert validate_tagged_text(crossing, "Hello beautiful world")[0] is False


def test_chinese_tag_boundaries_can_split_adjacent_han_text():
    group = build_wave_group(
        "你好世界。",
        {
            "1": "你好<rate, factor=0.8>世界</rate>。",
        },
        metadata={"num_waves": 1},
    )

    assert group.tagged_texts["1"] == "你好<rate, factor=0.8>世界</rate>。"


def test_build_wave_group_creates_only_original_bidirectional_pairs():
    group = build_wave_group(
        "Hello beautiful world",
        {
            "1": "<pitch, semitones=2>Hello</pitch> beautiful world",
            "2": "Hello <rate, factor=0.8>beautiful</rate> world",
            "3": "Hello beautiful <pitch, semitones=-3>world</pitch>",
        },
        metadata={"num_waves": 3},
    )

    assert [(p.source_id, p.target_id) for p in group.edit_pairs] == [
        (ORIGINAL_WAVE_ID, "1"),
        ("1", ORIGINAL_WAVE_ID),
        (ORIGINAL_WAVE_ID, "2"),
        ("2", ORIGINAL_WAVE_ID),
        (ORIGINAL_WAVE_ID, "3"),
        ("3", ORIGINAL_WAVE_ID),
    ]
    assert all("source" not in {p.source_id, p.target_id} for p in group.edit_pairs)


def test_ordered_pair_instructions_are_independent_and_structural():
    group = build_wave_group(
        "Hello beautiful world",
        {"1": "Hello <pitch, semitones=2>beautiful</pitch> world"},
        metadata={
            "pair_instructions": {
                "1": _pair_instructions(
                    "1",
                    "Hello <pitch, semitones=2>beautiful</pitch> world",
                )
            }
        },
    )

    forward, reverse = group.edit_pairs

    assert forward.instructions["nl"]["precise"] == "Edit the marked span."
    assert forward.instructions["st"] == (
        "Hello <pitch, semitones=2>beautiful</pitch> world"
    )
    assert reverse.instructions["nl"]["precise"] == "Reverse the marked pitch edit."
    assert reverse.instructions["st"] == (
        "Hello <pitch, semitones=-2>beautiful</pitch> world"
    )
    assert forward.instructions != reverse.instructions
    assert "instruction" + "_record" not in forward.to_metadata_dict()


def test_llm_structural_instructions_are_ignored_and_canonicalized():
    raw = json.loads(
        _llm_response(
            "Hello beautiful world.",
            "Hello <pitch, semitones=2>beautiful</pitch> world.",
        )
    )
    raw["waves"]["1"]["pair_instructions"]["original_to_1"]["st"] = "bad forward st"
    raw["waves"]["1"]["pair_instructions"]["1_to_original"]["st"] = "bad reverse st"

    result = generate_input(
        FakeSampler(),
        FakeLLMClient([json.dumps(raw)]),
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
    )

    assert result is not None
    forward, reverse = result.wave_group.edit_pairs
    assert forward.instructions["st"] == (
        "Hello <pitch, semitones=2>beautiful</pitch> world."
    )
    assert reverse.instructions["st"] == (
        "Hello <pitch, semitones=-2>beautiful</pitch> world."
    )


def test_reverse_rate_instructions_fill_placeholders_with_inverse_values():
    result = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world.",
                    "Hello <rate, factor=0.8>beautiful</rate> world.",
                )
            ]
        ),
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
    )

    assert result is not None
    reverse = result.wave_group.edit_pairs[1]
    assert reverse.source_id == "1"
    assert reverse.target_id == ORIGINAL_WAVE_ID
    assert reverse.instructions["nl"]["precise"] == (
        "Speed up 'beautiful' by 1.25x relative to the source rate"
    )
    assert reverse.instructions["st"] == (
        "Hello <rate, factor=1.25>beautiful</rate> world."
    )
    assert "[[RATE_VALUE_PH]]" not in json.dumps(reverse.instructions)


def test_reverse_mixed_instructions_invert_pitch_and_rate():
    result = generate_input(
        FakeSampler(FakeSegment(text="Hello beautiful world today again.")),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world today again.",
                    (
                        "<pitch, semitones=2>Hello</pitch> "
                        "<rate, factor=0.8>beautiful</rate> world today again."
                    ),
                )
            ]
        ),
        edit_kind="mixed",
        span_level="single_word",
        num_waves=1,
    )

    assert result is not None
    reverse = result.wave_group.edit_pairs[1]
    assert reverse.instructions["nl"]["precise"] == (
        "Lower the pitch of 'Hello' by 2 semitones, and speed up 'beautiful' "
        "by 1.25x relative to the source rate"
    )
    assert reverse.instructions["st"] == (
        "<pitch, semitones=-2>Hello</pitch> "
        "<rate, factor=1.25>beautiful</rate> world today again."
    )


def test_pair_instruction_placeholder_validation_retries():
    forward = _llm_response(
            "Hello beautiful world.",
            "Hello <rate, factor=0.8>beautiful</rate> world.",
    )
    bad_reverse = _reverse_llm_response(
        "1",
        "Restore rate to [[RATE_VALUE_PH]]x and [[RATE_VALUE_PH]]x",
    )
    good_forward = forward
    good_reverse = _reverse_llm_response("1", "Restore rate to [[RATE_VALUE_PH]]x")
    llm_client = FakeLLMClient([forward, bad_reverse, good_forward, good_reverse])

    result = generate_input(
        FakeSampler(),
        llm_client,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
        reverse_nl_mode="llm",
    )

    assert result is not None
    assert llm_client.calls == 4


def test_forward_llm_rejects_placeholders_and_reverse_llm_rejects_vague_placeholders():
    forward_bad = json.loads(_llm_response(
        "Hello beautiful world.",
        "Hello <rate, factor=0.8>beautiful</rate> world.",
    ))
    forward_bad["waves"]["1"]["forward_nl"] = {
        "precise": "Use [[RATE_VALUE_PH]] for the forward edit.",
        "vague": "Change delivery.",
    }
    reverse_vague_bad = _reverse_llm_response(
        "1",
        "Restore rate to [[RATE_VALUE_PH]]x",
        vague="Restore [[RATE_VALUE_PH]]",
    )
    good_forward = json.loads(_llm_response(
        "Hello beautiful world.",
        "Hello <rate, factor=0.8>beautiful</rate> world.",
    ))
    good_forward["waves"]["1"]["forward_nl"] = {
        "precise": "Slow beautiful to 0.8x.",
        "vague": "Slow beautiful down.",
    }
    good_reverse = _reverse_llm_response("1", "Restore rate to [[RATE_VALUE_PH]]x")
    llm_client = FakeLLMClient(
        [
            json.dumps(forward_bad),
            json.dumps(good_forward),
            reverse_vague_bad,
            json.dumps(good_forward),
            good_reverse,
        ]
    )

    result = generate_input(
        FakeSampler(),
        llm_client,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
        forward_nl_mode="llm",
        reverse_nl_mode="llm",
    )

    assert result is not None
    assert llm_client.calls == 5


def test_generate_input_retries_only_llm_validation_failures():
    bad = _llm_response(
        "Hello beautiful world.",
        "<pitch, semitones=2>Hello</pitch> changed world.",
    )
    good = _llm_response(
        "Hello beautiful world.",
        "<pitch, semitones=2>Hello</pitch> beautiful world.",
    )
    llm_client = FakeLLMClient([bad, bad, good])

    result = generate_input(
        FakeSampler(),
        llm_client,
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
    )

    assert result is not None
    assert llm_client.calls == PARAMETERS["llm_max_retries"]
    assert result.metadata["wave_specs"]["1"]["edit_kind"] == "pitch"
    assert result.metadata["wave_specs"]["1"]["span_granularity"] == "single_word"
    assert result.metadata["wave_specs"]["1"]["num_regions"] == 1


def test_short_rate_risk_uses_cjk_two_character_threshold():
    assert short_rate_risk("我想<rate, factor=0.8>东西</rate>").passed
    failure = short_rate_risk("我想<rate, factor=0.8>你</rate>")

    assert not failure.passed
    assert failure.risk_type == "short_rate_risk"
    assert failure.metadata["policy"] == "cjk"
    assert failure.metadata["char_count"] == 1


def test_short_rate_risk_applies_cjk_policy_to_latin_spans_in_chinese_text():
    assert short_rate_risk("这里有<rate, factor=0.8>AI</rate>工具").passed
    failure = short_rate_risk("这里有<rate, factor=0.8>A</rate>工具")

    assert not failure.passed
    assert failure.metadata["policy"] == "cjk"


def test_short_rate_risk_allows_single_english_word_but_keeps_stopword_filter():
    assert short_rate_risk("Say <rate, factor=0.8>hello</rate> now.").passed
    stopword_failure = tiny_stopword_rate_risk(
        "Say <rate, factor=0.8>the</rate> word."
    )
    assert not stopword_failure.passed
    assert stopword_failure.risk_type == "tiny_stopword_rate_risk"

    failures = run_all_filters("Say <rate, factor=0.8>the</rate> word.")
    assert [failure.risk_type for failure in failures] == ["tiny_stopword_rate_risk"]


def test_generate_input_from_sampled_data_uses_stage0_sample_and_timeout():
    sampled = Stage0Sample(
        clip_id="fixed_clip",
        transcript="Hello beautiful world.",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=1.25,
        metadata={"source": "unit"},
    )
    llm = FakeLLMClient(
        [
            _llm_response(
                "Hello beautiful world.",
                "Hello <rate, factor=0.8>beautiful</rate> world.",
            )
        ]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
        stage1_seed=123,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend="gemini", temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is not None
    assert result.clip_id == "fixed_clip"
    assert result.audio_bytes == b"fixed-wav"
    assert result.metadata["stage0_info"]["source"] == "unit"
    assert llm.kwargs[0]["timeout"] == LLM_SERVICE_TIMEOUT_SEC
    assert llm.kwargs[0]["backend"] == "gemini"


def test_generate_input_from_sampled_data_omits_llm_backend_when_none():
    sampled = Stage0Sample(
        clip_id="fixed_clip",
        transcript="Hello beautiful world.",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=1.25,
        metadata={"source": "unit"},
    )
    llm = FakeLLMClient(
        [
            _llm_response(
                "Hello beautiful world.",
                "Hello <rate, factor=0.8>beautiful</rate> world.",
            )
        ]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
        llm_backend=None,
        llm_model=None,
        stage1_seed=123,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is not None
    assert "backend" not in llm.kwargs[0]
    assert "model" not in llm.kwargs[0]
    assert result.metadata["llm_backend"] == "default"


def test_generate_input_from_sampled_data_passes_language_to_prompt():
    sampled = Stage0Sample(
        clip_id="zh_clip",
        transcript="你好世界。",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=1.25,
        metadata={"language": "zh"},
    )
    llm = FakeLLMClient(
        [
            _llm_response(
                "你好世界。",
                "你好<rate, factor=0.8>世界</rate>。",
            )
        ]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
        stage1_seed=123,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is not None
    assert result.metadata["language"] == "zh"
    assert "Mandarin Chinese" in llm.prompts[0]


def test_short_text_policy_caps_regions_and_rejects_mixed_for_tiny_text():
    assert input_generator.stage1_text_policy("你好世界", "zh")[
        "effective_max_regions"
    ] == 1
    assert input_generator.stage1_text_policy("你好世界今天天气", "zh")[
        "effective_max_regions"
    ] == 2
    assert input_generator.stage1_text_policy("hello beautiful world", "en")[
        "effective_max_regions"
    ] == 1
    assert input_generator.stage1_text_policy("hello beautiful world today again", "en")[
        "effective_max_regions"
    ] == 2

    diagnostics = {}
    sampled = Stage0Sample(
        clip_id="tiny_zh",
        transcript="你好世界",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=3.0,
        metadata={"language": "zh"},
    )
    result = generate_input_from_sampled_data(
        sampled,
        FakeLLMClient([]),
        edit_kind="mixed",
        span_level="single_word",
        num_waves=1,
        diagnostics=diagnostics,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is None
    assert diagnostics["effective_max_regions"] == 1
    assert diagnostics["error_type"] == "invalid_edit_controls"


def test_stage1_diagnostics_record_missing_wave_ids():
    diagnostics = {}
    sampled = Stage0Sample(
        clip_id="fixed_clip",
        transcript="Hello beautiful world today again.",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=3.0,
        metadata={"language": "en"},
    )
    result = generate_input_from_sampled_data(
        sampled,
        FakeLLMClient(
            [
                _llm_multi_response(
                    "Hello beautiful world today again.",
                    {"1": "<pitch, semitones=2>Hello</pitch> beautiful world today again."},
                )
                for _ in range(PARAMETERS["llm_max_retries"])
            ]
        ),
        edit_kind="pitch",
        span_level="single_word",
        num_waves=3,
        diagnostics=diagnostics,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is None
    assert diagnostics["missing_wave_ids"] == ["2", "3"]
    assert diagnostics["parsed_wave_ids"] == ["1"]
    assert diagnostics["raw_response_preview"]
    assert diagnostics["validation_attempts"][0]["error_message"] == (
        "LLM response missing wave '2'"
    )


def test_generate_input_from_sampled_data_can_raise_llm_service_error():
    sampled = Stage0Sample(
        clip_id="fixed_clip",
        transcript="Hello beautiful world.",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=1.25,
    )

    with pytest.raises(LLMCallError):
        generate_input_from_sampled_data(
            sampled,
            FakeLLMClient(exc=RuntimeError("read timeout")),
            edit_kind="rate",
            span_level="single_word",
            num_waves=1,
            raise_on_llm_service_error=True,
         parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])


def test_generate_input_returns_none_on_llm_service_failure_without_validation_retry():
    llm_client = FakeLLMClient(exc=RuntimeError("Gemini retry exhausted"))

    result = generate_input(
        FakeSampler(),
        llm_client,
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
    )

    assert result is None
    assert llm_client.calls == 1


def test_generate_input_samples_span_granularity_per_wave_without_shared_span():
    result = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_multi_response(
                    "Hello beautiful world.",
                    {
                        "1": "<pitch, semitones=2>Hello</pitch> beautiful world.",
                        "2": "Hello <pitch, semitones=-2>beautiful</pitch> world.",
                        "3": "Hello beautiful <pitch, semitones=3>world.</pitch>",
                    },
                )
            ]
        ),
        edit_kind="pitch",
        span_level_weights={"single_word": 1.0},
        num_waves=3,
    )

    assert result is not None
    assert [spec["span_granularity"] for spec in result.metadata["wave_specs"].values()] == [
        "single_word",
        "single_word",
        "single_word",
    ]
    assert len(set(result.wave_group.tagged_texts.values())) == 3


def test_generate_input_accepts_sentence_span_for_pitch_kind():
    result = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world.",
                    "<pitch, semitones=2>Hello beautiful world.</pitch>",
                )
            ]
        ),
        edit_kind="pitch",
        span_level="sentence",
        num_waves=1,
    )

    assert result is not None
    assert result.metadata["wave_specs"]["1"]["edit_kind"] == "pitch"
    assert result.metadata["wave_specs"]["1"]["span_granularity"] == "sentence"


def test_generate_input_saves_parameter_values_and_policy():
    result = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_multi_response(
                    "Hello beautiful world.",
                    {
                        "1": "<pitch, semitones=2>Hello</pitch> beautiful world.",
                        "2": "Hello <pitch, semitones=-2>beautiful</pitch> world.",
                        "3": "Hello beautiful <pitch, semitones=3>world.</pitch>",
                    },
                )
            ]
        ),
        edit_kind="pitch",
        span_level="single_word",
        num_waves=3,
    )

    assert result is not None
    assert len(result.metadata["parameter_values"]["pitch_values"]) == 9
    assert len(result.metadata["parameter_values"]["rate_values"]) == 9
    assert result.metadata["parameter_policy"]["max_edit_spans_per_wave"] == 3


def test_parameter_policy_validation_requires_complete_configuration():
    default_policy = normalize_parameter_policy( policy=PARAMETERS["parameter_policy"])
    assert default_policy["values_per_wave"] == 3
    assert default_policy["pitch"]["buckets"]["normal_down"]["values"] == [
        -6,
        -5,
        -4,
        -3,
        -2,
    ]
    assert default_policy["rate"]["decimals"] == 2

    incomplete = {"values_per_wave": 2, "pitch": {"buckets": {"normal_up": {"weight": 0.9}}}, "rate": {"decimals": 3}}
    with pytest.raises(ValueError, match="values must be a non-empty list"):
        normalize_parameter_policy(incomplete)
    with pytest.raises(ValueError, match="values_per_wave must be configured"):
        normalize_parameter_policy({})


def policy_with(overrides):
    policy = copy.deepcopy(PARAMETERS["parameter_policy"])
    for axis, value in overrides.items():
        if axis == "rate":
            value = {"decimals": policy["rate"]["decimals"], **value}
        policy[axis] = value
    return policy


def test_parameter_policy_rejects_invalid_controls():
    with pytest.raises(ValueError, match="must not contain 0"):
        normalize_parameter_policy(
            policy_with({"pitch": {"buckets": {"bad": {"weight": 1.0, "values": [0]}}}})
        )
    with pytest.raises(ValueError, match="must contain integers"):
        normalize_parameter_policy(
            policy_with({"pitch": {"buckets": {"bad": {"weight": 1.0, "values": [1.5]}}}})
        )
    with pytest.raises(ValueError, match="must not cross 1.0"):
        normalize_parameter_policy(
            policy_with({"rate": {"buckets": {"bad": {"weight": 1.0, "ranges": [[0.9, 1.1]]}}}})
        )
    with pytest.raises(ValueError, match="at least one positive-weight bucket"):
        normalize_parameter_policy(
            policy_with({
                "pitch": {
                    "buckets": {
                        "normal_down": {"weight": 0},
                        "normal_up": {"weight": 0},
                        "near_neutral": {"weight": 0},
                        "wide_down": {"weight": 0},
                        "wide_up": {"weight": 0},
                    }
                }
            })
        )
    with pytest.raises(ValueError, match="must not be empty"):
        normalize_parameter_policy(policy_with({"rate": {"buckets": {}}}))


def test_parameter_policy_sampling_uses_configured_buckets(monkeypatch):
    monkeypatch.undo()
    fixed_policy = normalize_parameter_policy(
        {
            "values_per_wave": 2,
            "pitch": {
                "buckets": {
                    "normal_down": {"weight": 0},
                    "normal_up": {"weight": 0},
                    "near_neutral": {"weight": 0},
                    "wide_down": {"weight": 0},
                    "wide_up": {"weight": 0},
                    "fixed": {"weight": 1.0, "values": [4]},
                }
            },
            "rate": {
                "decimals": 3,
                "buckets": {
                    "normal_slow": {"weight": 0},
                    "normal_fast": {"weight": 0},
                    "wide_slow": {"weight": 0},
                    "wide_fast": {"weight": 0},
                    "fixed": {"weight": 1.0, "ranges": [[1.234, 1.236]]},
                },
            },
        }
    )

    assert input_generator.sample_pitch_values(5, fixed_policy) == [4, 4, 4, 4, 4]
    rate_values = input_generator.sample_rate_values(5, fixed_policy)
    assert all(1.234 <= value <= 1.236 for value in rate_values)
    assert all(len(f"{value:.3f}".split(".")[1]) == 3 for value in rate_values)


def test_generate_input_from_sampled_data_uses_parameter_policy(monkeypatch):
    monkeypatch.undo()
    sampled = Stage0Sample(
        clip_id="fixed_clip",
        transcript="Hello beautiful world.",
        audio_bytes=b"fixed-wav",
        sample_rate=24000,
        duration=1.25,
    )
    policy = {
        "values_per_wave": 2,
        "pitch": {
            "buckets": {
                "normal_down": {"weight": 0},
                "normal_up": {"weight": 0},
                "near_neutral": {"weight": 0},
                "wide_down": {"weight": 0},
                "wide_up": {"weight": 0},
                "fixed": {"weight": 1.0, "values": [4]},
            }
        },
        "rate": {
            "decimals": 2,
            "buckets": {
                "normal_slow": {"weight": 0},
                "normal_fast": {"weight": 1.0, "ranges": [[1.2, 1.21]]},
                "wide_slow": {"weight": 0},
                "wide_fast": {"weight": 0},
            }
        },
    }
    llm = FakeLLMClient(
        [
            _llm_response(
                "Hello beautiful world.",
                "Hello <pitch, semitones=4>beautiful</pitch> world.",
            )
        ]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
        stage1_seed=123,
        parameter_policy=policy,
     edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is not None
    assert result.metadata["parameter_values"]["pitch_values"] == [4, 4]
    assert len(result.metadata["parameter_values"]["rate_values"]) == 2
    assert "pitch_values: [4, 4]" in llm.prompts[0]
    assert result.metadata["parameter_policy"]["values_per_wave"] == 2


def test_region_count_weights_enforce_exact_regions(monkeypatch):
    monkeypatch.undo()
    sampled = Stage0Sample(
        clip_id="clip",
        transcript="Hello beautiful world today again.",
        audio_bytes=b"wav",
        sample_rate=24000,
        duration=2.0,
    )
    policy = {
        "rate": PARAMETERS["parameter_policy"]["rate"],
        "values_per_wave": 2,
        "pitch": {
            "buckets": {
                "normal_down": {"weight": 0},
                "normal_up": {"weight": 0},
                "near_neutral": {"weight": 0},
                "wide_down": {"weight": 0},
                "wide_up": {"weight": 0},
                "fixed": {"weight": 1.0, "values": [4]},
            }
        },
    }
    llm = FakeLLMClient(
        [
            _llm_response(
                "Hello beautiful world today again.",
                "Hello <pitch, semitones=4>beautiful</pitch> world "
                "<pitch, semitones=4>today</pitch> again.",
            )
        ]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
        stage1_seed=123,
        parameter_policy=policy,
        region_count_weights={1: 0.0, 2: 1.0, 3: 0.0},
     edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is not None
    wave_spec = result.metadata["wave_specs"]["1"]
    assert wave_spec["target_num_regions"] == 2
    assert wave_spec["num_regions"] == 2
    assert result.metadata["region_count_weights"] == {1: 0.0, 2: 1.0, 3: 0.0}
    assert "Select exactly 2 non-overlapping spans" in llm.prompts[0]
    assert "Exact tagged edit spans: 2." in llm.prompts[0]


def test_region_count_weights_reject_wrong_exact_count(monkeypatch):
    monkeypatch.undo()
    sampled = Stage0Sample(
        clip_id="clip",
        transcript="Hello beautiful world today again.",
        audio_bytes=b"wav",
        sample_rate=24000,
        duration=2.0,
    )
    llm = FakeLLMClient(
        [
            _llm_response(
                "Hello beautiful world today again.",
                "Hello <pitch, semitones=4>beautiful</pitch> world today again.",
            )
        ]
        * PARAMETERS["llm_max_retries"]
    )

    result = generate_input_from_sampled_data(
        sampled,
        llm,
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
        stage1_seed=123,
        parameter_policy={
            "pitch": {
                "buckets": {
                    "normal_down": {"weight": 0},
                    "normal_up": {"weight": 0},
                    "near_neutral": {"weight": 0},
                    "wide_down": {"weight": 0},
                    "wide_up": {"weight": 0},
                    "fixed": {"weight": 1.0, "values": [4]},
                }
            }
        },
        region_count_weights={1: 0.0, 2: 1.0, 3: 0.0},
     edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert result is None


def test_region_effect_metrics_compute_rate_and_reverse_pitch_direction():
    class FakeF0:
        def analyze(self, audio_bytes, *, spans, **_kwargs):
            median = 200.0 if audio_bytes == b"generated" else 100.0
            return {
                "spans": [
                    {
                        "label": span["label"],
                        "median_f0_hz": median,
                        "voiced_fraction": 1.0,
                        "num_frames": 10,
                        "num_voiced_frames": 10,
                    }
                    for span in spans
                ]
            }

    forward = compute_region_effect_metrics(
        source_audio_bytes=b"original",
        target_audio_bytes=b"generated",
        tagged_text="<rate, factor=2.0>Hello</rate> <pitch, semitones=12>world</pitch>.",
        source_spans=[{"start": 0.0, "end": 2.0}, {"start": 2.0, "end": 3.0}],
        target_spans=[{"start": 0.0, "end": 1.0}, {"start": 1.0, "end": 2.0}],
        f0_client=FakeF0(),
    )
    assert forward["duration_l1_mean_sec"] == 0.0
    assert forward["pitch_l1_mean_semitones"] == 0.0
    assert forward["region_effects"][0]["expected_duration_sec"] == 1.0
    assert forward["region_effects"][1]["actual_shift_semitones"] == pytest.approx(12.0)

    reverse = compute_region_effect_metrics(
        source_audio_bytes=b"generated",
        target_audio_bytes=b"original",
        tagged_text="<pitch, semitones=-12>world</pitch>.",
        source_spans=[{"start": 1.0, "end": 2.0}],
        target_spans=[{"start": 2.0, "end": 3.0}],
        f0_client=FakeF0(),
    )
    assert reverse["pitch_l1_mean_semitones"] == 0.0
    assert reverse["region_effects"][0]["actual_shift_semitones"] == pytest.approx(-12.0)

    reverse_rate = compute_region_effect_metrics(
        source_audio_bytes=b"generated",
        target_audio_bytes=b"original",
        tagged_text="<rate, factor=0.5>Hello</rate>.",
        source_spans=[{"start": 0.0, "end": 1.0}],
        target_spans=[{"start": 0.0, "end": 2.0}],
        f0_client=None,
    )
    assert reverse_rate["duration_l1_mean_sec"] == 0.0
    assert reverse_rate["region_effects"][0]["expected_duration_sec"] == 2.0

    zero_rate = compute_region_effect_metrics(
        source_audio_bytes=b"original",
        target_audio_bytes=b"generated",
        tagged_text="<rate, factor=2.0>Hello</rate>.",
        source_spans=[{"start": 0.0, "end": 2.0}],
        target_spans=[{"start": 1.0, "end": 1.0}],
        f0_client=None,
    )
    assert zero_rate["region_effects"][0]["target_duration_sec"] == 0.0
    assert zero_rate["duration_l1_mean_sec"] == 1.0

    class CountingF0(FakeF0):
        def __init__(self):
            self.calls = []

        def analyze(self, audio_bytes, *, spans, **kwargs):
            self.calls.append({"audio_bytes": audio_bytes, "spans": list(spans)})
            return super().analyze(audio_bytes, spans=spans, **kwargs)

    f0 = CountingF0()
    zero_pitch = compute_region_effect_metrics(
        source_audio_bytes=b"original",
        target_audio_bytes=b"generated",
        tagged_text="<pitch, semitones=12>world</pitch>.",
        source_spans=[{"start": 0.0, "end": 1.0}],
        target_spans=[{"start": 1.0, "end": 1.0}],
        f0_client=f0,
    )
    assert len(f0.calls) == 1
    assert f0.calls[0]["audio_bytes"] == b"original"
    assert zero_pitch["region_effects"][0]["target_median_f0_hz"] == 100.0
    assert zero_pitch["pitch_l1_mean_semitones"] == 12.0


def test_stage3_evaluate_adds_region_effects_for_fresh_eval(monkeypatch):
    class FakeF0:
        def analyze(self, audio_bytes, *, spans, **_kwargs):
            median = 200.0 if audio_bytes == b"generated" else 100.0
            return {
                "spans": [
                    {
                        "label": span["label"],
                        "median_f0_hz": median,
                        "voiced_fraction": 1.0,
                        "num_frames": 10,
                        "num_voiced_frames": 10,
                    }
                    for span in spans
                ]
            }

    def fake_evaluate_pair(**_kwargs):
        return PairEvalResult(
            source_metrics=EvalMetrics(asr_text="Hello world.", wer=0.0),
            target_metrics=EvalMetrics(asr_text="Hello world.", wer=0.0, spk_similarity=0.95),
        )

    monkeypatch.setattr(pipeline_mod, "evaluate_pair", fake_evaluate_pair)
    pair = WaveEditPair(
        source_id=ORIGINAL_WAVE_ID,
        target_id="1",
        plain_text="Hello world.",
        source_tagged_text="Hello world.",
        target_tagged_text="<pitch, semitones=12>Hello</pitch> world.",
        instructions={"st": "<pitch, semitones=12>Hello</pitch> world."},
    )
    pipeline_input = PipelineInput(
        clip_id="clip",
        transcript="Hello world.",
        normed_text="Hello world.",
        audio_bytes=b"original",
        sample_rate=24000,
        duration=1.0,
        wave_group=WaveGroup(
            plain_text="Hello world.",
            tagged_texts={ORIGINAL_WAVE_ID: "Hello world.", "1": pair.target_tagged_text},
            edit_pairs=[pair],
            boundaries=[],
            metadata={},
        ),
        metadata={"language": "en"},
    )
    stage2_state = {
        "pipeline_input": pipeline_input,
        "sample_id": "sample_001",
        "t_start": 0.0,
        "wave_results": {
            "1": {
                "result": {
                    "alignment": {
                        "edit_regions": [
                            {"start_sec": 0.0, "end_sec": 1.0, "text": "Hello"}
                        ]
                    },
                    "edits_applied": [{"target_span": [0.0, 1.0], "words": "Hello"}],
                }
            }
        },
        "wave_audio_bytes": {"1": b"generated"},
        "pair_inputs": [(pair, b"original", b"generated")],
    }

    result = pipeline_mod.evaluate_sample(
        stage2_state,
        asr_client=None,
        wer_client=None,
        utmos_client=None,
        spk_sim_client=None,
        f0_client=FakeF0(),
        do_evaluate=True,
    )

    eval_results = result["pair_results"][0]["eval_results"]
    assert eval_results["region_effects_status"] == "ok"
    assert eval_results["pitch_l1_mean_semitones"] == 0.0
    assert eval_results["region_effects"][0]["actual_shift_semitones"] == pytest.approx(12.0)


def test_prosody_stage_adapter_passes_parameter_policy(monkeypatch):
    class DummyPipelineInput:
        metadata = {}

    captured = {}

    class DummyLLMClient:
        def __init__(self, url):
            self.url = url

    def fake_generate_input_from_sampled_data(*args, **kwargs):
        captured.update(kwargs)
        return DummyPipelineInput()

    monkeypatch.setattr("speech_edit.clients.services.LLMClient", DummyLLMClient)
    monkeypatch.setattr(
        input_generator,
        "generate_input_from_sampled_data",
        fake_generate_input_from_sampled_data,
    )
    policy = {"values_per_wave": 2}
    region_count_weights = {1: 0.0, 2: 1.0, 3: 0.0}
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/prosody_edit/defaults"),
                "seed": 42,
                "num_waves": 1,
                "parameter_policy": policy,
                "region_count_weights": region_count_weights,
            }
        },
        worker_id="worker-a",
        stage=StageSpec("plan", "in", "out", DEFAULT_STAGE_WORKER_PYTHON_BIN),
        services=direct_services(llm="http://llm"),
    )

    result = ProsodyStageAdapter().process(
        "plan",
        {
            "sample": Stage0Sample(
                clip_id="clip",
                transcript="Hello world.",
                audio_bytes=b"wav",
                sample_rate=24000,
                duration=1.0,
            ),
            "sample_id": "sample_1",
            "attempt_index": 7,
        },
        ctx,
    )

    assert captured["parameter_policy"] == policy
    assert captured["region_count_weights"] == region_count_weights
    assert result.outputs[0].payload.metadata["input_id"] == "sample_1"


def test_generate_input_rejects_values_outside_parameter_pool():
    bad = _llm_response(
        "Hello beautiful world.",
        "<pitch, semitones=9>Hello</pitch> beautiful world.",
    )
    good = _llm_response(
        "Hello beautiful world.",
        "<pitch, semitones=2>Hello</pitch> beautiful world.",
    )
    llm_client = FakeLLMClient([bad, good])

    result = generate_input(
        FakeSampler(),
        llm_client,
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
    )

    assert result is not None
    assert llm_client.calls == 2


def test_pitch_tags_reject_fractional_semitones():
    valid, message = validate_tagged_text(
        "<pitch, semitones=2.5>Hello</pitch> beautiful world",
        "Hello beautiful world",
    )

    assert valid is False
    assert "integer" in message


def test_sample_output_layout_uses_audios_and_bidirectional_pair_symlinks(tmp_path: Path):
    pipeline_input = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world.",
                    "<pitch, semitones=2>Hello</pitch> beautiful world.",
                )
            ]
        ),
        edit_kind="pitch",
        span_level="single_word",
        num_waves=1,
    )
    assert pipeline_input is not None
    pair_results = []
    for pair in pipeline_input.wave_group.edit_pairs:
        pair_metadata = pair.to_metadata_dict()
        pair_results.append(
            {
                "pair_dir_name": f"{pair.source_id}_to_{pair.target_id}",
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "source_audio_bytes": b"src",
                "target_audio_bytes": b"tgt",
                "instructions": pair_metadata["instructions"],
                "source_tagged_text": pair.source_tagged_text,
                "target_tagged_text": pair.target_tagged_text,
                "eval_results": {},
            }
        )
    wave_results = {
        "1": {
            "wave_id": "1",
            "tagged_text": pipeline_input.wave_group.tagged_texts["1"],
            "plain_text": pipeline_input.wave_group.plain_text,
            "edited_duration": 1.0,
            "result": {},
        }
    }
    metadata = build_sample_metadata(
        "sample_001",
        pipeline_input,
        wave_results,
        pair_results,
        pipeline_version="2.0.0",
        pipeline_type="prosody_edit",
    )

    sample_dir = tmp_path / "sample_001"
    save_sample_outputs(
        str(sample_dir),
        original_audio_bytes=b"original",
        wave_audio_bytes={"1": b"generated"},
        pair_results=pair_results,
        metadata=metadata,
    )

    assert (sample_dir / "audios" / "original.wav").read_bytes() == b"original"
    assert (sample_dir / "audios" / "generated_1.wav").read_bytes() == b"generated"
    assert (sample_dir / "edit_pairs" / "original_to_1" / "source.wav").is_symlink()
    assert (sample_dir / "edit_pairs" / "1_to_original" / "target.wav").is_symlink()
    saved_metadata = json.loads((sample_dir / "metadata.json").read_text())
    assert saved_metadata == metadata
    assert metadata["sample_dir"] == str(sample_dir)
    assert saved_metadata["wave_specs"]["1"]["edit_kind"] == "pitch"
    assert saved_metadata["parameter_values"]["pitch_values"]
    assert "instructions" in saved_metadata["edit_pairs"][0]
    assert "tag_diff" not in saved_metadata["edit_pairs"][0]
    assert "instruction" + "_record" not in saved_metadata["edit_pairs"][0]
    assert "st" + "_instruction" not in saved_metadata["edit_pairs"][0]
    assert "eval_summary" not in saved_metadata


def test_stage2_uses_chinese_aligner_language(monkeypatch):
    captured = {}

    def fake_apply_edits(
        audio_bytes,
        tagged_text,
        aligner_client,
        sample_rate,
        **kwargs,
    ):
        captured["language"] = kwargs["language"]
        return EditResult(
            edited_audio=np.zeros(240, dtype=np.float32),
            sample_rate=sample_rate,
            edits_applied=[],
            edit_time_sec=0.01,
        )

    monkeypatch.setattr(pipeline_mod, "apply_edits_from_service", fake_apply_edits)
    sampled = Stage0Sample(
        clip_id="zh_clip",
        transcript="你好世界。",
        audio_bytes=b"wav",
        sample_rate=24000,
        duration=1.0,
        metadata={"language": "zh"},
    )
    pipeline_input = generate_input_from_sampled_data(
        sampled,
        FakeLLMClient(
            [_llm_response("你好世界。", "你好<rate, factor=0.8>世界</rate>。")]
        ),
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
     parameter_policy=PARAMETERS["parameter_policy"], edit_kind_weights=PARAMETERS["edit_kind_weights"], span_level_weights=PARAMETERS["span_level_weights"], max_retries=3, llm_backend=None, temperature=PARAMETERS["llm_temperature"], forward_nl_mode=PARAMETERS["forward_nl_mode"], reverse_nl_mode=PARAMETERS["reverse_nl_mode"])

    assert pipeline_input is not None
    pipeline_mod.edit_sample(
        pipeline_input,
        aligner_client=object(),
        aligner_backend="qwen3",
        pitch_method="psola+qwen3_codec",
        rate_method="wsola+indextts2",
        psola_indextts2_cco_pitch_boost="none",
        pitch_client=None,
        rate_client=None,
        f5_client=None,
        crossfade_ms=40,
        smooth_margin_sec=0.1,
        smooth_policy="never",
        t_start=0,
     f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    assert captured["language"] == "zho"


def test_splice_crossfade_preserves_expected_duration():
    original = np.linspace(-1.0, 1.0, 1000, dtype=np.float32)
    replacement = np.zeros(250, dtype=np.float32)

    result, boundaries = _splice_replacements(
        original,
        [(200, 400, replacement)],
        sample_rate=1000,
        crossfade_ms=40,
    )

    assert len(result) == len(original) - 200 + 250
    assert boundaries == [0.2, 0.45]
    np.testing.assert_allclose(result[:200], original[:200])
    np.testing.assert_allclose(result[450:], original[400:])


def test_smooth_splice_boundaries_requires_24k_and_checks_f5_sample_rate():
    class FakeF5Client:
        def smooth_boundaries(self, **kwargs):
            assert kwargs["sample_rate"] == 24000
            return numpy_to_wav(np.zeros(24000, dtype=np.float32), 24000)

    smoothed, sample_rate = smooth_splice_boundaries(
        spliced_audio=np.zeros(24000, dtype=np.float32),
        sr=24000,
        plain_text="开发边界将作为城市发展的刚性约定",
        boundary_times=[0.5],
        f5_client=FakeF5Client(),
        margin_sec=0.1,
     f5_steps=32, source_crossfade_sec=0.0, regen_crossfade_sec=0.01, seed=None)

    assert sample_rate == 24000
    assert len(smoothed) == 24000

    with pytest.raises(ValueError, match="expects 24000 Hz"):
        smooth_splice_boundaries(
            spliced_audio=np.zeros(16000, dtype=np.float32),
            sr=16000,
            plain_text="开发边界将作为城市发展的刚性约定",
            boundary_times=[0.5],
            f5_client=FakeF5Client(),
            margin_sec=0.1,
         f5_steps=32, source_crossfade_sec=0.0, regen_crossfade_sec=0.01, seed=None)

    class BadF5Client:
        def smooth_boundaries(self, **_kwargs):
            return numpy_to_wav(np.zeros(22050, dtype=np.float32), 22050)

    with pytest.raises(ValueError, match="returned sr=22050"):
        smooth_splice_boundaries(
            spliced_audio=np.zeros(24000, dtype=np.float32),
            sr=24000,
            plain_text="开发边界将作为城市发展的刚性约定",
            boundary_times=[0.5],
            f5_client=BadF5Client(),
            margin_sec=0.1,
         f5_steps=32, source_crossfade_sec=0.0, regen_crossfade_sec=0.01, seed=None)


def test_working_sample_rate_resolver_covers_method_policy():
    assert (
        resolve_working_sample_rate(
            "<pitch, semitones=-4>Hello</pitch> world",
            source_sample_rate=16000,
            pitch_method="psola+qwen3_codec",
            rate_method="wsola+indextts2",
        )
        == 24000
    )
    assert (
        resolve_working_sample_rate(
            "<pitch, semitones=-4>Hello</pitch> world",
            source_sample_rate=16000,
            pitch_method="psola+indextts2_cco",
            rate_method="wsola+indextts2",
        )
        == 24000
    )
    assert (
        resolve_working_sample_rate(
            "Hello <rate, factor=0.85>world</rate>",
            source_sample_rate=16000,
            pitch_method="psola+qwen3_codec",
            rate_method="wsola+indextts2",
        )
        == 24000
    )
    assert (
        resolve_working_sample_rate(
            "<pitch, semitones=-4>Hello</pitch> <rate, factor=0.85>world</rate>",
            source_sample_rate=16000,
            pitch_method="psola+qwen3_codec",
            rate_method="wsola+indextts2",
        )
        == 24000
    )
    assert (
        resolve_working_sample_rate(
            "<pitch, semitones=-4>Hello</pitch> <rate, factor=0.85>world</rate>",
            source_sample_rate=16000,
            pitch_method="psola+indextts2_cco",
            rate_method="wsola+indextts2",
        )
        == 24000
    )


def test_psola_indextts2_cco_pitch_boost_resolves_effective_semitones():
    effective, metadata = resolve_psola_indextts2_cco_effective_semitones(
        -4,
        pitch_method="psola+indextts2_cco",
        psola_indextts2_cco_pitch_boost="add:2",
    )
    assert effective == -6
    assert metadata["requested_semitones"] == -4
    assert metadata["effective_semitones"] == -6
    assert metadata["boost_applied"] is True
    assert metadata["boost_clamped"] is False

    effective, metadata = resolve_psola_indextts2_cco_effective_semitones(
        4,
        pitch_method="psola+indextts2_cco",
        psola_indextts2_cco_pitch_boost="mul:1.5",
    )
    assert effective == 6
    assert metadata["boost_applied"] is True

    effective, metadata = resolve_psola_indextts2_cco_effective_semitones(
        -20,
        pitch_method="psola+indextts2_cco",
        psola_indextts2_cco_pitch_boost="add:6",
    )
    assert effective == -24
    assert metadata["boost_clamped"] is True

    effective, metadata = resolve_psola_indextts2_cco_effective_semitones(
        -4,
        pitch_method="psola+qwen3_codec",
        psola_indextts2_cco_pitch_boost="add:6",
    )
    assert effective == -4
    assert metadata["boost_applied"] is False

    with pytest.raises(ValueError):
        resolve_psola_indextts2_cco_effective_semitones(
            4,
            pitch_method="psola+indextts2_cco",
            psola_indextts2_cco_pitch_boost="add:-1",
        )


def test_apply_edits_passes_boosted_indextts2_pitch_to_service():
    class FakeAligner:
        def align(self, _audio_bytes, _text, **_kwargs):
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.0, "end": 1.0}],
            }

    class FakePitchClient:
        def __init__(self):
            self.calls = []

        def shift(self, **kwargs):
            self.calls.append(kwargs)
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            return numpy_to_wav(segment_audio, segment_sr)

    pitch_client = FakePitchClient()
    result = apply_edits_from_service(
        numpy_to_wav(np.zeros(22050, dtype=np.float32), 22050),
        "<pitch, semitones=-4>Hello world</pitch>",
        FakeAligner(),
        sample_rate=22050,
        aligner_backend="qwen3",
        language="eng",
        pitch_method="psola+indextts2_cco",
        psola_indextts2_cco_pitch_boost="add:2",
        pitch_client=pitch_client,
     rate_method=PARAMETERS["rate_method"], crossfade_ms=PARAMETERS["crossfade_ms"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    assert pitch_client.calls[0]["semitones"] == -6
    method = result.edits_applied[0]["method"]
    assert method["requested_semitones"] == -4
    assert method["effective_semitones"] == -6
    assert method["psola_indextts2_cco_pitch_boost"] == "add:2"
    assert result.sample_rate == 24000
    assert result.alignment_metadata["source_sample_rate"] == 22050
    assert result.alignment_metadata["normalized_sample_rate"] == 24000
    assert result.alignment_metadata["source_resampled_to_target"] is True
    assert result.alignment_metadata["psola_indextts2_cco_pitch_boost"] == "add:2"


def test_pitch_edit_preserves_unedited_24k_regions_and_normalizes_length():
    class FakeAligner:
        def align(self, audio_bytes, text, **kwargs):
            audio, sr = wav_to_numpy(audio_bytes)
            assert sr == 24000
            assert len(audio) == 24000
            assert text == "Hello world"
            assert kwargs["queries"] == ["Hello <q>world</q>"]
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.25, "end": 0.75}],
            }

    class FakePitchClient:
        def shift(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            assert len(segment_audio) == 12000
            return numpy_to_wav(np.zeros(6000, dtype=np.float32), 22050)

    source_audio = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    source_wav = numpy_to_wav(source_audio, 24000)
    decoded_source, _ = wav_to_numpy(source_wav)
    result = apply_edits_from_service(
        source_wav,
        "Hello <pitch, semitones=-4>world</pitch>",
        FakeAligner(),
        sample_rate=24000,
        aligner_backend="qwen3",
        language="eng",
        pitch_method="psola+indextts2_cco",
        pitch_client=FakePitchClient(),
        crossfade_ms=0,
        f5_client=None,
     rate_method=PARAMETERS["rate_method"], psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    assert result.sample_rate == 24000
    assert len(result.edited_audio) == len(decoded_source)
    np.testing.assert_array_equal(result.edited_audio[:6000], decoded_source[:6000])
    np.testing.assert_array_equal(result.edited_audio[18000:], decoded_source[18000:])

    method = result.edits_applied[0]["method"]
    assert method["service_output_sample_rate"] == 22050
    assert method["resampled_to_target_sample_rate"] is True
    assert method["length_normalized_to_source_window"] is True
    assert method["source_window_samples"] == 12000


def test_rate_edit_preserves_unedited_24k_regions_while_length_changes():
    class FakeAligner:
        def align(self, audio_bytes, _text, **_kwargs):
            _audio, sr = wav_to_numpy(audio_bytes)
            assert sr == 24000
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.25, "end": 0.75}],
            }

    class FakeRateClient:
        def stretch(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            assert len(segment_audio) == 12000
            return numpy_to_wav(np.zeros(18000, dtype=np.float32), 22050)

    source_audio = np.linspace(-0.5, 0.5, 24000, dtype=np.float32)
    source_wav = numpy_to_wav(source_audio, 24000)
    decoded_source, _ = wav_to_numpy(source_wav)
    result = apply_edits_from_service(
        source_wav,
        "Hello <rate, factor=0.75>world</rate>",
        FakeAligner(),
        sample_rate=24000,
        aligner_backend="qwen3",
        language="eng",
        rate_method="wsola+indextts2",
        rate_client=FakeRateClient(),
        crossfade_ms=0,
        f5_client=None,
     pitch_method=PARAMETERS["pitch_method"], psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    assert result.sample_rate == 24000
    assert len(result.edited_audio) > len(decoded_source)
    np.testing.assert_array_equal(result.edited_audio[:6000], decoded_source[:6000])
    np.testing.assert_array_equal(result.edited_audio[-6000:], decoded_source[18000:])


def test_rate_edit_normalizes_source_to_24k_and_resamples_replacement():
    class FakeAligner:
        def align(self, audio_bytes, text, **kwargs):
            audio, sr = wav_to_numpy(audio_bytes)
            assert sr == 24000
            assert len(audio) == 24000
            assert text == "Hello world"
            assert kwargs["queries"] == ["Hello <q>world</q>"]
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.5, "end": 1.0}],
            }

    class FakeRateClient:
        def stretch(self, **kwargs):
            assert kwargs["method"] == "wsola+indextts2"
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            assert len(segment_audio) == 12000
            return numpy_to_wav(np.zeros(11025, dtype=np.float32), 22050)

    source_audio = np.zeros(16000, dtype=np.float32)
    result = apply_edits_from_service(
        numpy_to_wav(source_audio, 16000),
        "Hello <rate, factor=0.85>world</rate>",
        FakeAligner(),
        sample_rate=16000,
        aligner_backend="qwen3",
        language="eng",
        pitch_method="psola+qwen3_codec",
        rate_method="wsola+indextts2",
        rate_client=FakeRateClient(),
        f5_client=None,
     psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], crossfade_ms=PARAMETERS["crossfade_ms"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    edit = result.edits_applied[0]
    assert result.sample_rate == 24000
    assert result.alignment_metadata["source_sample_rate"] == 16000
    assert result.alignment_metadata["normalized_sample_rate"] == 24000
    assert result.alignment_metadata["source_resampled_to_target"] is True
    assert result.alignment_metadata["working_sample_rate"] == 24000
    assert result.alignment_metadata["final_sample_rate"] == 24000
    assert edit["method"]["service_output_sample_rate"] == 22050
    assert edit["method"]["working_sample_rate"] == 24000
    assert edit["method"]["resampled_to_working_sample_rate"] is True
    assert edit["method"]["resampled_to_target_sample_rate"] is True


def test_mixed_pitch_and_rate_qwen3_policy_resamples_rate_output_to_24k():
    class FakeAligner:
        def align(self, audio_bytes, text, **kwargs):
            _audio, sr = wav_to_numpy(audio_bytes)
            assert sr == 24000
            assert kwargs["queries"] == ["<q>Hello</q> world", "Hello <q>world</q>"]
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [
                    {"start": 0.0, "end": 0.5},
                    {"start": 0.5, "end": 1.0},
                ],
            }

    class FakePitchClient:
        def shift(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            return numpy_to_wav(segment_audio, 24000)

    class FakeRateClient:
        def stretch(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            assert segment_sr == 24000
            assert len(segment_audio) == 12000
            return numpy_to_wav(np.zeros(11025, dtype=np.float32), 22050)

    result = apply_edits_from_service(
        numpy_to_wav(np.zeros(16000, dtype=np.float32), 16000),
        "<pitch, semitones=-4>Hello</pitch> <rate, factor=0.85>world</rate>",
        FakeAligner(),
        sample_rate=16000,
        aligner_backend="qwen3",
        language="eng",
        pitch_method="psola+qwen3_codec",
        rate_method="wsola+indextts2",
        pitch_client=FakePitchClient(),
        rate_client=FakeRateClient(),
        f5_client=None,
     psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], crossfade_ms=PARAMETERS["crossfade_ms"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    rate_edit = next(edit for edit in result.edits_applied if edit["edit_type"] == "rate")
    assert result.sample_rate == 24000
    assert result.alignment_metadata["working_sample_rate"] == 24000
    assert result.alignment_metadata["final_sample_rate"] == 24000
    assert rate_edit["method"]["service_output_sample_rate"] == 22050
    assert rate_edit["method"]["working_sample_rate"] == 24000
    assert rate_edit["method"]["resampled_to_working_sample_rate"] is True


def test_apply_edits_rejects_wrong_f5_sample_rate():
    class FakeAligner:
        def align(self, _audio_bytes, _text, **_kwargs):
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.25, "end": 0.75}],
            }

    class FakePitchClient:
        def shift(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            return numpy_to_wav(segment_audio, segment_sr)

    class FakeF5Client:
        def smooth_boundaries(self, **kwargs):
            assert kwargs["sample_rate"] == 24000
            return numpy_to_wav(np.zeros(22050, dtype=np.float32), 22050)

    with pytest.raises(ValueError, match="expected input sr=24000"):
        result = apply_edits_from_service(
            numpy_to_wav(np.zeros(24000, dtype=np.float32), 24000),
            "Hello <pitch, semitones=-4>world</pitch>",
            FakeAligner(),
            sample_rate=24000,
            aligner_backend="qwen3",
            language="eng",
            pitch_method="psola+indextts2_cco",
            pitch_client=FakePitchClient(),
            f5_client=FakeF5Client(),
         rate_method=PARAMETERS["rate_method"], psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], crossfade_ms=PARAMETERS["crossfade_ms"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)


def test_apply_edits_sends_hard_splice_to_f5_smoothing():
    class FakeAligner:
        def align(self, _audio_bytes, _text, **_kwargs):
            return {
                "backend": "qwen3",
                "language": "eng",
                "level": "word",
                "answers": [{"start": 0.25, "end": 0.75}],
            }

    class FakePitchClient:
        def shift(self, **kwargs):
            segment_audio, segment_sr = wav_to_numpy(kwargs["audio_bytes"])
            return numpy_to_wav(np.full_like(segment_audio, 0.8), segment_sr)

    class FakeF5Client:
        def __init__(self):
            self.kwargs = None

        def smooth_boundaries(self, **kwargs):
            self.kwargs = kwargs
            return numpy_to_wav(kwargs["waveform"], kwargs["sample_rate"])

    fake_f5 = FakeF5Client()
    result = apply_edits_from_service(
        numpy_to_wav(np.zeros(24000, dtype=np.float32), 24000),
        "Hello <pitch, semitones=-4>world</pitch>",
        FakeAligner(),
        sample_rate=24000,
        aligner_backend="qwen3",
        language="eng",
        pitch_method="psola+indextts2_cco",
        pitch_client=FakePitchClient(),
        f5_client=fake_f5,
        crossfade_ms=40,
     rate_method=PARAMETERS["rate_method"], psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"], smooth_margin_sec=PARAMETERS["smooth_margin_sec"], f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01)

    assert result.smoothing_applied is True
    assert fake_f5.kwargs is not None
    f5_waveform = fake_f5.kwargs["waveform"]
    left_boundary = int(0.25 * 24000)
    assert f5_waveform[left_boundary - 1] == 0.0
    assert f5_waveform[left_boundary] == pytest.approx(0.8, abs=1e-4)
    assert fake_f5.kwargs["source_crossfade_sec"] == 0.0
    assert fake_f5.kwargs["regen_crossfade_sec"] == 0.01
    assert result.alignment_metadata["pre_f5_crossfade_ms"] == 0.0
    assert "fallback_crossfade_ms" not in result.alignment_metadata


def test_prompt_does_not_advertise_combined_or_overlapping_tags():
    prompt = build_prompt(
        "Hello beautiful world.",
        wave_configs=[
            {"wave_id": "1", "edit_kind": "mixed", "span_level": "clause"},
        ],
        pitch_values=[2, -2, 3],
        rate_values=[0.8, 1.2, 0.7],
    )

    assert "word_combined" not in prompt
    assert "sentence_combined" not in prompt
    assert "must not overlap or nest" in prompt
    assert "pitch_values" in prompt
    assert "rate_values" in prompt


def test_prompt_keeps_required_schema_while_staying_compact():
    prompt = build_prompt(
        "The quick brown fox jumps over the lazy dog.",
        wave_configs=[
            {"wave_id": "1", "edit_kind": "pitch", "span_level": "single_word"},
            {"wave_id": "2", "edit_kind": "rate", "span_level": "short_phrase"},
            {"wave_id": "3", "edit_kind": "mixed", "span_level": "clause"},
        ],
        pitch_values=[-6, -4, -2, 2, 4, 6, -1, 1, 8],
        rate_values=[0.75, 0.82, 0.96, 1.05, 1.17, 1.32, 0.6, 1.5, 1.9],
    )

    assert "pair_instructions" not in prompt
    assert "forward_nl" not in prompt
    assert "Do not generate natural-language instructions" in prompt
    assert "Single-wave response example" not in prompt
    assert '["1", "2", "3"]' in prompt
    assert "Required complete output skeleton" in prompt
    assert "Do not include `st`" in prompt
    assert len(prompt) < 7500


def test_chinese_prompt_uses_chinese_normalization_without_ascii_rule():
    prompt = build_prompt(
        "你好世界。",
        wave_configs=[
            {"wave_id": "1", "edit_kind": "rate", "span_level": "single_word"},
        ],
        pitch_values=[2, -2, 3],
        rate_values=[0.8, 1.2, 0.7],
        language="zh",
    )

    assert "Mandarin Chinese" in prompt
    assert "Do not romanize Chinese text" in prompt
    assert "ASCII only" not in prompt
    assert "你好世界" in prompt


class FakeWDTWClient:
    def __init__(self):
        self.calls = []

    def compute(self, **kwargs):
        self.calls.append(kwargs)
        return {"wdtw": 0.12, "status": "ok"}


def _single_pair_stage2_state(*, cached_alignments=True):
    pipeline_input = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world.",
                    "Hello <rate, factor=0.8>beautiful</rate> world.",
                )
            ]
        ),
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
    )
    assert pipeline_input is not None
    pair = pipeline_input.wave_group.edit_pairs[0]
    audio = numpy_to_wav(np.zeros(240, dtype=np.float32), 24000)
    alignment = {
        "level": "word",
        "alignments": [{"word": "Hello", "start": 0.0, "end": 0.1}],
    }
    if not cached_alignments:
        alignment = {"level": "word", "edit_regions": []}
    return {
        "pipeline_input": pipeline_input,
        "sample_id": "sample_001",
        "t_start": 0.0,
        "wave_results": {
            "1": {
                "result": {
                    "alignment": alignment,
                    "edits_applied": [],
                }
            }
        },
        "wave_audio_bytes": {"1": audio},
        "pair_inputs": [(pair, audio, audio)],
    }


def test_stage3_wdtw_reuses_cached_source_alignment_and_delegates_missing_side(
    monkeypatch,
):
    def fake_evaluate_pair(**_kwargs):
        return PairEvalResult(EvalMetrics(), EvalMetrics(), eval_time_sec=0.01)

    monkeypatch.setattr(pipeline_mod, "evaluate_pair", fake_evaluate_pair)
    wdtw = FakeWDTWClient()

    state = pipeline_mod.evaluate_sample(
        _single_pair_stage2_state(cached_alignments=True),
        asr_client=None,
        wer_client=None,
        utmos_client=None,
        spk_sim_client=None,
        do_evaluate=True,
        wdtw_client=wdtw,
    )

    assert len(wdtw.calls) == 1
    call = wdtw.calls[0]
    assert call["source_alignments"] == [{"word": "Hello", "start": 0.0, "end": 0.1}]
    assert call["target_alignments"] is None
    assert call["source_audio"] is not None
    assert call["target_audio"] is not None
    assert "aligner_url" not in call
    assert state["pair_results"][0]["eval_results"]["wdtw"] == 0.12


def test_stage3_wdtw_falls_back_to_audio_when_cached_alignments_missing(
    monkeypatch,
):
    def fake_evaluate_pair(**_kwargs):
        return PairEvalResult(EvalMetrics(), EvalMetrics(), eval_time_sec=0.01)

    monkeypatch.setattr(pipeline_mod, "evaluate_pair", fake_evaluate_pair)
    wdtw = FakeWDTWClient()

    pipeline_mod.evaluate_sample(
        _single_pair_stage2_state(cached_alignments=False),
        asr_client=None,
        wer_client=None,
        utmos_client=None,
        spk_sim_client=None,
        do_evaluate=True,
        wdtw_client=wdtw,
    )

    call = wdtw.calls[0]
    assert call["source_alignments"] is None
    assert call["target_alignments"] is None
    assert call["source_audio"] is not None
    assert call["target_audio"] is not None
    assert "aligner_url" not in call


def _wav_base64(samples: list[float] | None = None, sample_rate: int = 24000) -> str:
    sf = pytest.importorskip("soundfile")
    audio = np.asarray(
        samples if samples is not None else [0.0, 0.01, -0.01],
        dtype=np.float32,
    )
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _write_parquet(path: Path, rows: list[dict]):
    pd = pytest.importorskip("pandas")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _read_rows(path: Path) -> list[dict]:
    pd = pytest.importorskip("pandas")
    return pd.read_parquet(path).to_dict(orient="records")


def _prosody_new_schema_row(audio_id: str = "sample--original_to_1") -> dict:
    return {
        "from_dataset": "Emilia-ProsodyEdit-V1",
        "audio_id": audio_id,
        "sample_rate": 24000,
        "audios": json.dumps(
            {"src": _wav_base64([0.0, 0.01]), "tgt": _wav_base64([0.0, -0.01])}
        ),
        "duration": json.dumps({"src": 0.1, "tgt": 0.1}),
        "segments": json.dumps(
            {
                "src": {
                    "chunk_id": "src",
                    "wave_id": "original",
                    "is_original": True,
                    "start": 0.0,
                    "end": 0.1,
                    "label": "Hello world.",
                    "transcript": "Hello world.",
                    "edit_time_spans": [],
                    "alignment": {},
                    "language": "en",
                },
                "tgt": {
                    "chunk_id": "tgt",
                    "wave_id": "1",
                    "is_original": False,
                    "start": 0.0,
                    "end": 0.1,
                    "label": "<rate, factor=0.7>Hello</rate> world.",
                    "transcript": "Hello world.",
                    "edit_time_spans": [{"text": "Hello", "span": [0.0, 0.05]}],
                    "alignment": {"edit_regions": []},
                    "language": "en",
                },
            }
        ),
        "instructions": json.dumps(
            {"nl": "Slow down hello.", "st": "..."},
            ensure_ascii=False,
        ),
        "eval_results": "",
        "edit_metadata": json.dumps({"edit_kind": "rate"}),
    }


def test_prosody_parquet_rows_write_segments_dict_and_pair_metadata():
    pipeline_input = generate_input(
        FakeSampler(),
        FakeLLMClient(
            [
                _llm_response(
                    "Hello beautiful world.",
                    "Hello <rate, factor=0.8>beautiful</rate> world.",
                )
            ]
        ),
        edit_kind="rate",
        span_level="single_word",
        num_waves=1,
    )
    assert pipeline_input is not None
    pair = pipeline_input.wave_group.edit_pairs[0]
    audio = numpy_to_wav(np.zeros(240, dtype=np.float32), 24000)
    stage3_state = {
        "pipeline_input": pipeline_input,
        "sample_id": "sample_001",
        "metadata": {
            "sample_id": "sample_001",
            "clip_id": pipeline_input.clip_id,
            "input_id": "input_001",
            "plain_text": pipeline_input.wave_group.plain_text,
            "normed_text": pipeline_input.normed_text,
            "language": "en",
            "tagged_texts": pipeline_input.wave_group.tagged_texts,
            "wave_specs": pipeline_input.metadata["wave_specs"],
            "sample_rate": 24000,
            "waves": {
                "1": {
                    "result": {
                        "alignment": {
                            "level": "word",
                            "alignments": [
                                {"word": "Hello", "start": 0.0, "end": 0.1}
                            ],
                            "edit_regions": [
                                {
                                    "text": "beautiful",
                                    "start_sec": 0.1,
                                    "end_sec": 0.2,
                                }
                            ],
                        },
                        "edits_applied": [
                            {
                                "words": "beautiful",
                                "target_span": [0.1, 0.25],
                            }
                        ],
                    }
                }
            },
        },
        "pair_results": [
            {
                "pair_dir_name": "original_to_1",
                "source_id": pair.source_id,
                "target_id": pair.target_id,
                "source_audio_bytes": audio,
                "target_audio_bytes": audio,
                "instructions": pair.instructions,
                "source_tagged_text": pair.source_tagged_text,
                "target_tagged_text": pair.target_tagged_text,
                "tag_diff": pair.tag_diff,
                "eval_results": {"wdtw": 0.0, "wdtw_result": {"wdtw": 0.0}},
            }
        ],
    }

    rows = ProsodyParquetRowProcessor().rows_from_payload(
        stage3_state,
        dataset_name="unit-dataset",
    )

    assert len(rows) == 1
    segments = json.loads(rows[0]["segments"])
    assert set(segments) == {"src", "tgt"}
    assert segments["src"]["wave_id"] == "original"
    assert segments["tgt"]["wave_id"] == "1"
    pair_metadata = json.loads(rows[0]["pair_metadata"])
    assert pair_metadata["sample_id"] == "sample_001"
    assert pair_metadata["src_id"] == "original"
    assert pair_metadata["tgt_id"] == "1"
    assert pair_metadata["edited_generation_id"] == "1"
    assert json.loads(rows[0]["tag_diff"])
    assert json.loads(rows[0]["eval_results"])["wdtw"] == 0.0


def _install_fake_prosody_eval(monkeypatch, calls: list[dict]):
    class DummyClient:
        def __init__(self, url):
            self.url = url

    def fake_evaluate_pair(**kwargs):
        calls.append(kwargs)
        return PairEvalResult(
            source_metrics=EvalMetrics(
                asr_text="Hello world.",
                wer=0.0,
                wer_backend="wer_service",
                wer_metric="wer",
                utmos_score=4.0,
            ),
            target_metrics=EvalMetrics(
                asr_text="Hello world.",
                wer=0.1,
                wer_backend="wer_service",
                wer_metric="wer",
                utmos_score=3.9,
                spk_similarity=0.95,
            ),
            eval_time_sec=0.5,
        )

    monkeypatch.setattr(pipeline_mod, "ASRClient", DummyClient)
    monkeypatch.setattr(pipeline_mod, "WERClient", DummyClient)
    monkeypatch.setattr(pipeline_mod, "UTMOSClient", DummyClient)
    monkeypatch.setattr(pipeline_mod, "SpkSimClient", DummyClient)
    monkeypatch.setattr(pipeline_mod, "evaluate_pair", fake_evaluate_pair)


def test_stage_i_llm_client_timeout_allows_service_long_retry():
    llm = FakeLLMClient(
        [
            json.dumps(
                {
                    "normed_text": "Hello beautiful world.",
                    "waves": {
                        "1": {
                            "tagged_text": "Hello <rate, factor=0.8>beautiful</rate> world.",
                            "pair_instructions": _pair_instructions(
                                "1",
                                "Hello <rate, factor=0.8>beautiful</rate> world.",
                            ),
                        }
                    },
                }
            )
        ]
    )

    result = generate_input(
        FakeSampler(),
        llm,
        edit_kind="rate",
        num_waves=1,
        span_level="single_word",
    )

    assert result is not None
    assert llm.kwargs[0]["timeout"] == LLM_SERVICE_TIMEOUT_SEC


def _service_wav(duration_sec: float = 1.0) -> bytes:
    from speech_edit.pipelines.prosody.editor import numpy_to_wav

    samples = int(24000 * duration_sec)
    audio = np.zeros(samples, dtype=np.float32)
    return numpy_to_wav(audio, 24000)


def _patch_service_core(monkeypatch):
    from speech_edit.pipelines.prosody.service import api_service

    captured = {}

    def fake_apply_edits(audio_bytes, tagged_text, aligner_client, sample_rate, **kwargs):
        captured["tagged_text"] = tagged_text
        captured["kwargs"] = kwargs
        return EditResult(
            edited_audio=np.ones(1200, dtype=np.float32) * 0.05,
            sample_rate=sample_rate,
            edits_applied=[
                {
                    "type": "service_fake",
                    "text": "beautiful",
                    "start_sec": 0.1,
                    "end_sec": 0.3,
                }
            ],
            edit_time_sec=0.01,
            boundary_times=[0.1, 0.3],
            alignment_metadata={"backend": kwargs["aligner_backend"]},
        )

    monkeypatch.setattr(pipeline_mod, "apply_edits_from_service", fake_apply_edits)
    monkeypatch.setattr(
        api_service,
        "_make_services",
        lambda *, evaluate, smooth_policy: {
            "aligner_client": object(),
            "pitch_client": object(),
            "rate_client": object(),
            "f5_client": object() if smooth_policy != "never" else None,
            "asr_client": None,
            "wer_client": None,
            "utmos_client": None,
            "spk_sim_client": None,
        },
    )
    return api_service, captured


def generate_input(sampler, llm_client, **kwargs):
    """Exercise the production sampling and generation stages together."""
    segment = next(sampler)
    sampled = input_generator.Stage0Sample(
        clip_id=segment.clip_id, transcript=segment.text,
        audio_bytes=segment.audio_bytes, sample_rate=segment.sample_rate,
        duration=segment.duration, metadata={"clip_id": segment.clip_id},
    )
    return input_generator.generate_input_from_sampled_data(sampled, llm_client, **({key: PARAMETERS[key] for key in ("parameter_policy", "edit_kind_weights", "span_level_weights", "num_waves", "forward_nl_mode", "reverse_nl_mode")} | {"temperature": PARAMETERS["llm_temperature"], "llm_backend": None} | kwargs), max_retries=3)


def test_service_persistence_rejects_sample_id_outside_artifacts(tmp_path, monkeypatch):
    from speech_edit.pipelines.prosody.service import api_service

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(api_service, "_decode_audio", lambda _: (b"audio", 24000, 1.0))
    monkeypatch.setattr(api_service, "_make_services", lambda **_: pytest.fail("Must reject before inference"))
    request = api_service.ExecuteRequest(
        audio={"data": "unused", "format": "wav"}, source_text="Hello.",
        instruction_st="<pitch, semitones=2>Hello.</pitch>", persist=True,
        evaluate=False, sample_id=str(tmp_path / "outside"),
    )
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        api_service.execute(request)
    assert not (tmp_path / "outside").exists()


@pytest.mark.parametrize("persist", [False, True])
def test_execute_persistence_matches_response_metadata(tmp_path, monkeypatch, persist):
    from fastapi.testclient import TestClient

    api, _ = _patch_service_core(monkeypatch)
    original_make_services = api._make_services
    monkeypatch.setattr(api, "_make_services", lambda **kw: {
        **original_make_services(**kw), "f0_client": None, "wdtw_client": None,
    })
    wav = _service_wav()
    output_dir = tmp_path / "interactive"
    request = {
        "audio": {"data": base64.b64encode(wav).decode("ascii")},
        "source_text": "Hello beautiful world.",
        "instruction_st": "Hello <pitch, semitones=2>beautiful</pitch> world.",
        "sample_id": "sample", "persist": persist, "evaluate": False,
        "output_dir": str(output_dir),
    }
    response = TestClient(api.app).post("/execute", json=request)
    assert response.status_code == 200, response.text
    payload = response.json()
    if persist:
        sample_dir = output_dir / "sample"
        assert payload["metadata"]["sample_dir"] == str(sample_dir)
        assert json.loads((sample_dir / "metadata.json").read_text()) == payload["metadata"]
        assert (sample_dir / "audios/original.wav").read_bytes() == wav
        assert (sample_dir / "audios/generated_1.wav").read_bytes() == base64.b64decode(payload["audio"]["data"])
        assert (sample_dir / "edit_pairs/original_to_1/source.wav").read_bytes() == wav
    else:
        assert not output_dir.exists()
        assert "sample_dir" not in payload["metadata"]


@pytest.mark.parametrize("kind", ["pitch", "rate"])
def test_transform_service_failure_propagates(kind):
    from speech_edit.pipelines.prosody.editor import _run_pitch_shift, _run_rate_stretch
    from types import SimpleNamespace

    calls = []
    def fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError("transform failed")

    with pytest.raises(RuntimeError, match="transform failed"):
        if kind == "pitch":
            _run_pitch_shift(np.zeros(2400, dtype=np.float32), 24000, 2,
                             pitch_client=SimpleNamespace(shift=fail), text="hello", pitch_method=PARAMETERS["pitch_method"], psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"])
        else:
            _run_rate_stretch(np.zeros(2400, dtype=np.float32), 24000, 1.2,
                              rate_client=SimpleNamespace(stretch=fail), text="hello",
                              spk_audio_bytes=b"speaker", rate_method=PARAMETERS["rate_method"])
    assert len(calls) == 1


def test_edit_spans_match_integer_splice_boundaries_after_rate_change():
    class FakeAligner:
        def align(self, _audio_bytes, _text, **_kwargs):
            return {
                "backend": "qwen3", "language": "eng", "level": "word",
                "answers": [{"start": 1.04, "end": 1.44}, {"start": 2.32, "end": 2.72}],
            }

    class FakeRateClient:
        def stretch(self, **kwargs):
            segment, sr = wav_to_numpy(kwargs["audio_bytes"])
            return numpy_to_wav(np.pad(segment, (0, 17)), sr)

    class FakePitchClient:
        def shift(self, **kwargs):
            return kwargs["audio_bytes"]

    source = numpy_to_wav(np.linspace(-0.5, 0.5, 72000, dtype=np.float32), 24000)
    decoded, _ = wav_to_numpy(source)
    result = apply_edits_from_service(
        source, 'One <rate, factor=0.8>two</rate> and <pitch, semitones=2>three</pitch>.',
        FakeAligner(), sample_rate=24000, aligner_backend="qwen3", language="eng",
        pitch_method=PARAMETERS["pitch_method"], rate_method=PARAMETERS["rate_method"],
        psola_indextts2_cco_pitch_boost=PARAMETERS["psola_indextts2_cco_pitch_boost"],
        pitch_client=FakePitchClient(), rate_client=FakeRateClient(), crossfade_ms=0,
        f5_client=None, smooth_margin_sec=PARAMETERS["smooth_margin_sec"],
        f5_steps=32, f5_seed=None, f5_source_crossfade_sec=0.0, f5_regen_crossfade_sec=0.01,
    )
    # 2.32 seconds floors to sample 55679 in the actual source slicer.
    expected_source = [[1.04, 1.44], [2.32, 2.72]]
    expected_target = [(24960, 34577), (55696, 65297)]
    for edit, source_span, target_span in zip(result.edits_applied, expected_source, expected_target):
        assert edit["source_span"] == source_span
        assert edit["target_span"] == [sample / 24000 for sample in target_span]
    assert result.boundary_times == [sample / 24000 for span in expected_target for sample in span]
    np.testing.assert_array_equal(result.edited_audio[34577:55696], decoded[34560:55679])


@pytest.mark.parametrize("forward_nl_mode", ["template", "llm"])
@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("edit_kind", ["pitch", "rate", "mixed"])
def test_rendered_prompt_json_and_single_word_examples(
    forward_nl_mode, language, edit_kind,
):
    prompt = build_prompt(
        "One, two, it is one, two, you." if language == "en" else "我们明天讨论。",
        wave_configs=[{
            "wave_id": "1", "edit_kind": edit_kind,
            "span_level": "single_word", "max_regions": 2,
        }],
        pitch_values=[4, -3], rate_values=[1.2, 0.8],
        language=language, forward_nl_mode=forward_nl_mode,
    )
    format_json = prompt.split("Respond with a JSON object:\n", 1)[1].split("\n\nImportant:", 1)[0]
    assert set(json.loads(format_json)) == {"normed_text", "waves"}
    skeleton = prompt.split("Required complete output skeleton:\n", 1)[1].split("\n\nNow generate", 1)[0]
    assert set(json.loads(skeleton)["waves"]) == {"1"}
    examples = [
        json.loads(line.removeprefix("Tagged-text example: "))
        for line in prompt.splitlines() if line.startswith("Tagged-text example: ")
    ]
    assert examples
    for example in examples:
        segments = parse_tags(example["tagged_text"])
        assert "".join(segment.text for segment in segments) == example["normed_text"]
        marked = [segment for segment in segments if segment.is_tagged]
        assert len(marked) == 2
        if language == "en":
            assert all(re.fullmatch(r"[A-Za-z]+", segment.text) for segment in marked)
            if edit_kind == "rate":
                assert [segment.text for segment in marked] == ["fox", "dog"]
        else:
            assert [segment.text for segment in marked] == {
                "pitch": ["美丽", "晚霞"],
                "rate": ["明天", "讨论"],
                "mixed": ["事情", "突然"],
            }[edit_kind]
        if forward_nl_mode == "llm":
            assert set(example["forward_nl"]) == {"precise", "vague"}
        else:
            assert "forward_nl" not in example


@pytest.mark.parametrize('tagged,plain', [
    ('ＩＰＯ<pitch, semitones=3>发行</pitch>数量', 'ＩＰＯ发行数量'),
    ('中国<pitch, semitones=3>ＩＰＯ</pitch>发行', '中国ＩＰＯ发行'),
    ('IPO<pitch, semitones=3>发行</pitch>数量', 'IPO发行数量'),
    ('中国<pitch, semitones=3>IPO</pitch>发行', '中国IPO发行'),
])
def test_prosody_mixed_script_boundaries_share_validation_with_wave_construction(tagged, plain):
    assert validate_tagged_text(tagged, plain) == (True, '')
    assert strip_tags(tagged) == plain
    assert len(build_wave_group(plain, {'1': tagged}).edit_pairs) == 2


@pytest.mark.parametrize('tagged,plain', [
    ('I<pitch, semitones=3>PO</pitch>发行', 'IPO发行'),
    ('Ｉ<pitch, semitones=3>ＰＯ</pitch>发行', 'ＩＰＯ发行'),
    ('<pitch, semitones=3>fav</pitch>orite', 'favorite'),
    ('<pitch, semitones=3>ｆａｖ</pitch>ｏｒｉｔｅ', 'ｆａｖｏｒｉｔｅ'),
])
def test_prosody_word_interior_is_rejected_before_plan_acceptance(tagged, plain):
    valid, error = validate_tagged_text(tagged, plain)
    assert not valid and 'Tag boundary' in error
    with pytest.raises(ValueError, match='Tag boundary'):
        input_generator._validate_wave_result(
            wave_id='1', result={'tagged_text': tagged}, normed_text=plain,
            wave_config={'edit_kind': 'pitch', 'span_level': 'single_word', 'max_regions': 3},
            pitch_values=[3], rate_values=[1.2],
        )
    with pytest.raises(ValueError, match='Tag boundary'):
        build_wave_group(plain, {'1': tagged})


def test_prosody_actual_fullwidth_ipo_plan_reply_builds_all_waves():
    plain = '二零一七年是中国股市走向成熟的一年。两市ＩＰＯ发行数量穷冠事件。'
    waves = {
        '1': '二零一七年是<pitch, semitones=3>中国股市</pitch>走向成熟的一年。两市ＩＰＯ<pitch, semitones=-8>发行数量</pitch>穷冠事件。',
        '2': '<rate, factor=0.82>二零一七年是中国股市</rate>走向成熟的一年。<pitch, semitones=5>两市ＩＰＯ发行数量</pitch>穷冠事件。',
        '3': '二零一七年是<pitch, semitones=2>中国</pitch>股市走向<pitch, semitones=-7>成熟</pitch>的一年。两市ＩＰＯ发行数量<pitch, semitones=9>穷</pitch>冠事件。',
    }
    for wave_id, tagged in waves.items():
        assert strip_tags(tagged) == plain
        assert validate_tagged_text(tagged, plain) == (True, '')
        input_generator._validate_wave_result(
            wave_id=wave_id, result={'tagged_text': tagged}, normed_text=plain,
            wave_config={'edit_kind': 'mixed' if wave_id == '2' else 'pitch',
                         'span_level': {'1': 'short_phrase', '2': 'clause', '3': 'single_word'}[wave_id],
                         'max_regions': 3},
            pitch_values=[2, 3, 5, 9, -7, -8], rate_values=[0.82],
        )
    group = build_wave_group(plain, waves)
    assert group.plain_text == plain
    assert len(group.edit_pairs) == 6
    assert group.tagged_texts == waves
