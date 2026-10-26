from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/emotion_style_edit/defaults")

import base64
import json
import os
import random
import re
from pathlib import Path

import pytest

from speech_edit.clients.services import ServiceError
from speech_edit.pipelines.emotion import input_generator
from speech_edit.pipelines.emotion import prompts
from speech_edit.pipelines.emotion import synthesis_profile
from speech_edit.pipelines.emotion import synthesizer
from speech_edit.pipelines.emotion.service_contract import (
    build_service_edit_contract,
)
from speech_edit.pipelines.emotion.speaker_sources import (
    resolve_language_speaker_parquet_sources,
)
from speech_edit.pipelines.emotion.tag_utils import (
    build_tagged_text,
    convert_levels_to_alpha,
    describe_tag_diff,
    parse_tags,
    EditPair,
    render_emo_tag,
    TaggedTextGroup,
    TextBoundary,
    texts_equivalent_ignoring_safe_whitespace,
)
from speech_edit.data.parquet.reader import Segment


class _StaticLLMClient:
    def __init__(self, response: str):
        self.response = response

    def generate(self, *args, **kwargs):
        return self.response


class _FakeAssemblyClient:
    def __init__(self, result: bytes = b"RIFFfakeassembled"):
        self.result = result
        self.calls = []

    def assemble(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeSmootherClient:
    def __init__(self, result: bytes = b"RIFFsmoothed"):
        self.result = result
        self.calls = []

    def smooth(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


def _service_wav() -> bytes:
    return synthesizer._numpy_to_wav(
        synthesizer.np.zeros(24000, dtype=synthesizer.np.float32),
        24000,
    )


def _make_group(index: int) -> TaggedTextGroup:
    return TaggedTextGroup(

        sub_type="single",
        plain_text=f"sample text {index}",
        speaker_prompt_audio=b"RIFFfakewav",
        speaker_prompt_text=f"prompt {index}",
        tagged_texts={
            "1": f"<emo type=\"happy\">sample text {index}</emo>",
            "2": f"<emo type=\"sad\">sample text {index}</emo>",
        },
        edit_pairs=[
            EditPair(
                source_id="1",
                target_id="2",
                nl_instruction="make it sad",
                st_instruction="happy->sad",
            )
        ],
        boundaries=[TextBoundary(char_idx=6)],
        metadata={
            "original_plain_text": f"orig {index}",
            "was_rewritten": False,
            "llm_prompt": f"prompt {index}",
            "speaker_prompt_clip_id": f"clip_{index}",
        },
    )


def _make_emotion_plan(edit_mode="type_change", emotion_prompt_mode="level"):
    return input_generator.EmotionEditPlan(
        allowed_emotions=["happy", "angry", "sad", "surprised"],
        edit_mode=edit_mode,
        emotion_prompt_mode=emotion_prompt_mode,
        emotion_tag_weights={emotion: 1.0 for emotion in input_generator.EMOTION_TAGS},
    )


def test_select_emotion_edit_plan_defaults_equal_weights_fixed_seed():
    plan = input_generator.select_emotion_edit_plan(rng=random.Random(123), emotion_edit_mode_probs=PARAMETERS["emotion_edit_mode_probs"], emotion_tag_weights=PARAMETERS["emotion_tag_weights"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})

    assert plan.edit_mode == "type_change"
    assert plan.emotion_prompt_mode == "level"
    assert plan.allowed_emotions == ["afraid", "happy", "calm", "angry", "melancholic"]
    assert len(set(plan.allowed_emotions)) == 5
    assert plan.emotion_tag_weights == {
        emotion: 1.0 for emotion in input_generator.EMOTION_TAGS
    }


def test_select_emotion_edit_plan_uses_condition_scoped_edit_mode_defaults():
    type_plan = input_generator.select_emotion_edit_plan(
        emotion_prompt_mode_probs={"level": 1.0, "description": 0.0},
        emotion_edit_mode_probs={
            "type_level": {"type_change": 0.0, "intensity_change": 1.0},
            "desc": {"type_change": 1.0, "intensity_change": 0.0},
        },
        rng=random.Random(3),
     emotion_tag_weights=PARAMETERS["emotion_tag_weights"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})
    desc_plan = input_generator.select_emotion_edit_plan(
        emotion_prompt_mode_probs={"level": 0.0, "description": 1.0},
        rng=random.Random(3),
     emotion_edit_mode_probs=PARAMETERS["emotion_edit_mode_probs"], emotion_tag_weights=PARAMETERS["emotion_tag_weights"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})

    assert type_plan.emotion_prompt_mode == "level"
    assert type_plan.edit_mode == "intensity_change"
    assert desc_plan.emotion_prompt_mode == "description"
    assert desc_plan.edit_mode == "type_change"


def test_no_self_transition_matrix_preserves_target_distribution():
    weights = {
        emotion: (0.05 if emotion in {"calm", "disgusted"} else 0.15)
        for emotion in input_generator.EMOTION_TAGS
    }
    matrix = input_generator.build_no_self_transition_matrix(weights)

    for source, row in matrix.items():
        assert row[source] == 0.0
        assert sum(row.values()) == pytest.approx(1.0)
    for target in input_generator.EMOTION_TAGS:
        stationary_probability = sum(
            weights[source] * matrix[source][target]
            for source in input_generator.EMOTION_TAGS
        )
        assert stationary_probability == pytest.approx(weights[target])


def test_strict_normal_plan_uses_two_or_three_regions_and_full_coverage():
    weights = {
        emotion: (0.05 if emotion in {"calm", "disgusted"} else 0.15)
        for emotion in input_generator.EMOTION_TAGS
    }
    for seed in range(30):
        plan = input_generator.select_emotion_edit_plan(
            emotion_prompt_mode_probs={"level": 1.0, "description": 0.0},
            emotion_tag_weights=weights,
            strict=True,
            sub_type="multi_emo_normal",
            rng=random.Random(seed),
         emotion_edit_mode_probs=PARAMETERS["emotion_edit_mode_probs"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})
        assert plan.strict is True
        assert plan.edit_mode == "type_change"
        assert plan.region_count in {2, 3}
        assert set(plan.change_masks["1->2"]) | set(plan.change_masks["2->3"]) == set(
            range(plan.region_count)
        )
        assert 0 < len(plan.change_masks["1->2"]) < plan.region_count
        assert 0 < len(plan.change_masks["2->3"]) < plan.region_count
        assert plan.emotion_matrix["1"] != plan.emotion_matrix["3"]


def test_safe_whitespace_equivalence_protects_english_word_boundaries():
    assert texts_equivalent_ignoring_safe_whitespace("你 好，world !", "你好，world!")
    assert texts_equivalent_ignoring_safe_whitespace("hello world", "hello   world")
    assert not texts_equivalent_ignoring_safe_whitespace("hello world", "helloworld")
    assert not texts_equivalent_ignoring_safe_whitespace("hello world", "hello worlds")
    assert not texts_equivalent_ignoring_safe_whitespace('"Hello"', "Hello")
    assert not texts_equivalent_ignoring_safe_whitespace("'Hello'", "Hello")
    assert not texts_equivalent_ignoring_safe_whitespace("a—b", "a--b")


def test_intertag_whitespace_repair_moves_only_whitespace_into_left_tag():
    repaired, count = input_generator._absorb_intertag_whitespace(
        '<emo type="happy" level="2">hello</emo>  '
        '<emo type="sad" level="2">world</emo>'
    )
    assert count == 1
    assert repaired == (
        '<emo type="happy" level="2">hello  </emo>'
        '<emo type="sad" level="2">world</emo>'
    )
    unchanged, count = input_generator._absorb_intertag_whitespace(
        '<emo type="happy" level="2">hello</emo>,'
        '<emo type="sad" level="2">world</emo>'
    )
    assert count == 0
    assert unchanged.endswith("</emo>,<emo type=\"sad\" level=\"2\">world</emo>")


def test_strict_matrix_prompt_and_validation_leave_only_boundaries_to_llm():
    plan = input_generator.EmotionEditPlan(
        allowed_emotions=list(input_generator.EMOTION_TAGS),
        edit_mode="type_change",
        emotion_prompt_mode="level",
        emotion_tag_weights={emotion: 0.125 for emotion in input_generator.EMOTION_TAGS},
        strict=True,
        region_count=2,
        emotion_matrix={
            "1": ["happy", "sad"],
            "2": ["angry", "sad"],
            "3": ["angry", "calm"],
        },
        change_masks={"1->2": [0], "2->3": [1]},
        region_levels=[2, 1],
        coverage_mode="all",
    )
    response = json.dumps(
        {
            "canonical_text": "Hello world, this is fine.",
            "tagged_texts": {
                "1": (
                    '<emo type="happy" level="2">Hello world,</emo> '
                    '<emo type="sad" level="1">this is fine.</emo>'
                ),
                "2": (
                    '<emo type="angry" level="2">Hello world,</emo> '
                    '<emo type="sad" level="1">this is fine.</emo>'
                ),
                "3": (
                    '<emo type="angry" level="2">Hello world,</emo> '
                    '<emo type="calm" level="1">this is fine.</emo>'
                ),
            },
        }
    )
    tagged_texts, edit_pairs, canonical_text = input_generator._generate_tagged_texts(
        "Hello world, this is fine.",
        "multi_emo_normal",
        _StaticLLMClient(response),
        emotion_edit_plan=plan,
        natural_language_instruction_mode="template",
    )

    assert canonical_text == "Hello world, this is fine."
    assert len(tagged_texts) == 3
    assert len(edit_pairs) == 6
    assert tagged_texts["1"].startswith(
        '<emo type="happy" level="2">Hello world, </emo><emo'
    )
    prompt = prompts.get_prompt(
        "multi_emo_normal",
        canonical_text,
        emotion_edit_plan=plan.to_prompt_dict(),
        natural_language_instruction_mode="template",
    )
    assert "Variant 1: region 1=happy, region 2=sad" in prompt
    assert "Split into exactly 2 complete clauses" in prompt
    assert "choose emotion" not in prompt.lower()


def test_strict_description_plan_reuses_desc_iff_emotion_is_unchanged():
    plan = input_generator.EmotionEditPlan(
        allowed_emotions=list(input_generator.EMOTION_TAGS),
        edit_mode="type_change",
        emotion_prompt_mode="description",
        emotion_tag_weights={emotion: 0.125 for emotion in input_generator.EMOTION_TAGS},
        strict=True,
        region_count=3,
        emotion_matrix={
            "1": ["happy", "sad", "afraid"],
            "2": ["angry", "sad", "afraid"],
            "3": ["happy", "calm", "afraid"],
        },
        change_masks={"1->2": [0], "2->3": [0, 1]},
        region_levels=[2, 1, 2],
        coverage_mode="partial",
    )

    def response(*, final_happy_desc: str, angry_desc: str) -> str:
        tagged_texts = {
            "1": (
                '<emo desc="bright cheerful delivery" type_ref="happy">One two </emo>'
                '<emo desc="quiet lingering sadness" type_ref="sad">three four </emo>'
                '<emo desc="tense trembling fear" type_ref="afraid">five six</emo>'
            ),
            "2": (
                f'<emo desc="{angry_desc}" type_ref="angry">One two </emo>'
                '<emo desc="quiet lingering sadness" type_ref="sad">three four </emo>'
                '<emo desc="tense trembling fear" type_ref="afraid">five six</emo>'
            ),
            "3": (
                f'<emo desc="{final_happy_desc}" type_ref="happy">One two </emo>'
                '<emo desc="steady gentle calm" type_ref="calm">three four </emo>'
                '<emo desc="tense trembling fear" type_ref="afraid">five six</emo>'
            ),
        }
        return json.dumps(
            {"canonical_text": "One two three four five six", "tagged_texts": tagged_texts}
        )

    _tagged_texts, edit_pairs, _canonical_text = input_generator._generate_tagged_texts(
        "One two three four five six",
        "multi_emo_frequent",
        _StaticLLMClient(
            response(
                final_happy_desc="bright cheerful delivery",
                angry_desc="sharp forceful anger",
            )
        ),
        emotion_edit_plan=plan,
        natural_language_instruction_mode="template",
    )
    assert len(edit_pairs) == 6

    with pytest.raises(ValueError, match="reuse the same desc"):
        input_generator._generate_tagged_texts(
            "One two three four five six",
            "multi_emo_frequent",
            _StaticLLMClient(
                response(
                    final_happy_desc="different radiant joy",
                    angry_desc="sharp forceful anger",
                )
            ),
            emotion_edit_plan=plan,
            natural_language_instruction_mode="template",
        )
    with pytest.raises(ValueError, match="use different desc values"):
        input_generator._generate_tagged_texts(
            "One two three four five six",
            "multi_emo_frequent",
            _StaticLLMClient(
                response(
                    final_happy_desc="bright cheerful delivery",
                    angry_desc="bright cheerful delivery",
                )
            ),
            emotion_edit_plan=plan,
            natural_language_instruction_mode="template",
        )


@pytest.mark.parametrize(
    ("description", "emotion", "error"),
    [
        ("happy", "happy", "label wrapper"),
        ("happy tone", "happy", "label wrapper"),
        ("very happy tone", "happy", "label wrapper"),
        ("a happy voice", "happy", "label wrapper"),
        ("a sad tone", "sad", "label wrapper"),
        ("an angry delivery", "angry", "label wrapper"),
        ("happy vocal tone", "happy", "label wrapper"),
        ("sad voice", "sad", "label wrapper"),
        ("悲伤语气", "sad", "label wrapper"),
        ("非常愤怒的声音", "angry", "label wrapper"),
        ("忧郁语气", "melancholic", "label wrapper"),
        ("带有惆怅的声音", "melancholic", "label wrapper"),
        ("惊讶语气", "surprised", "label wrapper"),
        ("稍微吃惊的嗓音", "surprised", "label wrapper"),
        ("平静语气", "calm", "label wrapper"),
        ("非常镇定的声音", "calm", "label wrapper"),
        ("soft sadness", "sad", "too short"),
    ],
)
def test_strict_description_rejects_labels_and_short_phrases(
    description: str,
    emotion: str,
    error: str,
):
    with pytest.raises(ValueError, match=error):
        input_generator._validate_strict_emotion_description(
            description,
            expected_emotion=emotion,
            text_id="1",
            region=0,
        )


@pytest.mark.parametrize(
    ("description", "emotion"),
    [
        ("bright and cheerful delivery", "happy"),
        ("tense, trembling breath", "afraid"),
        ("低沉而带有哀伤感", "sad"),
        ("轻快明亮的带笑语气", "happy"),
    ],
)
def test_strict_description_accepts_concise_bilingual_vocal_phrases(
    description: str,
    emotion: str,
):
    input_generator._validate_strict_emotion_description(
        description,
        expected_emotion=emotion,
        text_id="1",
        region=0,
    )


def test_strict_description_prompt_requires_natural_vocal_phrases():
    plan = input_generator.select_emotion_edit_plan(
        emotion_prompt_mode_probs={"description": 1.0, "level": 0.0},
        strict=True,
        sub_type="single",
        rng=random.Random(7),
     emotion_edit_mode_probs=PARAMETERS["emotion_edit_mode_probs"], emotion_tag_weights=PARAMETERS["emotion_tag_weights"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})

    prompt = prompts.get_prompt(
        "single",
        "Hello there",
        emotion_edit_plan=plan.to_prompt_dict(),
        natural_language_instruction_mode="template",
    )

    assert "3-8 English words or 4-16 Chinese characters" in prompt
    assert "Do not copy the emotion label" in prompt
    assert "bright and cheerful delivery" in prompt
    assert "低沉而带有哀伤感" in prompt


def test_emotion_edit_mode_probs_reject_flat_and_invalid_weights():
    with pytest.raises(ValueError, match="type_change"):
        input_generator.normalize_emotion_edit_mode_probs(
            {"type_change": 1.0, "intensity_change": 0.0},
            emotion_condition_mode="type_level",
        )
    with pytest.raises(ValueError, match="unknown"):
        input_generator.normalize_emotion_edit_mode_probs(
            {"type_level": {"unknown": 1.0}},
            emotion_condition_mode="type_level",
        )
    with pytest.raises(ValueError, match="must be non-negative"):
        input_generator.normalize_emotion_edit_mode_probs(
            {"type_level": {"type_change": -1.0}},
            emotion_condition_mode="type_level",
        )
    with pytest.raises(ValueError, match="positive total"):
        input_generator.normalize_emotion_edit_mode_probs(
            {"type_level": {"type_change": 0.0, "intensity_change": 0.0}},
            emotion_condition_mode="type_level",
        )


def test_select_emotion_edit_plan_custom_weights_without_replacement():
    weights = {
        "happy": 10.0,
        "angry": 10.0,
        "sad": 10.0,
        "afraid": 10.0,
        "disgusted": 10.0,
        "melancholic": 0.0,
        "surprised": 0.0,
        "calm": 0.0,
    }
    plan = input_generator.select_emotion_edit_plan(
        emotion_tag_weights=weights,
        rng=random.Random(1),
     emotion_edit_mode_probs=PARAMETERS["emotion_edit_mode_probs"], emotion_region_count_weights={"single": {1: 1.0}, "multi_emo_normal": {2: 0.75, 3: 0.25}, "multi_emo_frequent": {2: 0.9, 3: 0.1}})

    assert set(plan.allowed_emotions) == {
        "happy",
        "angry",
        "sad",
        "afraid",
        "disgusted",
    }
    assert len(plan.allowed_emotions) == len(set(plan.allowed_emotions)) == 5


def test_emotion_prompt_uses_allowed_emotions_and_no_calm_bias():
    plan = _make_emotion_plan("type_change").to_prompt_dict()
    prompt = prompts.build_multi_normal_prompt(
        "hello there, general kenobi",
        emotion_edit_plan=plan,
    )

    assert "`allowed_emotions`: happy, angry, sad, surprised" in prompt
    assert "`edit_mode` is `type_change`" in prompt
    assert "Encourage using the calm tag" not in prompt
    assert "calm" not in prompt


def test_emotion_template_prompt_omits_instruction_outputs():
    prompt = prompts.build_multi_normal_prompt(
        "hello there, general kenobi",
        emotion_edit_plan=_make_emotion_plan("type_change").to_prompt_dict(),
        natural_language_instruction_mode="template",
    )

    assert "nl_inst" not in prompt
    assert "st_inst" not in prompt
    assert "structured instruction" not in prompt.lower()
    assert "Do not generate edit instructions" in prompt


def test_emotion_llm_prompt_keeps_nl_but_omits_st_outputs():
    prompt = prompts.build_single_prompt(
        "hello there",
        emotion_edit_plan=_make_emotion_plan("type_change").to_prompt_dict(),
        natural_language_instruction_mode="llm",
    )

    assert "nl_inst" in prompt
    assert "st_inst" not in prompt
    assert "structured instruction" not in prompt.lower()


def test_generate_tagged_texts_rejects_emotion_outside_allowed_set(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"calm\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {"1->2": "make it calm"},
            "st_inst": {"1->2": ""},
        }
    )

    try:
        input_generator._generate_tagged_texts(
            plain_text="Hello there",
            sub_type="single",
            llm_client=_StaticLLMClient(llm_response),
            emotion_edit_plan=_make_emotion_plan("type_change"),
        )
    except ValueError as exc:
        assert "outside allowed_emotions" in str(exc)
    else:
        raise AssertionError("Expected out-of-plan emotion to fail")


def test_generate_tagged_texts_template_mode_builds_pairs_without_llm_instructions(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"angry\" level=\"2\">Hello there</emo>",
            },
        }
    )

    tagged_texts, edit_pairs, canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("type_change"),
        natural_language_instruction_mode="template",
    )

    assert canonical == "Hello there"
    assert {(pair.source_id, pair.target_id) for pair in edit_pairs} == {
        ("1", "2"),
        ("2", "1"),
        ("1", "3"),
        ("3", "1"),
        ("2", "3"),
        ("3", "2"),
    }
    first = next(pair for pair in edit_pairs if pair.source_id == "1" and pair.target_id == "2")
    assert first.nl_instruction.startswith("Please change")
    assert (
        first.st_instruction
        == '<emo type="sad" level="2">Hello there</emo>'
    )
    assert describe_tag_diff(tagged_texts["1"], tagged_texts["2"])


def test_generate_tagged_texts_llm_mode_uses_local_structured_instruction(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"angry\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {"1->2": "llm structured output must be ignored"},
        }
    )

    _tagged_texts, edit_pairs, _canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("type_change"),
        natural_language_instruction_mode="llm",
    )

    assert len(edit_pairs) == 1
    assert edit_pairs[0].nl_instruction == "make it sad"
    assert edit_pairs[0].st_instruction == '<emo type="sad" level="2">Hello there</emo>'


def test_generate_tagged_texts_records_exact_llm_prompt(monkeypatch):
    monkeypatch.setattr(
        input_generator,
        "get_prompt",
        lambda *args, **kwargs: "exact prompt with allowed_emotions",
    )
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"angry\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {},
        }
    )
    diagnostics = {}

    input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("type_change"),
        diagnostics=diagnostics,
    )

    assert diagnostics["llm_prompt"] == "exact prompt with allowed_emotions"
    assert diagnostics["llm_raw_response"] == llm_response


def test_generate_tagged_texts_uses_gemma_compatible_output_token_cap(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"angry\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {},
        }
    )

    class _CapturingLLMClient:
        def __init__(self):
            self.kwargs = None

        def generate(self, *args, **kwargs):
            self.kwargs = kwargs
            return llm_response

    llm_client = _CapturingLLMClient()
    input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=llm_client,
        emotion_edit_plan=_make_emotion_plan("type_change"),
    )

    assert llm_client.kwargs["max_tokens"] == input_generator.STAGE1_LLM_MAX_TOKENS
    assert llm_client.kwargs["max_tokens"] <= 4096


def test_generate_tagged_texts_keeps_diff_pairs_for_type_change_plan(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"happy\" level=\"3\">Hello there</emo>",
                "3": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {
                "1->2": "make it happier",
                "2->1": "make it less happy",
                "1->3": "make it sad",
                "3->1": "make it happy",
                "2->3": "make it sad",
                "3->2": "make it happy",
            },
            "st_inst": {},
        }
    )

    tagged_texts, edit_pairs, _canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("type_change"),
    )

    assert {(ep.source_id, ep.target_id) for ep in edit_pairs} == {
        ("1", "2"),
        ("2", "1"),
        ("1", "3"),
        ("3", "1"),
        ("2", "3"),
        ("3", "2"),
    }
    assert input_generator.summarize_prompted_edit_modes(
        edit_pairs,
        _make_emotion_plan("type_change"),
    ) == {"type_change": 6}


def test_generate_tagged_texts_keeps_diff_pairs_for_intensity_plan(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"1\">Hello there</emo>",
                "2": "<emo type=\"happy\" level=\"3\">Hello there</emo>",
                "3": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {
                "1->2": "make it happier",
                "2->1": "make it less happy",
                "1->3": "make it sad",
                "3->1": "make it happy",
            },
            "st_inst": {},
        }
    )

    _tagged_texts, edit_pairs, _canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("intensity_change"),
    )

    pairs_by_ids = {(ep.source_id, ep.target_id): ep for ep in edit_pairs}
    assert set(pairs_by_ids) == {("1", "2"), ("2", "1")}
    assert pairs_by_ids[("1", "2")].st_instruction == (
        '<emo type="happy" level="inc">Hello there</emo>'
    )
    assert pairs_by_ids[("2", "1")].st_instruction == (
        '<emo type="happy" level="dec">Hello there</emo>'
    )


def test_type_level_intensity_st_marks_only_changed_spans():
    source = (
        '<emo type="happy" level="1">Hello there, </emo>'
        '<emo type="sad" level="2">general kenobi</emo>'
    )
    target = (
        '<emo type="happy" level="3">Hello there, </emo>'
        '<emo type="sad" level="2">general kenobi</emo>'
    )

    st_instruction = input_generator.build_template_st_instruction(
        source,
        target,
        emotion_edit_plan=_make_emotion_plan("intensity_change"),
    )

    assert st_instruction == (
        '<emo type="happy" level="inc">Hello there, </emo>general kenobi'
    )


def test_generate_tagged_texts_drops_pairs_without_tag_diff(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"sad\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {
                "1->2": "keep it happy",
                "1->3": "make it sad",
            },
            "st_inst": {},
        }
    )

    _tagged_texts, edit_pairs, _canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan("type_change"),
    )

    assert [(ep.source_id, ep.target_id) for ep in edit_pairs] == [("1", "3")]


def test_generate_tagged_texts_rejects_when_all_pairs_have_no_tag_diff(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "2": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
                "3": "<emo type=\"happy\" level=\"2\">Hello there</emo>",
            },
            "nl_inst": {
                "1->2": "keep it happy",
                "2->3": "keep it happy",
            },
            "st_inst": {},
        }
    )

    with pytest.raises(ValueError, match="No edit pairs contained a tag diff"):
        input_generator._generate_tagged_texts(
            plain_text="Hello there",
            sub_type="single",
            llm_client=_StaticLLMClient(llm_response),
            emotion_edit_plan=_make_emotion_plan("type_change"),
        )


def test_generate_tagged_texts_accepts_description_mode(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": '<emo desc="soft, tearful sadness" type_ref="sad">Hello there</emo>',
                "2": '<emo desc="bright relief with a smile" type_ref="happy">Hello there</emo>',
                "3": '<emo desc="tense, clipped anger" type_ref="angry">Hello there</emo>',
            },
            "nl_inst": {"1->2": "make it relieved", "2->3": "make it angry"},
            "st_inst": {
                "1->2": '<emo desc="bright relief with a smile" type_ref="happy">Hello there</emo>',
                "2->3": '<emo desc="tense, clipped anger" type_ref="angry">Hello there</emo>',
            },
        }
    )

    tagged_texts, edit_pairs, canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan(
            "type_change", emotion_prompt_mode="description"
        ),
    )

    assert canonical == "Hello there"
    assert len(edit_pairs) == 2
    assert describe_tag_diff(tagged_texts["1"], tagged_texts["2"])[0][
        "target_emotion_type_ref"
    ] == "happy"
    assert edit_pairs[0].st_instruction == (
        '<emo desc="bright relief with a smile">Hello there</emo>'
    )
    assert "type_ref" not in edit_pairs[0].st_instruction


def test_desc_intensity_prompt_requests_st_inst_without_type_ref():
    prompt = prompts.build_single_prompt(
        "Hello there",
        emotion_edit_plan=_make_emotion_plan(
            "intensity_change", emotion_prompt_mode="description"
        ).to_prompt_dict(),
        natural_language_instruction_mode="template",
    )

    assert '"st_inst"' in prompt
    assert "Do NOT include `type_ref` in `st_inst`" in prompt
    assert "Do not generate natural-language edit instructions" in prompt


def test_generate_tagged_texts_desc_intensity_uses_llm_st_without_type_ref(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": '<emo desc="soft happy delivery" type_ref="happy">Hello there</emo>',
                "2": '<emo desc="bright, intense happy delivery" type_ref="happy">Hello there</emo>',
                "3": '<emo desc="soft sad delivery" type_ref="sad">Hello there</emo>',
            },
            "st_inst": {
                "1->2": '<emo desc="增强快乐的情感">Hello there</emo>',
                "2->1": '<emo desc="soften the happy emotion">Hello there</emo>',
                "1->3": '<emo desc="make it brighter">Hello there</emo>',
            },
        }
    )

    _tagged_texts, edit_pairs, _canonical = input_generator._generate_tagged_texts(
        plain_text="Hello there",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        emotion_edit_plan=_make_emotion_plan(
            "intensity_change", emotion_prompt_mode="description"
        ),
        natural_language_instruction_mode="template",
    )

    pairs_by_ids = {(ep.source_id, ep.target_id): ep for ep in edit_pairs}
    assert set(pairs_by_ids) == {("1", "2"), ("2", "1")}
    assert pairs_by_ids[("1", "2")].st_instruction == (
        '<emo desc="增强快乐的情感">Hello there</emo>'
    )
    assert "type_ref" not in pairs_by_ids[("1", "2")].st_instruction


def test_generate_tagged_texts_desc_intensity_rejects_bad_st_inst(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": '<emo desc="soft happy delivery" type_ref="happy">Hello there</emo>',
                "2": '<emo desc="bright, intense happy delivery" type_ref="happy">Hello there</emo>',
                "3": '<emo desc="brighter happy delivery" type_ref="happy">Hello there</emo>',
            },
            "st_inst": {
                "1->2": '<emo desc="增强快乐的情感" type_ref="happy">Hello there</emo>',
            },
        }
    )

    with pytest.raises(ValueError, match="must not include type_ref"):
        input_generator._generate_tagged_texts(
            plain_text="Hello there",
            sub_type="single",
            llm_client=_StaticLLMClient(llm_response),
            emotion_edit_plan=_make_emotion_plan(
                "intensity_change", emotion_prompt_mode="description"
            ),
            natural_language_instruction_mode="template",
        )


def test_synthesis_profile_estimates_code_switching_duration():
    assert synthesis_profile.estimate_text_duration_sec("hello world", en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"]) == pytest.approx(
        0.44
    )
    assert synthesis_profile.estimate_text_duration_sec("你好世界", en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"]) == pytest.approx(0.8)
    assert synthesis_profile.estimate_text_duration_sec("hello 世界", en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"]) == pytest.approx(
        0.62
    )
    assert synthesis_profile.estimate_text_duration_sec(
        "hello 世界",
        en_word_sec=0.30,
        cjk_char_sec=0.10,
    ) == pytest.approx(0.50)
    assert synthesis_profile.ted_max_mel_tokens_from_duration_sec(22.0) == 1100


def test_synthesis_profile_selects_legal_window_and_profile():
    segments = [
        Segment("s1", "Hello world.", b"1", 1.0, 16000),
        Segment("s2", "你好世界。", b"2", 1.0, 16000),
    ]
    profile, selected = synthesis_profile.select_profile_and_segments(
        segments,
        rng=random.Random(7),
        synthesis_mode="multi_emo_frequent",
        emotion_condition_mode="desc",
     duration_estimation_en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], duration_estimation_cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"], ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"], synthesis_profile_weights=PARAMETERS["synthesis_profile_weights"])

    assert profile["synthesis_backend"] == "ted"
    assert profile["synthesis_mode"] == "multi_emo_frequent"
    assert profile["emotion_condition_mode"] == "desc"
    assert selected
    assert profile["estimated_duration_sec"] <= 17.0
    assert profile["ted_max_mel_tokens"] == 1100


def test_synthesis_profile_weights_choose_profile_before_window():
    segments = [
        Segment("s1", "One two. Three four.", b"1", 1.0, 16000),
        Segment("s2", "Five six.", b"2", 1.0, 16000),
    ]

    profile, selected = synthesis_profile.select_profile_and_segments(
        segments,
        rng=random.Random(3),
        synthesis_profile_weights={
            "single": {"type_level": 0.0, "desc": 0.0},
            "multi_emo_normal": {"type_level": 1.0, "desc": 0.0},
            "multi_emo_frequent": {"desc": 0.0},
        },
        language="en",
        plan_text_unit_cap_en=2,
     duration_estimation_en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], duration_estimation_cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"], ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])

    assert profile["synthesis_backend"] == "index"
    assert profile["synthesis_mode"] == "multi_emo_normal"
    assert profile["selected_text_window"]["text_unit_count"] <= 2
    assert len(selected) == 1


def test_synthesis_profile_condition_matrix_routes_backend():
    assert (
        synthesis_profile.build_synthesis_profile(
            synthesis_mode="single",
            emotion_condition_mode="desc",
         ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])["synthesis_backend"]
        == "index"
    )
    assert (
        synthesis_profile.build_synthesis_profile(
            synthesis_mode="multi_emo_normal",
            emotion_condition_mode="desc",
         ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])["synthesis_backend"]
        == "index"
    )
    assert (
        synthesis_profile.build_synthesis_profile(
            synthesis_mode="multi_emo_frequent",
            emotion_condition_mode="desc",
         ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])["synthesis_backend"]
        == "ted"
    )
    with pytest.raises(ValueError, match="multi_emo_frequent \\+ type_level"):
        synthesis_profile.build_synthesis_profile(
            synthesis_mode="multi_emo_frequent",
            emotion_condition_mode="type_level",
         ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])
    with pytest.raises(ValueError, match="desc_dur"):
        synthesis_profile.build_synthesis_profile(
            synthesis_mode="single",
            emotion_condition_mode="desc_dur",
         ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])


def test_synthesis_profile_weights_reject_old_backend_keys():
    with pytest.raises(ValueError, match="single.index"):
        synthesis_profile.normalize_synthesis_profile_weights(
            {"single": {"index": 1.0}}
        )


def test_synthesis_profile_rejects_frequent_desc_when_window_exceeds_limit():
    long_text = " ".join(f"word{i}" for i in range(100))
    segments = [Segment("s1", long_text, b"1", 30.0, 16000)]

    with pytest.raises(ValueError, match="No legal synthesis profile"):
        synthesis_profile.select_profile_and_segments(
            segments,
            rng=random.Random(0),
            synthesis_mode="multi_emo_frequent",
            emotion_condition_mode="desc",
            language="en",
         duration_estimation_en_word_sec=PARAMETERS["duration_estimation_en_word_sec"], duration_estimation_cjk_char_sec=PARAMETERS["duration_estimation_cjk_char_sec"], ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"], synthesis_profile_weights=PARAMETERS["synthesis_profile_weights"])


def test_generate_group_desc_profile_forces_description_prompt_without_duration(monkeypatch):
    speaker_prompt = Segment("spk_1", "speaker prompt", b"speaker", 1.2, 16000)
    gen_segments = [
        Segment("gen_1", "hello there friend", b"gen1", 2.5, 16000),
        Segment("gen_2", "general kenobi returns", b"gen2", 2.8, 16000),
    ]
    llm_response = json.dumps(
        {
            "canonical_text": "hello there friend general kenobi returns",
            "tagged_texts": {
                "1": (
                    '<emo desc="bright, cheerful, brisk" type_ref="happy">hello there friend </emo>'
                    '<emo desc="calm, steady" type_ref="calm">general kenobi returns</emo>'
                ),
                "2": (
                    '<emo desc="sad, soft, slower" type_ref="sad">hello there friend </emo>'
                    '<emo desc="calm, steady" type_ref="calm">general kenobi returns</emo>'
                ),
                "3": (
                    '<emo desc="sad, soft, slower" type_ref="sad">hello there friend </emo>'
                    '<emo desc="angry, clipped, faster" type_ref="angry">general kenobi returns</emo>'
                ),
            },
            "nl_inst": {"1->2": "make hello sad", "2->3": "make kenobi angry"},
            "st_inst": {},
        }
    )

    monkeypatch.setattr(
        input_generator,
        "maybe_rewrite_text",
        lambda text, llm_client, **kwargs: (text, False),
    )
    group = input_generator.generate_tagged_text_group_attempt(
        speaker_prompt=speaker_prompt,
        gen_segments=gen_segments,
        language='en',
        llm_client=_StaticLLMClient(llm_response),
        stage1_seed=123,
        emotion_tag_weights={'happy': 1.0, 'sad': 1.0, 'calm': 1.0, 'angry': 1.0, 'surprised': 1.0},
        synthesis_profile=synthesis_profile.build_synthesis_profile(
            synthesis_mode='multi_emo_normal',
            emotion_condition_mode='desc',
            estimated_duration_sec=1.2,
            ted_estimated_duration_soft_limit_sec=PARAMETERS['ted_estimated_duration_soft_limit_sec'],
            ted_max_generated_duration_sec=PARAMETERS['ted_max_generated_duration_sec'],
        ),
        emotion_edit_mode_probs=PARAMETERS['emotion_edit_mode_probs'],
        natural_language_instruction_mode='llm',
        rewrite_prob=0.0,
        strict_emotion_plan=False,
        emotion_region_count_weights=PARAMETERS['emotion_region_count_weights'],
        max_stage1_text_words_en=PARAMETERS['plan_text_unit_cap_en'],
        max_stage1_text_words_zh=PARAMETERS['plan_text_unit_cap_zh'],
    )

    assert group.sub_type == "multi_emo_normal"
    assert group.metadata["emotion_edit_plan"]["emotion_prompt_mode"] == "description"
    assert "dur" not in group.metadata["llm_prompt"]
    assert [segment.emotion_type_ref for segment in parse_tags(group.tagged_texts["1"])] == [
        "happy",
        "calm",
    ]


def test_generate_group_records_llm_prompt_in_synthesis_profile(monkeypatch):
    speaker_prompt = Segment("spk_1", "speaker prompt", b"speaker", 1.2, 16000)
    gen_segments = [Segment("gen_1", "hello there", b"gen1", 2.5, 16000)]
    llm_response = json.dumps(
        {
            "canonical_text": "hello there",
            "tagged_texts": {
                "1": '<emo desc="calm, steady" type_ref="calm">hello there</emo>',
                "2": '<emo desc="sad, soft" type_ref="sad">hello there</emo>',
                "3": '<emo desc="angry, clipped" type_ref="angry">hello there</emo>',
            },
            "nl_inst": {"1->2": "make it sad", "2->3": "make it angry"},
            "st_inst": {},
        }
    )
    profile = synthesis_profile.build_synthesis_profile(
        synthesis_mode="single",
        emotion_condition_mode="desc",
        selected_text_window={"text": "hello there", "estimated_duration_sec": 0.44},
        estimated_duration_sec=0.44,
     ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])

    monkeypatch.setattr(
        input_generator,
        "maybe_rewrite_text",
        lambda text, llm_client, **kwargs: (text, False),
    )
    group = input_generator.generate_tagged_text_group_attempt(
        speaker_prompt=speaker_prompt,
        gen_segments=gen_segments,
        language='en',
        llm_client=_StaticLLMClient(llm_response),
        stage1_seed=123,
        synthesis_profile=profile,
        emotion_tag_weights={'calm': 1.0, 'sad': 1.0, 'angry': 1.0, 'happy': 1.0, 'surprised': 1.0},
        emotion_edit_mode_probs=PARAMETERS['emotion_edit_mode_probs'],
        natural_language_instruction_mode='llm',
        rewrite_prob=0.0,
        strict_emotion_plan=False,
        emotion_region_count_weights=PARAMETERS['emotion_region_count_weights'],
        max_stage1_text_words_en=PARAMETERS['plan_text_unit_cap_en'],
        max_stage1_text_words_zh=PARAMETERS['plan_text_unit_cap_zh'],
    )

    profile_meta = group.metadata["synthesis_profile"]
    assert profile_meta["synthesis_backend"] == "index"
    assert profile_meta["emotion_condition_mode"] == "desc"
    assert profile_meta["llm_prompt"] == group.metadata["llm_prompt"]
    assert profile_meta["llm_raw_response"] == llm_response
    assert "Retry Feedback" not in profile_meta["llm_prompt"]
    assert "llm_raw_response" not in group.metadata["stage1_diagnostics"]


def test_index_profile_prompt_is_level_only():
    prompt = prompts.build_single_prompt(
        "hello there",
        emotion_edit_plan={
            "allowed_emotions": ["happy", "sad"],
            "edit_mode": "type_change",
            "emotion_prompt_mode": "level",
        },
    )

    assert "alpha" not in prompt.lower()
    assert '<emo type="TAG_NAME" level="N">text</emo>' in prompt
    assert "Every non-neutral emotion tag MUST include an explicit `level`" in prompt


def test_index_profile_rejects_llm_alpha_tags():
    profile = synthesis_profile.build_synthesis_profile(
        synthesis_mode="single",
        emotion_condition_mode="type_level",
     ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"], ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"])

    with pytest.raises(ValueError, match="explicit level"):
        synthesis_profile.validate_tagged_texts_for_profile(
            {"1": '<emo type="happy" alpha="0.93">hello there</emo>'},
            profile,
            require_explicit_index_level=True,
        )


def test_generate_tagged_texts_description_mode_requires_type_ref(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there",
            "tagged_texts": {
                "1": '<emo desc="soft sadness">Hello</emo><emo desc="calm delivery" type_ref="calm"> there</emo>',
                "2": '<emo desc="angry delivery" type_ref="angry">Hello</emo><emo desc="calm delivery" type_ref="calm"> there</emo>',
                "3": '<emo desc="calm delivery" type_ref="calm">Hello</emo><emo desc="calm delivery" type_ref="calm"> there</emo>',
            },
            "nl_inst": {"1->2": "make hello angry"},
            "st_inst": {"1->2": '<emo desc="angry delivery" type_ref="angry">Hello</emo> there'},
        }
    )

    with pytest.raises(ValueError, match="type_ref"):
        input_generator._generate_tagged_texts(
            plain_text="Hello there",
            sub_type="multi_emo_normal",
            llm_client=_StaticLLMClient(llm_response),
            emotion_edit_plan=_make_emotion_plan(
                "type_change", emotion_prompt_mode="description"
            ),
        )


def test_generate_tagged_text_group_metadata_records_emotion_plan(monkeypatch):
    speaker_prompt = Segment("spk_1", "speaker prompt", b"speaker", 1.2, 16000)
    gen_segments = [
        Segment("gen_1", "hello there", b"gen1", 2.5, 16000),
        Segment("gen_2", "general kenobi", b"gen2", 2.8, 16000),
    ]
    llm_response = json.dumps(
        {
            "canonical_text": "hello there general kenobi",
            "tagged_texts": {
                "1": "<emo type=\"happy\" level=\"2\">hello there general kenobi</emo>",
                "2": "<emo type=\"sad\" level=\"2\">hello there general kenobi</emo>",
                "3": "<emo type=\"angry\" level=\"2\">hello there general kenobi</emo>",
            },
            "nl_inst": {
                "1->2": "make it sad",
                "2->1": "make it happy",
                "1->3": "make it angry",
            },
            "st_inst": {},
        }
    )

    monkeypatch.setattr(
        input_generator,
        "maybe_rewrite_text",
        lambda text, llm_client, **kwargs: (text, False),
    )
    group = input_generator.generate_tagged_text_group_attempt(
        speaker_prompt=speaker_prompt,
        gen_segments=gen_segments,
        language='en',
        llm_client=_StaticLLMClient(llm_response),
        rewrite_prob=0.0,
        stage1_seed=123,
        emotion_edit_mode_probs={'type_level': {'type_change': 1.0, 'intensity_change': 0.0}, 'desc': {'type_change': 1.0, 'intensity_change': 0.0}},
        emotion_tag_weights={'happy': 1.0, 'angry': 1.0, 'sad': 1.0, 'afraid': 1.0, 'disgusted': 0.0, 'melancholic': 0.0, 'surprised': 1.0, 'calm': 0.0},
        synthesis_profile=synthesis_profile.build_synthesis_profile(
            synthesis_mode='single',
            emotion_condition_mode='type_level',
            ted_estimated_duration_soft_limit_sec=PARAMETERS['ted_estimated_duration_soft_limit_sec'],
            ted_max_generated_duration_sec=PARAMETERS['ted_max_generated_duration_sec'],
        ),
        natural_language_instruction_mode='llm',
        strict_emotion_plan=False,
        emotion_region_count_weights=PARAMETERS['emotion_region_count_weights'],
        max_stage1_text_words_en=PARAMETERS['plan_text_unit_cap_en'],
        max_stage1_text_words_zh=PARAMETERS['plan_text_unit_cap_zh'],
    )

    plan = group.metadata["emotion_edit_plan"]
    assert plan["edit_mode"] == "type_change"
    assert set(plan["allowed_emotions"]) == {
        "happy",
        "angry",
        "sad",
        "afraid",
        "surprised",
    }
    assert group.metadata["pair_change_summary"] == {"type_change": 3}
    assert group.metadata["llm_prompt"]
    assert "`allowed_emotions`" in group.metadata["llm_prompt"]
    assert "llm_prompt" not in group.metadata["stage1_diagnostics"]


def test_default_rewrite_probability_is_zero():
    assert input_generator.TEXT_REWRITE_PROB == 0.0


def test_synthesizer_maps_pipeline_language_to_aligner_code():
    assert synthesizer._aligner_language_code("en") == "eng"
    assert synthesizer._aligner_language_code("zh") == "zho"
    assert synthesizer._aligner_language_code("eng") == "eng"


def test_generate_tagged_texts_uses_internal_canonical_text_not_raw_input(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    llm_response = json.dumps(
        {
            "normed_text": "this is ignored in favor of tagged text consensus",
            "tagged_texts": {
                "1": "<emo type=\"happy\">Hello there doctor</emo>",
                "2": "<emo type=\"sad\">Hello there doctor</emo>",
                "3": "<emo type=\"calm\">Hello there doctor</emo>",
            },
            "nl_inst": {
                "1->2": "make it sad",
                "1->3": "make it calm",
            },
            "st_inst": {"1->2": "", "1->3": ""},
        }
    )

    tagged_texts, edit_pairs, canonical_text = input_generator._generate_tagged_texts(
        plain_text="raw ### noisy text",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
    )

    assert canonical_text == "Hello there doctor"
    assert tagged_texts["1"] == "<emo type=\"happy\">Hello there doctor</emo>"
    assert len(edit_pairs) == 2


def test_generate_tagged_texts_accepts_explicit_canonical_text_field(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    llm_response = json.dumps(
        {
            "canonical_text": "Hello there doctor",
            "tagged_texts": {
                "1": "<emo type=\"happy\">Hello there doctor</emo>",
                "2": "<emo type=\"sad\">Hello there doctor</emo>",
                "3": "<emo type=\"calm\">Hello there doctor</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {"1->2": ""},
        }
    )

    tagged_texts, edit_pairs, canonical_text = input_generator._generate_tagged_texts(
        plain_text="raw noisy text",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
    )

    assert canonical_text == "Hello there doctor"
    assert tagged_texts["2"] == "<emo type=\"sad\">Hello there doctor</emo>"
    assert len(edit_pairs) == 1


def test_generate_tagged_texts_rejects_internally_inconsistent_tagged_texts(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    llm_response = json.dumps(
        {
            "normed_text": "some text",
            "tagged_texts": {
                "1": "<emo type=\"happy\">Hello there</emo>",
                "2": "<emo type=\"sad\">Hello doctor</emo>",
            },
            "nl_inst": {"1->2": "change it"},
            "st_inst": {"1->2": ""},
        }
    )

    try:
        input_generator._generate_tagged_texts(
            plain_text="raw noisy text",
            sub_type="single",
            llm_client=_StaticLLMClient(llm_response),
        )
    except ValueError as exc:
        assert "not internally consistent" in str(exc)
    else:
        raise AssertionError("Expected internal canonical text mismatch to fail")


def test_generate_tagged_texts_repairs_single_variants_by_majority(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    diagnostics = {
        "llm_generate_sec": 0.0,
        "postprocess_sec": 0.0,
        "repair_sec": 0.0,
        "single_repair_applied": False,
    }
    llm_response = json.dumps(
        {
            "canonical_text": "Hello there doctor",
            "tagged_texts": {
                "1": "<emo type=\"happy\">Hello there doctor</emo>",
                "2": "<emo type=\"sad\">Hello there doctor</emo>",
                "3": "<emo type=\"calm\">Hello there dockter</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {"1->2": ""},
        }
    )

    tagged_texts, _edit_pairs, canonical_text = input_generator._generate_tagged_texts(
        plain_text="raw noisy text",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        diagnostics=diagnostics,
    )

    assert canonical_text == "Hello there doctor"
    assert tagged_texts["3"] == "<emo type=\"calm\">Hello there doctor</emo>"
    assert diagnostics["single_repair_applied"] is True
    assert diagnostics["repair_sec"] >= 0.0


def test_generate_tagged_texts_majority_repair_preserves_surface_text(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    llm_response = json.dumps(
        {
            "canonical_text": "ignored",
            "tagged_texts": {
                "1": '<emo type=\"happy\">"Hello   there"</emo>',
                "2": '<emo type=\"sad\">"Hello   there"</emo>',
                "3": "<emo type=\"calm\">Hello there</emo>",
            },
            "nl_inst": {"1->2": "make it sad"},
            "st_inst": {"1->2": ""},
        }
    )

    tagged_texts, _edit_pairs, canonical_text = input_generator._generate_tagged_texts(
        plain_text="raw noisy text",
        sub_type="single",
        llm_client=_StaticLLMClient(llm_response),
        diagnostics={
            "llm_generate_sec": 0.0,
            "postprocess_sec": 0.0,
            "repair_sec": 0.0,
            "single_repair_applied": False,
        },
    )

    assert canonical_text == '"Hello   there"'
    assert tagged_texts["3"] == '<emo type=\"calm\">"Hello   there"</emo>'


def test_generate_tagged_texts_single_three_way_mismatch_still_fails(monkeypatch):
    monkeypatch.setattr(input_generator, "get_prompt", lambda *args, **kwargs: "prompt")

    llm_response = json.dumps(
        {
            "canonical_text": "ignored",
            "tagged_texts": {
                "1": "<emo type=\"happy\">Hello there doctor</emo>",
                "2": "<emo type=\"sad\">Hello there nurse</emo>",
                "3": "<emo type=\"calm\">Hello there patient</emo>",
            },
            "nl_inst": {"1->2": "change it"},
            "st_inst": {"1->2": ""},
        }
    )

    try:
        input_generator._generate_tagged_texts(
            plain_text="raw noisy text",
            sub_type="single",
            llm_client=_StaticLLMClient(llm_response),
            diagnostics={
                "llm_generate_sec": 0.0,
                "postprocess_sec": 0.0,
                "repair_sec": 0.0,
                "single_repair_applied": False,
            },
        )
    except ValueError as exc:
        assert "not internally consistent" in str(exc)
    else:
        raise AssertionError("Expected three-way mismatch to fail without repair")


def test_generate_tagged_texts_single_repair_requires_strict_majority(monkeypatch):
    repaired, canonical_text = input_generator._repair_single_tagged_texts_by_majority(
        {
            "1": "<emo type=\"happy\">Hello</emo>",
            "2": "<emo type=\"sad\">Hello</emo>",
            "3": "<emo type=\"calm\">Hi</emo>",
            "4": "<emo type=\"angry\">Hey</emo>",
        }
    )

    assert repaired is None
    assert canonical_text is None


def test_resolve_language_speaker_parquet_sources_from_root_dir(tmp_path: Path):
    root = tmp_path / "speaker_timbre_library"
    zh_dir = root / "zh"
    en_dir = root / "en"
    zh_dir.mkdir(parents=True)
    en_dir.mkdir(parents=True)
    zh_manifest = zh_dir / "parquet_files.list"
    en_manifest = en_dir / "parquet_files.list"
    zh_manifest.write_text("/abs/zh.parquet\n", encoding="utf-8")
    en_manifest.write_text("/abs/en.parquet\n", encoding="utf-8")

    resolved = resolve_language_speaker_parquet_sources([str(root)])

    assert resolved == {"en": [str(en_manifest)], "zh": [str(zh_manifest)]}


def test_resolve_language_speaker_parquet_sources_ignores_empty_manifests(
    tmp_path: Path,
):
    root = tmp_path / "speaker_timbre_library"
    zh_dir = root / "zh"
    en_dir = root / "en"
    zh_dir.mkdir(parents=True)
    en_dir.mkdir(parents=True)
    (zh_dir / "parquet_files.list").write_text("\n", encoding="utf-8")
    en_manifest = en_dir / "parquet_files.list"
    en_manifest.write_text("/abs/en.parquet\n", encoding="utf-8")

    resolved = resolve_language_speaker_parquet_sources([str(root)])

    assert resolved == {"en": [str(en_manifest)]}


def test_resolve_language_speaker_parquet_sources_skips_empty_explicit_manifest(
    tmp_path: Path,
):
    zh_dir = tmp_path / "speaker_timbre_library" / "zh"
    en_dir = tmp_path / "speaker_timbre_library" / "en"
    zh_dir.mkdir(parents=True)
    en_dir.mkdir(parents=True)
    zh_manifest = zh_dir / "parquet_files.list"
    en_manifest = en_dir / "parquet_files.list"
    zh_manifest.write_text("/abs/zh.parquet\n", encoding="utf-8")
    en_manifest.write_text("\n", encoding="utf-8")

    resolved = resolve_language_speaker_parquet_sources(
        [str(en_manifest), str(zh_manifest)]
    )

    assert resolved == {"zh": [str(zh_manifest)]}


def test_convert_levels_to_alpha_maps_level_zero_to_none_tag():
    tagged = "<emo type=\"happy\" level=\"0\">Hello</emo><emo type=\"sad\" level=\"2\">World</emo>"
    converted = convert_levels_to_alpha(tagged)
    assert converted == "<emo type=\"none\">Hello</emo><emo type=\"sad\" alpha=\"0.9\">World</emo>"


def test_convert_levels_to_alpha_materializes_default_alpha_for_bare_emotion_tags():
    tagged = "<emo type=\"happy\">Hello</emo><emo type=\"none\">still</emo><emo type=\"sad\" level=\"none\">World</emo>"

    converted = convert_levels_to_alpha(tagged)

    assert (
        converted
        == "<emo type=\"happy\" alpha=\"0.93\">Hello</emo><emo type=\"none\">still</emo><emo type=\"sad\" alpha=\"0.9\">World</emo>"
    )


def test_parse_render_and_diff_preserve_duration_targets():
    tagged = (
        '<emo desc="soft, slower sadness" type_ref="sad" dur="1.2">Hello</emo>'
        '<emo type="none" dur="0.8"> world</emo>'
    )

    segments = parse_tags(tagged)

    assert segments[0].dur == 1.2
    assert segments[0].emotion_type_ref == "sad"
    assert segments[1].dur == 0.8
    assert render_emo_tag(
        text="Hello",
        emotion_desc="soft, slower sadness",
        emotion_type_ref="sad",
        dur=1.2,
    ) == '<emo desc="soft, slower sadness" type_ref="sad" dur="1.2">Hello</emo>'
    assert build_tagged_text(
        [
            (None, "Hello", None, "soft, slower sadness", 1.2, "sad"),
            ("none", " world", None, None, 0.8),
        ]
    ) == tagged

    diffs = describe_tag_diff(
        '<emo desc="soft sadness" type_ref="sad" dur="1.2">Hello</emo>',
        '<emo desc="soft sadness" type_ref="sad" dur="1.6">Hello</emo>',
    )
    assert diffs[0]["source_dur"] == 1.2
    assert diffs[0]["target_dur"] == 1.6
    assert diffs[0]["target_emotion_type_ref"] == "sad"


def test_parse_tags_uses_emotion_specific_level_mapping_and_default_alpha():
    segments = parse_tags(
        "<emo type=\"surprised\" level=\"3\">Wait</emo><emo type=\"happy\">Go</emo><emo type=\"angry\" level=\"none\">Now</emo>"
    )

    assert segments[0].alpha == 0.88
    assert segments[0].effective_alpha == 0.88
    assert segments[1].alpha is None
    assert segments[1].effective_alpha == 0.93
    assert segments[2].alpha is None
    assert segments[2].effective_alpha == 0.85


def test_service_contract_builds_full_tagged_and_assembly_markup():
    contract = build_service_edit_contract(
        source_text="Hello beautiful world today.",
        instruction_st="Hello <emo type=\"happy\" level=\"2\">beautiful world</emo> today.",
    )

    assert contract.plain_text == "Hello beautiful world today."
    assert contract.target_tagged_text == (
        "<emo type=\"none\">Hello </emo>"
        "<emo type=\"happy\" alpha=\"0.93\">beautiful world</emo>"
        "<emo type=\"none\"> today.</emo>"
    )
    assert contract.source_tagged_text == (
        "<emo type=\"none\">Hello </emo><emo type=\"none\">beautiful world</emo><emo type=\"none\"> today.</emo>"
    )
    assert contract.edit_tagged_text == "Hello <target>beautiful world</target> today."
    assert contract.changed_segments == [
        {
            "segment_idx": 1,
            "segment_text": "beautiful world",
            "source_tag": "none",
            "target_tag": "happy",
            "source_alpha": None,
            "target_alpha": 0.93,
            "target_emotion_control_mode": "alpha",
            "target_emotion_desc": None,
            "target_emotion_type_ref": None,
            "source_dur": None,
            "target_dur": None,
        }
    ]
    assert contract.target_spans[0]["start_char"] == len("Hello ")
    assert contract.target_spans[0]["end_char"] == len("Hello beautiful world")
    assert contract.synthesis_sub_type == "multi_emo_frequent"


def test_service_contract_allows_safe_whitespace_but_rejects_english_joining():
    contract = build_service_edit_contract(
        source_text="你好， world!",
        instruction_st='你 好，<emo type="happy" level="2">world !</emo>',
    )
    assert contract.plain_text == "你 好，world !"

    with pytest.raises(ValueError, match="preserve source_text"):
        build_service_edit_contract(
            source_text="hello world",
            instruction_st='<emo type="happy" level="2">helloworld</emo>',
        )


def test_service_contract_single_whole_sentence_uses_single_mode():
    contract = build_service_edit_contract(
        source_text="Hello beautiful world.",
        instruction_st="<emo type=\"sad\" alpha=\"0.9\">Hello beautiful world.</emo>",
    )

    assert contract.target_tagged_text == (
        "<emo type=\"sad\" alpha=\"0.9\">Hello beautiful world.</emo>"
    )
    assert contract.edit_tagged_text == "<target>Hello beautiful world.</target>"
    assert contract.synthesis_sub_type == "single"


def test_service_contract_accepts_emotion_desc_instruction():
    contract = build_service_edit_contract(
        source_text="Hello beautiful world.",
        instruction_st='Hello <emo desc="soft, tearful sadness" type_ref="sad">beautiful</emo> world.',
    )

    assert contract.target_tagged_text == (
        '<emo type="none">Hello </emo>'
        '<emo desc="soft, tearful sadness" type_ref="sad">beautiful</emo>'
        '<emo type="none"> world.</emo>'
    )
    assert contract.changed_segments == [
        {
            "segment_idx": 1,
            "segment_text": "beautiful",
            "source_tag": "none",
            "target_tag": "sad",
            "source_alpha": None,
            "target_alpha": None,
            "target_emotion_control_mode": "description",
            "target_emotion_desc": "soft, tearful sadness",
            "target_emotion_type_ref": "sad",
            "source_dur": None,
            "target_dur": None,
        }
    ]


def test_service_contract_carries_duration_targets():
    contract = build_service_edit_contract(
        source_text="Hello world.",
        instruction_st='<emo desc="soft, slower sadness" dur="1.5">Hello world.</emo>',
    )

    assert contract.target_tagged_text == (
        '<emo desc="soft, slower sadness" dur="1.5">Hello world.</emo>'
    )
    assert contract.changed_segments[0]["target_dur"] == 1.5
    assert contract.target_spans[0]["dur"] == 1.5


def test_service_contract_multi_span_uses_multi_emo_frequent_without_external_none():
    contract = build_service_edit_contract(
        source_text="Hello beautiful world today.",
        instruction_st=(
            "<emo type=\"happy\">Hello</emo> beautiful <emo type=\"sad\">world</emo> today."
        ),
    )

    assert contract.synthesis_sub_type == "multi_emo_frequent"
    assert contract.edit_tagged_text == (
        "<target>Hello</target> beautiful <target>world</target> today."
    )
    assert len(contract.target_spans) == 2
    assert len(contract.changed_segments) == 2
    assert contract.changed_segments[0]["target_tag"] == "happy"
    assert contract.changed_segments[1]["target_tag"] == "sad"
    assert "<emo type=\"none\"> beautiful </emo>" in contract.target_tagged_text


def test_service_contract_rejects_invalid_partial_instructions():
    bad_cases = [
        (
            "Hello beautiful world.",
            "Hello <emo type=\"happy\">different</emo> world.",
            "preserve source_text",
        ),
        (
            "Hello beautiful world.",
            "Hello <pitch, semitones=3>beautiful</pitch> world.",
            "Nested, malformed, or unsupported tags",
        ),
        (
            "Hello beautiful world.",
            "Hello <joy>beautiful</joy> world.",
            "Nested, malformed, or unsupported tags",
        ),
        (
            "Hello beautiful world.",
            "Hello <emo type=\"happy\"><emo type=\"sad\">beautiful</emo></emo> world.",
            "Nested, malformed, or unsupported tags",
        ),
        (
            "Hello beautiful world.",
            "Hello <emo type=\"none\">beautiful</emo> world.",
            "not allowed in instruction_st",
        ),
    ]

    for source_text, instruction_st, expected in bad_cases:
        with pytest.raises(ValueError, match=expected):
            build_service_edit_contract(
                source_text=source_text,
                instruction_st=instruction_st,
            )


def test_build_edit_tagged_text_and_diffs_marks_only_changed_segments():
    edit_tagged_text, diffs = synthesizer._build_edit_tagged_text_and_diffs(
        "<emo type=\"happy\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
        "<emo type=\"sad\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
    )

    assert edit_tagged_text == "<target>Hello</target> world again"
    assert len(diffs) == 1
    assert diffs[0]["segment_idx"] == 0
    assert diffs[0]["source_tag"] == "happy"
    assert diffs[0]["target_tag"] == "sad"
    assert diffs[0]["segment_text"] == "Hello"


def test_build_edit_tagged_text_and_diffs_marks_adjacent_changed_segments():
    edit_tagged_text, diffs = synthesizer._build_edit_tagged_text_and_diffs(
        "<emo type=\"happy\">Hello</emo><emo type=\"sad\"> there</emo><emo type=\"none\"> friend</emo>",
        "<emo type=\"sad\">Hello</emo><emo type=\"happy\"> there</emo><emo type=\"none\"> friend</emo>",
    )

    assert edit_tagged_text == "<target>Hello</target><target> there</target> friend"
    assert [d["segment_idx"] for d in diffs] == [0, 1]


def test_build_edit_tagged_text_and_diffs_preserves_description_diff_fields():
    edit_tagged_text, diffs = synthesizer._build_edit_tagged_text_and_diffs(
        '<emo desc="soft, tearful sadness" type_ref="sad">Hello</emo><emo type="none"> world</emo>',
        '<emo desc="bright relief with a smile" type_ref="happy">Hello</emo><emo type="none"> world</emo>',
    )

    assert edit_tagged_text == "<target>Hello</target> world"
    assert diffs == [
        {
            "segment_idx": 0,
            "segment_text": "Hello",
            "source_tag": "sad",
            "target_tag": "happy",
            "source_alpha": None,
            "target_alpha": None,
            "source_emotion_control_mode": "description",
            "target_emotion_control_mode": "description",
            "source_emotion_desc": "soft, tearful sadness",
            "target_emotion_desc": "bright relief with a smile",
            "source_emotion_type_ref": "sad",
            "target_emotion_type_ref": "happy",
            "source_dur": None,
            "target_dur": None,
        }
    ]


def test_build_edit_tagged_text_and_diffs_rejects_segment_text_mismatch():
    try:
        synthesizer._build_edit_tagged_text_and_diffs(
            "<emo type=\"happy\">Hello</emo><emo type=\"none\"> world</emo>",
            "<emo type=\"happy\">Hello</emo><emo type=\"none\"> WORLD</emo>",
        )
    except ValueError as exc:
        assert "Segment text mismatch" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched segment text")


def test_assemble_edit_pair_uses_edit_tagged_text(monkeypatch):
    monkeypatch.setattr(synthesizer, "_get_wav_duration", lambda _: 1.23)

    assembly_client = _FakeAssemblyClient()
    assembled = synthesizer._assemble_edit_pair(
        source_synth=synthesizer.SynthesizedAudio(
            text_id="1",
            tagged_text="<emo type=\"happy\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
            audio_bytes=b"RIFFsource",
            sample_rate=24000,
            duration_sec=1.0,
            segment_timestamps=[],
            synthesis_mode="single",
        ),
        target_synth=synthesizer.SynthesizedAudio(
            text_id="2",
            tagged_text="<emo type=\"sad\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
            audio_bytes=b"RIFFtarget",
            sample_rate=24000,
            duration_sec=1.1,
            segment_timestamps=[],
            synthesis_mode="single",
        ),
        nl_instruction="make first segment sad",
        st_instruction="metadata only",
        assembly_client=assembly_client,
        aligner_language="eng",
        aligner_backend="qwen3",
     assembly_f5_steps=32)

    assert assembly_client.calls == [
        {
            "source_audio_bytes": b"RIFFtarget",
            "target_audio_bytes": b"RIFFsource",
            "edit_tagged_text": "<target>Hello</target> world again",
            "language": "eng",
            "aligner_backend": "qwen3",
            "f5_steps": 32,
            "timeout": 300,
        }
    ]
    assert assembled.changed_segments[0]["segment_idx"] == 0
    assert assembled.assembly_method == "splice_and_smooth"
    assert assembled.st_instruction == "metadata only"


def test_assemble_edit_pair_refactor_preserves_old_base_donor_mapping(monkeypatch):
    monkeypatch.setattr(synthesizer, "_get_wav_duration", lambda _: 1.23)
    source_synth = synthesizer.SynthesizedAudio(
        text_id="1",
        tagged_text="<emo type=\"happy\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
        audio_bytes=b"RIFFsource",
        sample_rate=24000,
        duration_sec=1.0,
        segment_timestamps=[],
        synthesis_mode="single",
    )
    target_synth = synthesizer.SynthesizedAudio(
        text_id="2",
        tagged_text="<emo type=\"sad\">Hello</emo><emo type=\"none\"> world</emo><emo type=\"sad\"> again</emo>",
        audio_bytes=b"RIFFtarget",
        sample_rate=24000,
        duration_sec=1.1,
        segment_timestamps=[],
        synthesis_mode="single",
    )

    public_client = _FakeAssemblyClient(result=b"RIFFassembled")
    public_result = synthesizer._assemble_edit_pair(
        source_synth=source_synth,
        target_synth=target_synth,
        nl_instruction="make first segment sad",
        st_instruction="metadata only",
        assembly_client=public_client,
        aligner_language="eng",
        aligner_backend="qwen3",
     assembly_f5_steps=32)
    edit_tagged_text, changed_segments = synthesizer._build_edit_tagged_text_and_diffs(
        source_synth.tagged_text,
        target_synth.tagged_text,
    )
    direct_client = _FakeAssemblyClient(result=b"RIFFassembled")
    direct_audio = synthesizer._assemble_from_base_and_donor(
        base_audio_bytes=target_synth.audio_bytes,
        donor_audio_bytes=source_synth.audio_bytes,
        edit_tagged_text=edit_tagged_text,
        assembly_client=direct_client,
        aligner_language="eng",
        aligner_backend="qwen3",
     assembly_f5_steps=32)

    assert public_result.source_audio == direct_audio == b"RIFFassembled"
    assert public_result.target_audio == target_synth.audio_bytes
    assert public_result.changed_segments == changed_segments
    assert public_result.assembly_method == "splice_and_smooth"
    assert public_client.calls == direct_client.calls
    assert public_client.calls[0]["source_audio_bytes"] == target_synth.audio_bytes
    assert public_client.calls[0]["target_audio_bytes"] == source_synth.audio_bytes


def test_assemble_edit_pair_rejects_text_mismatch():
    with pytest.raises(ValueError, match=''):
        assembled = synthesizer._assemble_edit_pair(
            source_synth=synthesizer.SynthesizedAudio(
                text_id="1",
                tagged_text="<emo type=\"happy\">Hello</emo><emo type=\"none\"> world</emo>",
                audio_bytes=b"RIFFsource",
                sample_rate=24000,
                duration_sec=1.0,
                segment_timestamps=[],
                synthesis_mode="single",
            ),
            target_synth=synthesizer.SynthesizedAudio(
                text_id="2",
                tagged_text="<emo type=\"sad\">Hello</emo><emo type=\"none\"> WORLD</emo>",
                audio_bytes=b"RIFFtarget",
                sample_rate=24000,
                duration_sec=1.1,
                segment_timestamps=[],
                synthesis_mode="single",
            ),
            nl_instruction="mismatch",
            st_instruction="metadata only",
            assembly_client=_FakeAssemblyClient(),
         assembly_f5_steps=32)


def test_synthesize_multi_frequent_ted_uses_tagged_endpoint(monkeypatch):
    captured = {}

    class _FakeTedTTSClient:
        def health(self):
            return {"status": "healthy", "model_loaded": True}

        def synthesize_tagged(self, **kwargs):
            captured.update(kwargs)
            return b"RIFFted"

    monkeypatch.setattr(synthesizer, "_resample_wav", lambda wav, _: wav)
    monkeypatch.setattr(synthesizer, "_get_wav_duration", lambda _: 2.8)

    result = synthesizer._synthesize_multi_frequent_ted(
        tagged_text="<emo type=\"calm\" level=\"1\">Hello</emo><emo type=\"sad\" level=\"3\"> there</emo>",
        text_id="1",
        speaker_prompt_bytes=b"RIFFspeaker",
        ted_tts_client=_FakeTedTTSClient(),
        max_candidates=1,
    )

    assert result.audio_bytes == b"RIFFted"
    assert result.synthesis_mode == "multi_frequent_ted"
    assert captured["tagged_text"] == "<emo type=\"calm\" level=\"1\">Hello</emo><emo type=\"sad\" level=\"3\"> there</emo>"
    assert captured["spk_audio_bytes"] == b"RIFFspeaker"
    assert captured["method"] == "hmm"


def test_synthesize_multi_frequent_ted_duration_disables_fallback(monkeypatch):
    captured = {}

    class _FakeTedTTSClient:
        def health(self):
            return {"status": "healthy", "model_loaded": True}

        def synthesize_tagged(self, **kwargs):
            captured.update(kwargs)
            return b"RIFFted"

    monkeypatch.setattr(synthesizer, "_resample_wav", lambda wav, _: wav)
    monkeypatch.setattr(synthesizer, "_get_wav_duration", lambda _: 3.0)

    result = synthesizer._synthesize_multi_frequent_ted(
        tagged_text=(
            '<emo desc="soft, slower sadness" dur="1.2">Hello</emo>'
            '<emo desc="urgent anger, faster" dur="1.5"> there</emo>'
        ),
        text_id="1",
        speaker_prompt_bytes=b"RIFFspeaker",
        ted_tts_client=_FakeTedTTSClient(),
        max_candidates=1,
    )

    assert captured["max_mel_tokens"] == 850
    assert captured["duration_mode"] == "both"
    assert captured["eos_control_policy"] == "attention_gated_delayed"
    assert captured["eos_grace_seconds"] == 1.0
    assert result.duration_control == {
        "enabled": True,
        "backend": "ted",
        "duration_mode": "both",
        "eos_control_policy": "attention_gated_delayed",
        "eos_grace_seconds": 1.0,
        "target_seconds": [1.2, 1.5],
        "max_mel_tokens": 850,
        "max_seconds": 17.0,
    }
    assert [segment["dur"] for segment in result.segment_timestamps] == [1.2, 1.5]


def test_synthesize_multi_frequent_ted_duration_rejects_partial_dur(monkeypatch):
    class _FakeTedTTSClient:
        def health(self):
            return {"status": "healthy", "model_loaded": True}

    with pytest.raises(ValueError, match="dur must be present"):
        synthesizer._synthesize_multi_frequent_ted(
            tagged_text=(
                '<emo desc="soft, slower sadness" dur="1.2">Hello</emo>'
                '<emo desc="urgent anger, faster"> there</emo>'
            ),
            text_id="1",
            speaker_prompt_bytes=b"RIFFspeaker",
            ted_tts_client=_FakeTedTTSClient(),
            max_candidates=1,
        )


def test_synthesize_multi_frequent_ted_does_not_fallback_to_indextts(monkeypatch):
    captured = {}

    class _FakeTedTTSClient:
        def health(self):
            raise RuntimeError("offline")

    class _FakeIndexTTSClient:
        def health(self):
            return {"status": "healthy"}

        def synthesize(self, **kwargs):
            captured.update(kwargs)
            return b"RIFFindextts"

    monkeypatch.setattr(synthesizer, "_resample_wav", lambda wav, _: wav)
    monkeypatch.setattr(synthesizer, "_get_wav_duration", lambda _: 2.8)

    with pytest.raises(ServiceError, match="TED-TTS service unreachable"):
        synthesizer._synthesize_multi_frequent_ted(
            tagged_text=(
                '<emo desc="calm delivery" dur="1.0">Hello</emo>'
                '<emo desc="sad delivery" dur="1.0"> there</emo>'
            ),
            text_id="1",
            speaker_prompt_bytes=b"RIFFspeaker",
            ted_tts_client=_FakeTedTTSClient(),
            aligner_url="http://aligner.test",
            smoother_url="http://smoother.test",
            max_candidates=1,
        )

    assert captured == {}


def test_synthesize_tagged_text_group_routes_multi_frequent_to_ted(monkeypatch):
    calls = {}

    class _FakeTedTTSClient:
        def __init__(self, base_url):
            calls["ted_url"] = base_url

    class _FakeIndexTTSClient:
        def __init__(self, base_url):
            calls["indextts_url"] = base_url

    def fake_ted_synth(*args, **kwargs):
        calls["ted_synth"] = True
        return synthesizer.SynthesizedAudio(
            text_id=kwargs["text_id"],
            tagged_text=kwargs["tagged_text"],
            audio_bytes=b"RIFFted",
            sample_rate=24000,
            duration_sec=1.0,
            segment_timestamps=[],
            synthesis_mode="multi_frequent_ted",
        )

    monkeypatch.setattr(synthesizer, "TedTTSClient", _FakeTedTTSClient)
    monkeypatch.setattr(synthesizer, "IndexTTSClient", _FakeIndexTTSClient)
    monkeypatch.setattr(synthesizer, "_synthesize_multi_frequent_ted", fake_ted_synth)

    group = TaggedTextGroup(

        sub_type="multi_emo_frequent",
        plain_text="Hello there",
        speaker_prompt_audio=b"RIFFspeaker",
        speaker_prompt_text="prompt",
        tagged_texts={
            "1": (
                '<emo desc="happy delivery" type_ref="happy">Hello</emo>'
                '<emo desc="sad delivery" type_ref="sad"> there</emo>'
            )
        },
        edit_pairs=[],
        boundaries=[],
        metadata={"language": "en"},
    )

    group.metadata["synthesis_profile"] = synthesis_profile.build_synthesis_profile(
        synthesis_mode=group.sub_type, emotion_condition_mode="desc",
        ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"],
        ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"],
    )

    result = synthesizer.synthesize_tagged_text_group(
        group,
        indextts_url="http://index.test",
        ted_tts_url="http://ted.test",
        parallel=False,
     assembly_f5_steps=32)

    assert calls["ted_url"] == "http://ted.test"
    assert calls["indextts_url"] == "http://index.test"
    assert calls["ted_synth"] is True
    assert result.synthesized_audios["1"].synthesis_mode == "multi_frequent_ted"


def test_synthesize_tagged_text_group_routes_ted_without_fallback(monkeypatch):
    calls = {}

    class _FakeTedTTSClient:
        def __init__(self, base_url):
            calls["ted_url"] = base_url

    class _FakeIndexTTSClient:
        def __init__(self, base_url):
            calls["indextts_url"] = base_url

    def fake_ted_synth(*args, **kwargs):
        calls["kwargs"] = kwargs
        return synthesizer.SynthesizedAudio(
            text_id=kwargs["text_id"],
            tagged_text=kwargs["tagged_text"],
            audio_bytes=b"RIFFted",
            sample_rate=24000,
            duration_sec=1.0,
            segment_timestamps=[],
            synthesis_mode="multi_frequent_ted",
            duration_control={"enabled": False, "max_mel_tokens": kwargs["ted_max_mel_tokens"]},
        )

    monkeypatch.setattr(synthesizer, "TedTTSClient", _FakeTedTTSClient)
    monkeypatch.setattr(synthesizer, "IndexTTSClient", _FakeIndexTTSClient)
    monkeypatch.setattr(synthesizer, "_synthesize_multi_frequent_ted", fake_ted_synth)

    group = TaggedTextGroup(

        sub_type="multi_emo_frequent",
        plain_text="Hello there",
        speaker_prompt_audio=b"RIFFspeaker",
        speaker_prompt_text="prompt",
        tagged_texts={
            "1": (
                '<emo desc="soft sadness" type_ref="sad">Hello</emo>'
                '<emo desc="urgent anger" type_ref="angry"> there</emo>'
            )
        },
        edit_pairs=[],
        boundaries=[],
        metadata={"language": "en"},
    )

    group.metadata["synthesis_profile"] = synthesis_profile.build_synthesis_profile(
        synthesis_mode=group.sub_type, emotion_condition_mode="desc",
        ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"],
        ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"],
    )

    result = synthesizer.synthesize_tagged_text_group(
        group,
        indextts_url="http://index.test",
        ted_tts_url="http://ted.test",
        parallel=False,
     assembly_f5_steps=32)

    assert calls["ted_url"] == "http://ted.test"
    assert "fallback_indextts_client" not in calls["kwargs"]
    assert calls["kwargs"]["ted_max_mel_tokens"] == 1100
    assert result.synthesized_audios["1"].duration_control == {
        "enabled": False,
        "max_mel_tokens": 1100,
    }


def test_synthesize_tagged_text_group_rejects_untagged_spans():
    group = TaggedTextGroup(

        sub_type="single",
        plain_text="Hello world",
        speaker_prompt_audio=b"RIFFfakewav",
        speaker_prompt_text="prompt",
        tagged_texts={"1": "<emo type=\"happy\">Hello</emo> world"},
        edit_pairs=[],
        boundaries=[],
        metadata={},
    )

    group.metadata["synthesis_profile"] = synthesis_profile.build_synthesis_profile(
        synthesis_mode=group.sub_type, emotion_condition_mode="type_level",
        ted_estimated_duration_soft_limit_sec=PARAMETERS["ted_estimated_duration_soft_limit_sec"],
        ted_max_generated_duration_sec=PARAMETERS["ted_max_generated_duration_sec"],
    )

    try:
        synthesizer.synthesize_tagged_text_group(group, assembly_f5_steps=32)
    except ValueError as exc:
        assert "explicitly tag the full text" in str(exc)
    else:
        raise AssertionError("Expected fully-tagged validation error")


def test_synthesize_segment_forwards_emotion_desc_to_indextts(monkeypatch):
    class _FakeIndex:
        def __init__(self):
            self.calls = []

        def synthesize(self, **kwargs):
            self.calls.append(kwargs)
            return b"RIFFdesc"

    segment = parse_tags('<emo desc="soft, tearful sadness">Hello</emo>')[0]
    fake_index = _FakeIndex()
    monkeypatch.setattr(synthesizer, "_resample_wav", lambda wav, sr: wav)

    wav = synthesizer._synthesize_segment(
        segment,
        speaker_prompt_bytes=b"RIFFspeaker",
        speaker_prompt_text="prompt",
        indextts_client=fake_index,
    )

    assert wav == b"RIFFdesc"
    assert fake_index.calls[0]["emo_text"] == "soft, tearful sadness"
    assert "emo_vector" not in fake_index.calls[0]


def test_synthesize_segment_ted_uses_desc_and_duration(monkeypatch):
    class _FakeTed:
        def __init__(self):
            self.calls = []

        def synthesize_tagged(self, **kwargs):
            self.calls.append(kwargs)
            return b"RIFFted"

    segment = parse_tags('<emo desc="soft sadness" type_ref="sad">Hello</emo>')[0]
    fake_ted = _FakeTed()
    monkeypatch.setattr(synthesizer, "_resample_wav", lambda wav, sr: wav)

    wav = synthesizer._synthesize_segment(
        segment,
        speaker_prompt_bytes=b"RIFFspeaker",
        speaker_prompt_text="prompt",
        indextts_client=None,
        backend="ted",
        ted_tts_client=fake_ted,
        ted_max_mel_tokens=1100,
    )

    assert wav == b"RIFFted"
    assert fake_ted.calls[0]["tagged_text"] == (
        '<emo desc="soft sadness" type_ref="sad">Hello</emo>'
    )
    assert "duration_mode" not in fake_ted.calls[0]
    assert "eos_control_policy" not in fake_ted.calls[0]
    assert "eos_grace_seconds" not in fake_ted.calls[0]
    assert fake_ted.calls[0]["max_mel_tokens"] == 1100


def test_synthesize_multi_frequent_rejects_index_backend():
    with pytest.raises(ValueError, match=r"multi_emo_frequent \+ index"):
        synthesizer._synthesize_multi_frequent(
            tagged_text='<emo desc="happy" dur="1.0">Hello</emo>',
            text_id="1",
            speaker_prompt_bytes=b"RIFFspeaker",
            backend="index",
        )


def test_synthesize_multi_normal_forwards_qwen_aligner_to_smoother(monkeypatch):
    monkeypatch.setattr(
        synthesizer,
        "_synthesize_segment",
        lambda *args, **kwargs: b"RIFFsegment",
    )
    monkeypatch.setattr(
        synthesizer,
        "_wav_to_numpy",
        lambda wav: (synthesizer.np.zeros(24000, dtype=synthesizer.np.float32), 24000),
    )
    monkeypatch.setattr(synthesizer, "_numpy_to_wav", lambda audio, sr: b"RIFFconcat")
    smoother_client = _FakeSmootherClient(result=b"RIFFsmoothed")

    result = synthesizer._synthesize_multi_normal(
        tagged_text="<emo type=\"happy\">Hello</emo><emo type=\"sad\"> world</emo>",
        text_id="1",
        speaker_prompt_bytes=b"RIFFspeaker",
        speaker_prompt_text="prompt",
        boundaries=[],
        indextts_client=None,
        smoother_client=smoother_client,
        aligner_language="eng",
        aligner_backend="qwen3",
        max_candidates=1,
    )

    assert result.audio_bytes == b"RIFFsmoothed"
    assert len(smoother_client.calls) == 1
    points = smoother_client.calls[0]["points"]
    assert len(points) == 1
    assert points[0]["regen_margin_sec"] == 0.2
    assert smoother_client.calls[0]["aligner_backend"] == "qwen3"
    assert smoother_client.calls[0]["language"] == "eng"


def _patch_emotion_service_core(monkeypatch):
    from speech_edit.pipelines.emotion.service import api_service

    captured = {}
    synthetic_audio = _service_wav()
    edited_audio = _service_wav()

    class FakeAssemblyClient:
        def __init__(self):
            self.calls = []

        def assemble(self, **kwargs):
            self.calls.append(kwargs)
            return edited_audio

    fake_assembly = FakeAssemblyClient()

    def fake_synthesize(group, **kwargs):
        import inspect
        inspect.signature(synthesizer.synthesize_tagged_text_group).bind(group, **kwargs)
        captured["group"] = group
        captured["synthesize_kwargs"] = kwargs
        return synthesizer.SynthesisResult(
            synthesized_audios={
                "target": synthesizer.SynthesizedAudio(
                    text_id="target",
                    tagged_text=group.tagged_texts["target"],
                    audio_bytes=synthetic_audio,
                    sample_rate=24000,
                    duration_sec=1.0,
                    segment_timestamps=[],
                    synthesis_mode=group.sub_type,
                )
            },
            edit_pairs=[],
            timing={"fake": 0.01},
        )

    monkeypatch.setattr(api_service, "synthesize_tagged_text_group", fake_synthesize)
    monkeypatch.setattr(
        api_service,
        "_make_audio_assembly_client",
        lambda: fake_assembly,
    )
    monkeypatch.setattr(
        api_service,
        "_evaluate_service_output",
        lambda **kwargs: {
            "pair_evaluations": [
                {
                    "eval_results": {
                        "version": "v3",
                        "emotion_acoustic_effect_v1": {
                            "version": "v1",
                            "status": "ok",
                            "score": 0.8,
                            "regions": [],
                        },
                    }
                }
            ]
        },
    )
    captured["assembly_client"] = fake_assembly
    captured["synthetic_audio"] = synthetic_audio
    captured["edited_audio"] = edited_audio
    return api_service, captured


def test_emotion_api_health_shallow_and_deep(monkeypatch):
    from fastapi.testclient import TestClient
    from speech_edit.pipelines.emotion.service import api_service

    class FakeHealthClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def health(self):
            return {"status": "healthy", "base_url": self.base_url}

    for name in [
        "IndexTTSClient",
        "TedTTSClient",
        "AlignmentClient",
        "BoundaryPauseSmootherClient",
        "AudioAssemblyClient",
    ]:
        monkeypatch.setattr(api_service, name, FakeHealthClient)

    client = TestClient(api_service.app)
    shallow = client.get("/health")
    assert shallow.status_code == 200
    assert shallow.json()["status"] == "healthy"

    deep = client.get("/health?deep=true")
    assert deep.status_code == 200
    data = deep.json()
    assert data["status"] == "healthy"
    assert set(data["dependencies"]) == {
        "indextts",
        "ted_tts",
        "aligner",
        "smoother",
        "assembly",
    }


def test_emotion_api_deep_health_degrades_on_not_ready_payload(monkeypatch):
    from fastapi.testclient import TestClient
    from speech_edit.pipelines.emotion.service import api_service

    class FakeHealthyClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def health(self):
            return {"status": "healthy", "base_url": self.base_url}

    class FakeNotReadyClient(FakeHealthyClient):
        def health(self):
            return {"status": "not_ready", "model_loaded": False}

    for name in [
        "IndexTTSClient",
        "TedTTSClient",
        "AlignmentClient",
        "BoundaryPauseSmootherClient",
        "AudioAssemblyClient",
    ]:
        monkeypatch.setattr(api_service, name, FakeHealthyClient)
    monkeypatch.setattr(api_service, "IndexTTSClient", FakeNotReadyClient)

    response = TestClient(api_service.app).get("/health?deep=true")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "degraded"
    assert data["dependencies"]["indextts"]["status"] == "not_ready"


def test_emotion_api_eval_health_has_no_classifier_dependencies(monkeypatch):
    from fastapi.testclient import TestClient
    from speech_edit.pipelines.emotion.service import api_service

    class FakeHealthClient:
        def __init__(self, base_url):
            self.base_url = base_url

        def health(self):
            return {"status": "healthy"}

    for name in (
        "IndexTTSClient",
        "TedTTSClient",
        "AlignmentClient",
        "BoundaryPauseSmootherClient",
        "AudioAssemblyClient",
        "ASRClient",
        "WERClient",
        "UTMOSClient",
        "SpkSimClient",
        "WDTWClient",
    ):
        monkeypatch.setattr(api_service, name, FakeHealthClient)
    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(init_evaluate_services=True),
    )

    response = TestClient(api_service.app).get("/health?deep=true")

    assert response.status_code == 200
    dependencies = response.json()["dependencies"]
    assert {"asr", "wer", "utmos", "spk_sim", "wdtw"}.issubset(dependencies)
    assert "clsp" not in dependencies
    assert "whisper_ser" not in dependencies


def test_emotion_api_execute_generation_only_and_persist(monkeypatch, tmp_path: Path):
    from fastapi.testclient import TestClient

    api_service, captured = _patch_emotion_service_core(monkeypatch)
    source_audio = _service_wav()

    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(source_audio).decode("utf-8")},
            "source_text": "Hello beautiful world today.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful world</emo> today.",
            "evaluate": False,
            "persist": True,
            "output_dir": str(tmp_path),
            "sample_id": "svc_emotion",
            "return_synthetic_target_audio": True,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["status"] == "ok"
    assert data["pipeline_type"] == "emotion_style_edit"
    assert data["sample_id"] == "svc_emotion"
    assert data["target_tagged_text"] == (
        "<emo type=\"none\">Hello </emo>"
        "<emo type=\"happy\" alpha=\"0.93\">beautiful world</emo>"
        "<emo type=\"none\"> today.</emo>"
    )
    assert data["edit_tagged_text"] == "Hello <target>beautiful world</target> today."
    assert len(data["changed_segments"]) == 1
    decoded_audio = base64.b64decode(data["audio"]["data"])
    decoded_synthetic = base64.b64decode(data["synthetic_target_audio"]["data"])
    synthesizer._wav_to_numpy(decoded_audio)
    synthesizer._wav_to_numpy(decoded_synthetic)
    assert data["eval_results"] == {}
    assert data["metadata"]["language"] == "eng"
    assert data["metadata"]["language_source"] == "inferred"
    assert data["metadata"]["synthesis"]["mode"] == "multi_emo_normal"

    group = captured["group"]
    assert group.sub_type == "multi_emo_normal"
    assert group.edit_pairs == []
    assert group.speaker_prompt_audio == source_audio
    assert captured["synthesize_kwargs"]["aligner_backend"] == "qwen3"
    assert captured["assembly_client"].calls == [
        {
            "source_audio_bytes": source_audio,
            "target_audio_bytes": captured["synthetic_audio"],
            "edit_tagged_text": "Hello <target>beautiful world</target> today.",
            "language": "eng",
            "aligner_backend": "qwen3",
            "f5_steps": 32,
            "timeout": 300,
        }
    ]

    sample_dir = tmp_path / "svc_emotion"
    assert (sample_dir / "source.wav").read_bytes() == source_audio
    assert (sample_dir / "synthetic_target.wav").read_bytes() == captured["synthetic_audio"]
    assert (sample_dir / "edited.wav").read_bytes() == captured["edited_audio"]
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["assembly"]["method"] == "source_base_synthetic_target_donor"
    assert metadata["language"] == "eng"
    assert metadata["language_source"] == "inferred"
    assert metadata["synthesis"]["mode"] == "multi_emo_normal"


def test_emotion_api_execute_accepts_description_mode(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, captured = _patch_emotion_service_core(monkeypatch)
    source_audio = _service_wav()

    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(source_audio).decode("utf-8")},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo desc="soft, tearful sadness" type_ref="sad">beautiful</emo> world.',
            "emotion_condition_mode": "desc",
            "evaluate": False,
        },
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["changed_segments"][0]["target_emotion_control_mode"] == "description"
    assert data["changed_segments"][0]["target_emotion_desc"] == "soft, tearful sadness"
    assert data["changed_segments"][0]["target_emotion_type_ref"] == "sad"
    assert data["metadata"]["synthesis_backend"] == "index"
    assert data["metadata"]["emotion_condition_mode"] == "desc"
    assert captured["group"].metadata["emotion_control_mode"] == "description"
    assert (
        captured["group"].tagged_texts["target"]
        == '<emo type="none">Hello </emo>'
        '<emo desc="soft, tearful sadness" type_ref="sad">beautiful</emo>'
        '<emo type="none"> world.</emo>'
    )


def test_emotion_api_execute_rejects_pipeline_service_duration_mode(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, captured = _patch_emotion_service_core(monkeypatch)
    source_audio = _service_wav()

    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(source_audio).decode("utf-8")},
            "source_text": "Hello beautiful world.",
            "instruction_st": '<emo desc="soft, slower sadness" type_ref="sad" dur="1.6">Hello beautiful world.</emo>',
            "evaluate": False,
        },
    )

    assert response.status_code == 400
    assert "Duration tags are not supported" in response.json()["detail"]


def test_emotion_api_does_not_autofill_partial_target_durations(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, _captured = _patch_emotion_service_core(monkeypatch)
    source_audio = _service_wav()

    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(source_audio).decode("utf-8")},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo desc="soft, slower sadness" type_ref="sad" dur="1.6">beautiful</emo> world.',
            "evaluate": False,
        },
    )

    assert response.status_code == 400
    assert "Duration tags are not supported" in response.json()["detail"]


def test_emotion_api_execute_with_eval_returns_payload(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, _captured = _patch_emotion_service_core(monkeypatch)
    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(init_evaluate_services=True),
    )

    response = TestClient(api_service.app).post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(_service_wav()).decode("utf-8")},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"sad\">beautiful</emo> world.",
            "language": "en",
            "evaluate": True,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["eval_results"] == {
        "pair_evaluations": [
            {
                "eval_results": {
                    "version": "v3",
                    "emotion_acoustic_effect_v1": {
                        "version": "v1",
                        "status": "ok",
                        "score": 0.8,
                        "regions": [],
                    },
                }
            }
        ]
    }


def test_emotion_service_eval_aligns_final_pair_before_acoustic_metric(
    monkeypatch,
):
    from speech_edit.pipelines.emotion.service import api_service

    source_audio = _service_wav()
    edited_audio = _service_wav()
    contract = build_service_edit_contract(
        source_text="Hello beautiful world.",
        instruction_st='Hello <emo type="sad">beautiful</emo> world.',
    )
    request = api_service.ExecuteRequest(
        audio=api_service.AudioPayload(
            data=base64.b64encode(source_audio).decode("utf-8")
        ),
        source_text="Hello beautiful world.",
        instruction_st='Hello <emo type="sad">beautiful</emo> world.',
    )
    calls = []

    def fake_align(**kwargs):
        calls.append(kwargs)
        pair = kwargs["edit_pairs"][0]
        pair.source_alignment = {
            "level": "word",
            "alignments": [{"char": "beautiful", "start": 0.2, "end": 0.6}],
        }
        pair.target_alignment = {
            "level": "word",
            "alignments": [{"char": "beautiful", "start": 0.2, "end": 0.6}],
        }
        pair.source_segment_timestamps = [
            {
                "text": "beautiful",
                "start_sec": 0.2,
                "end_sec": 0.6,
            }
        ]
        pair.target_segment_timestamps = [
            {
                "text": "beautiful",
                "start_sec": 0.2,
                "end_sec": 0.6,
            }
        ]

    def fake_evaluate(synthesis_result, **kwargs):
        assert kwargs["wer_url"] == api_service.RUNTIME_CONFIG.wer_url
        pair = synthesis_result.edit_pairs[0]
        assert pair.source_alignment["level"] == "word"
        assert pair.target_alignment["level"] == "word"
        assert pair.target_audio == edited_audio
        return {
            "pair_evaluations": [
                {
                    "eval_results": {
                        "version": "v3",
                        "emotion_acoustic_effect_v1": {
                            "version": "v1",
                            "status": "ok",
                            "score": 0.5,
                            "regions": [],
                        },
                    }
                }
            ]
        }

    monkeypatch.setattr(api_service, "_align_edit_pair_outputs", fake_align)
    monkeypatch.setattr(api_service, "evaluate_synthesis_result", fake_evaluate)

    result = api_service._evaluate_service_output(
        request=request,
        contract=contract,
        source_audio=source_audio,
        source_sample_rate=24000,
        aligner_backend="qwen3",
        aligner_language="eng",
        edited_audio=edited_audio,
        synthetic_target_audio=edited_audio,
        edited_duration=1.0,
        synthetic_duration=1.0,
    )

    assert len(calls) == 1
    assert calls[0]["aligner_backend"] == "qwen3"
    assert calls[0]["aligner_language"] == "eng"
    assert result["pair_evaluations"][0]["eval_results"]["version"] == "v3"


def test_emotion_api_rejects_invalid_requests(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, _captured = _patch_emotion_service_core(monkeypatch)
    client = TestClient(api_service.app)
    source_audio = base64.b64encode(_service_wav()).decode("utf-8")

    unknown_field = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
            "save_parquet": True,
        },
    )
    assert unknown_field.status_code == 422

    unknown_audio_field = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio, "codec": "pcm16"},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
        },
    )
    assert unknown_audio_field.status_code == 422

    bad_format = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio, "format": "mp3"},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
        },
    )
    assert bad_format.status_code == 400
    assert "only WAV audio is supported" in bad_format.json()["detail"]

    bad_audio = client.post(
        "/execute",
        json={
            "audio": {"data": base64.b64encode(b"not wav").decode("utf-8")},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
        },
    )
    assert bad_audio.status_code == 400
    assert "Invalid audio payload" in bad_audio.json()["detail"]

    pitch_tag = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <pitch, semitones=3>beautiful</pitch> world.",
            "language": "en",
        },
    )
    assert pitch_tag.status_code == 400

    no_edit = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"none\">beautiful</emo> world.",
            "language": "en",
        },
    )
    assert no_edit.status_code == 400
    assert "not allowed in instruction_st" in no_edit.json()["detail"]

    unsupported_language = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
            "language": "fr",
        },
    )
    assert unsupported_language.status_code == 400
    assert "only supports English/Chinese" in unsupported_language.json()["detail"]

    eval_without_stack = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
            "language": "en",
            "evaluate": True,
        },
    )
    assert eval_without_stack.status_code == 400
    assert "without evaluation models" in eval_without_stack.json()["detail"]

    old_backend_field = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo desc="soft sadness">beautiful</emo> world.',
            "synthesis_backend": "index",
        },
    )
    assert old_backend_field.status_code == 422

    old_duration_field = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo type="happy">beautiful</emo> world.',
            "duration_control_mode": "ted_infer",
        },
    )
    assert old_duration_field.status_code == 422

    desc_request_without_type_ref = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo desc="soft sadness">beautiful</emo> world.',
            "emotion_condition_mode": "desc",
        },
    )
    assert desc_request_without_type_ref.status_code == 400
    assert "type_ref" in desc_request_without_type_ref.json()["detail"]

    frequent_type_level = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "Hello beautiful world.",
            "instruction_st": 'Hello <emo type="happy">beautiful</emo> world.',
            "synthesis_mode": "multi_emo_frequent",
            "emotion_condition_mode": "type_level",
        },
    )
    assert frequent_type_level.status_code == 400
    assert "multi_emo_frequent + type_level" in frequent_type_level.json()["detail"]


def test_emotion_api_infers_language_and_rejects_unavailable_backend(monkeypatch):
    from fastapi.testclient import TestClient

    api_service, captured = _patch_emotion_service_core(monkeypatch)
    client = TestClient(api_service.app)
    source_audio = base64.b64encode(_service_wav()).decode("utf-8")
    payload = {
        "audio": {"data": source_audio},
        "source_text": "Hello beautiful world.",
        "instruction_st": "Hello <emo type=\"happy\">beautiful</emo> world.",
    }

    inferred_language = client.post("/execute", json=payload)
    assert inferred_language.status_code == 200, inferred_language.text
    assert captured["synthesize_kwargs"]["aligner_backend"] == "qwen3"
    assert captured["assembly_client"].calls[-1]["language"] == "eng"

    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(pipeline_parameters={**PARAMETERS, "aligner_backend": "ctc"}),
    )
    ctc_response = client.post("/execute", json=payload)
    assert ctc_response.status_code == 400
    assert "qwen3" in ctc_response.json()["detail"]

    monkeypatch.setattr(
        api_service,
        "RUNTIME_CONFIG",
        api_service.ServiceRuntimeConfig(pipeline_parameters={**PARAMETERS, "aligner_backend": "qwen3"}),
    )
    zh_response = client.post(
        "/execute",
        json={
            "audio": {"data": source_audio},
            "source_text": "你好世界。",
            "instruction_st": "你好<emo type=\"happy\">世界</emo>。",
        },
    )
    assert zh_response.status_code == 200, zh_response.text
    assert zh_response.json()["metadata"]["language"] == "zho"
    assert zh_response.json()["metadata"]["language_source"] == "inferred"
    assert captured["assembly_client"].calls[-1]["language"] == "zho"
