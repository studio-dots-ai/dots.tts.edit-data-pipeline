"""Tests for pause_edit_pipeline.input_generator (no-service tests only).

These tests only exercise utility functions that don't depend on torch or
service clients.
"""

import io
import json

import numpy as np
import pytest
import soundfile as sf

from speech_edit.pipelines.pause.input_generator import (
    PauseProfileValidationError,
    PauseProfileTarget,
    _generate_profiles,
    _validate_profile,
    generate_profiles_with_retries,
    parse_llm_json,
    sample_pause_profile_targets,
    sample_duration_pool,
    _is_reasonable_normed_text,
    _normalize_plain_text,
)

def _wav_bytes(fill_value: float, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(
        buf,
        np.full(sr // 10, fill_value, dtype=np.float32),
        sr,
        format="WAV",
        subtype="PCM_16",
    )
    return buf.getvalue()


class TestSampleDurationPool:
    """Test duration pool sampling."""

    def test_returns_correct_count(self):
        pool = sample_duration_pool(3)
        assert len(pool) == 3

    def test_returns_fixed_level_mapping(self):
        assert sample_duration_pool(3) == [200, 500, 800]

    def test_single(self):
        pool = sample_duration_pool(1)
        assert pool == [200]


class TestSamplePauseProfileTargets:
    def test_weighted_sampling_is_deterministic(self):
        first = sample_pause_profile_targets(
            plain_text="One, two, three.",
            num_profiles=2,
            num_operation_weights={1: 0.0, 2: 1.0, 3: 0.0},
            boundary_type_weights={"punct": 1.0, "nonpunct": 0.0},
            pause_level_weights={1: 0.0, 2: 0.0, 3: 1.0},
            seed=123,
        )
        second = sample_pause_profile_targets(
            plain_text="One, two, three.",
            num_profiles=2,
            num_operation_weights={1: 0.0, 2: 1.0, 3: 0.0},
            boundary_type_weights={"punct": 1.0, "nonpunct": 0.0},
            pause_level_weights={1: 0.0, 2: 0.0, 3: 1.0},
            seed=123,
        )

        assert first == second
        assert [target.num_operations for target in first] == [2, 2]
        assert first[0].boundary_types == ["punct", "punct"]
        assert first[0].levels == [3, 3]

    def test_rejects_empty_weight_mass(self):
        with pytest.raises(ValueError, match="At least one weight"):
            sample_pause_profile_targets(
                plain_text="One, two, three.",
                num_profiles=1,
                num_operation_weights={1: 0.0, 2: 0.0, 3: 0.0},
            )

    @pytest.mark.parametrize("text", [
        "Going whitewater rafting is too scary.",
        "Rowing a boat on Sun Moon Lake is relaxing.",
    ])
    def test_terminal_punctuation_cannot_supply_punct_targets(self, text):
        targets = sample_pause_profile_targets(
            plain_text=text, num_profiles=40,
            num_operation_weights={1: 1, 2: 2, 3: 3},
            boundary_type_weights={"punct": 9, "nonpunct": 1},
            pause_level_weights={1: 1, 2: 1, 3: 1}, seed=7,
        )
        assert all(set(target.boundary_types) == {"nonpunct"} for target in targets)
        assert {target.num_operations for target in targets} == {1, 2, 3}

    @pytest.mark.parametrize("text, count", [
        ("One, two, it's one, two, you.", 3),
        ("My favorite's my train. What's your favorite toy, Susie?", 2),
        ("甲，乙，丙。", 2),
    ])
    def test_internal_punctuation_and_apostrophes(self, text, count):
        targets = sample_pause_profile_targets(
            plain_text=text, num_profiles=1,
            num_operation_weights={count: 1},
            boundary_type_weights={"punct": 1},
            pause_level_weights={1: 1}, seed=2,
        )
        assert targets[0].boundary_types == ["punct"] * count

    @pytest.mark.parametrize("text, count, boundary_type", [
        ("One, two.", 2, "punct"),
        ("it's", 1, "punct"),
        ("it's", 1, "nonpunct"),
        ("你hello好", 3, "nonpunct"),
        ("Hello.", 1, "punct"),
    ])
    def test_no_positive_feasible_combination_fails(self, text, count, boundary_type):
        # The mixed-script text has only two internal lexical boundaries.
        with pytest.raises(ValueError, match="feasible"):
            sample_pause_profile_targets(
                plain_text=text, num_profiles=1,
                num_operation_weights={count: 1},
                boundary_type_weights={boundary_type: 1},
                pause_level_weights={1: 1}, seed=2,
            )

    def test_joint_weights_are_conditioned_not_position_weighted(self, monkeypatch):
        from speech_edit.pipelines.pause import input_generator as module
        captured = []
        original = module._weighted_choice

        def capture(weights, rng):
            captured.append(dict(weights))
            return original(weights, rng)

        monkeypatch.setattr(module, "_weighted_choice", capture)
        sample_pause_profile_targets(
            plain_text="One two, three four.", num_profiles=1,
            num_operation_weights={1: 1, 2: 1, 3: 1},
            boundary_type_weights={"nonpunct": 3, "punct": 1},
            pause_level_weights={1: 1}, seed=2,
        )
        assert captured[0] == pytest.approx({
            ("nonpunct",): 0.75,
            ("punct",): 0.25,
            ("nonpunct", "nonpunct"): 0.5625,
            ("nonpunct", "punct"): 0.1875,
            ("punct", "nonpunct"): 0.1875,
            ("nonpunct", "punct", "nonpunct"): 0.140625,
        })

    @pytest.mark.parametrize("weight", [float("nan"), float("inf"), -1.0])
    def test_rejects_invalid_weights(self, weight):
        with pytest.raises(ValueError, match="finite and non-negative"):
            sample_pause_profile_targets(
                plain_text="One two.", num_profiles=1,
                num_operation_weights={1: weight},
                boundary_type_weights={"nonpunct": 1},
                pause_level_weights={1: 1},
            )

    def test_zero_weight_is_never_selected_at_rng_zero(self):
        from speech_edit.pipelines.pause.input_generator import _weighted_choice
        from unittest.mock import Mock
        assert _weighted_choice({1: 0, 2: 1, 3: 0}, Mock(random=lambda: 0.0)) == 2

    def test_cjk_and_mixed_words_have_distinct_boundaries(self):
        targets = sample_pause_profile_targets(
            plain_text="你好hello世界", num_profiles=1,
            num_operation_weights={3: 1},
            boundary_type_weights={"nonpunct": 1},
            pause_level_weights={2: 1}, seed=4,
        )
        assert targets[0].boundary_types == ["nonpunct"] * 3


class TestParseLlmJson:
    """Test LLM JSON parsing with error recovery."""

    def test_clean_json(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_fenced(self):
        result = parse_llm_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_trailing_comma(self):
        result = parse_llm_json('{"key": "value",}')
        assert result["key"] == "value"

    def test_truncated_json(self):
        result = parse_llm_json('{"key": "value"')
        assert result["key"] == "value"


class TestNormalizePlainText:
    """Test plain text normalization."""

    def test_strips_quotes(self):
        assert _normalize_plain_text('"Hello world"') == "Hello world"

    def test_normalizes_smart_quotes(self):
        result = _normalize_plain_text("\u201cHello\u201d")
        assert "\u201c" not in result

    def test_collapses_whitespace(self):
        assert _normalize_plain_text("Hello   world") == "Hello world"


class TestIsReasonableNormedText:
    """Test normed text drift detection."""

    def test_identical(self):
        assert _is_reasonable_normed_text("Hello world.", "Hello world.")

    def test_minor_normalization(self):
        assert _is_reasonable_normed_text('"Hello world"', "Hello world")

    def test_expansion_allowed(self):
        assert _is_reasonable_normed_text("Mr Smith", "Mister Smith")

    def test_rewrite_rejected(self):
        assert not _is_reasonable_normed_text(
            "Hello world.", "Goodbye everyone."
        )

    def test_empty_rejected(self):
        assert not _is_reasonable_normed_text("Hello world.", "")


class TestValidateProfile:
    def test_rejects_english_word_changes_in_mixed_script_text(self):
        profile = _validate_profile(
            {"profile_id": 1,
             "operation": '我们 are apart<pause act="ins" level="1"/> of it.'},
            "我们 are a part of it.", [200, 500, 800], 1,
        )
        assert profile is None

    def test_instruction_metadata_comes_from_executed_operation(self):
        operation = 'One<pause act="ins" level="1"/> two three.'
        profile = _validate_profile(
            {"profile_id": 1, "operation": operation,
             "st_inst": {"increase_pause": ['One two<pause act="ins" level="3"/> three.']},
             "nl_inst": {"increase_pause": ["An unrelated instruction."]}},
            "One two three.", [200, 500, 800], 1,
        )
        assert profile is not None
        assert profile.st_inst["increase_pause"] == [operation]
        assert profile.st_inst["decrease_pause"] == [operation.replace('act="ins"', 'act="red"')]
        assert profile.tags[0].char_idx == 3
        assert profile.tags[0].level == 1
        assert profile.nl_inst["increase_pause"] != ["An unrelated instruction."]

    def test_accepts_multiple_pause_positions_up_to_limit(self):
        normed = "She walked slowly through the beautiful garden."
        operation = (
            'She walked<pause act="ins" level="1"/> slowly through the '
            'beautiful<pause act="ins" level="2"/> garden.'
        )
        profile = _validate_profile(
            {
                "profile_id": 1,
                "operation": operation,
                "st_inst": {
                    "increase_pause": [operation],
                    "decrease_pause": [
                        'She walked<pause act="red" level="1"/> slowly through the '
                        'beautiful<pause act="red" level="2"/> garden.'
                    ],
                },
                "nl_inst": {
                    "increase_pause": ["Add pauses."],
                    "decrease_pause": ["Reduce pauses."],
                },
            },
            normed,
            [200, 500, 800],
            3,
        )

        assert profile is not None
        assert len(profile.tags) == 2

    def test_rejects_too_many_pause_positions(self):
        normed = "a b c d e"
        operation = (
            'a<pause act="ins" level="1"/> b<pause act="ins" level="1"/> '
            'c<pause act="ins" level="2"/> d<pause act="ins" level="3"/> e'
        )
        profile = _validate_profile(
            {
                "profile_id": 1,
                "operation": operation,
                "st_inst": {
                    "increase_pause": [operation],
                    "decrease_pause": [
                        'a<pause act="red" level="1"/> b<pause act="red" level="1"/> '
                        'c<pause act="red" level="2"/> d<pause act="red" level="3"/> e'
                    ],
                },
                "nl_inst": {
                    "increase_pause": ["Add pauses."],
                    "decrease_pause": ["Reduce pauses."],
                },
            },
            normed,
            [200, 500, 800],
            3,
        )

        assert profile is None

    def test_rejects_profile_that_violates_target_slots(self):
        normed = "hello, world again"
        operation = 'hello,<pause act="ins" level="2"/> world again'
        profile = _validate_profile(
            {
                "profile_id": 1,
                "operation": operation,
            },
            normed,
            [200, 500, 800],
            1,
            profile_target=PauseProfileTarget(
                profile_id=1,
                boundary_types=["nonpunct"],
                levels=[2],
            ),
        )

        assert profile is None


class TestDetailedValidationErrors:
    @pytest.mark.parametrize(
        "response",
        [{}, {"normed_text": None}, {"normed_text": ""},
         {"normed_text": "   "}, {"normed_text": "Goodbye everyone."}],
    )
    def test_generate_profiles_requires_valid_reference_text(self, response):
        class _FakeLlm:
            def generate(self, *args, **kwargs):
                return json.dumps(response)

        with pytest.raises(PauseProfileValidationError, match="normed_text"):
            _generate_profiles(
                plain_text="Hello world.",
                duration_pool=[200, 500, 800],
                profile_targets=[PauseProfileTarget(
                    profile_id=1, boundary_types=["nonpunct"], levels=[1],
                )],
                llm_client=_FakeLlm(),
            )

    def test_generate_profiles_maps_cjk_spacing_to_reference_coordinates(self):
        class _FakeLlm:
            def generate(self, *args, **kwargs):
                return (
                    '{"normed_text": "大家都说越努力,越幸运这句话不适用于钉钉 app.", '
                    '"profiles": [{'
                    '"profile_id": 1, '
                    '"operation": "大家都说<pause act=\\"ins\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 app.", '
                    '"st_inst": {'
                    '  "increase_pause": ["大家都说<pause act=\\"ins\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 app."], '
                    '  "decrease_pause": ["大家都说<pause act=\\"red\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 app."]'
                    '}, '
                    '"nl_inst": {'
                    '  "increase_pause": ["加停顿。"], '
                    '  "decrease_pause": ["减停顿。"]'
                    '}'
                    '}]}'
                )

        profiles, normed_text = _generate_profiles(
            plain_text="大家都说越努力,越幸运这句话不适用于钉钉 app.",
            duration_pool=[200, 500, 800],
            profile_targets=[
                PauseProfileTarget(
                    profile_id=1,
                    boundary_types=["nonpunct"],
                    levels=[1],
                )
            ],
            llm_client=_FakeLlm(),
        )

        assert normed_text == "大家都说越努力,越幸运这句话不适用于钉钉 app."
        assert profiles[0].operation == "大家都说<pause act=\"ins\" level=\"1\"/> 越努力,越幸运这句话不适用于钉钉 app."
        assert profiles[0].tags[0].char_idx == 4
        assert profiles[0].tags[0].left_context == "大家都说"
        assert profiles[0].tags[0].right_context.startswith("越努力")
        assert profiles[0].st_inst["increase_pause"][0] == profiles[0].operation

    def test_generate_profiles_with_retries_surfaces_validation_details(self):
        class _FakeLlm:
            def generate(self, *args, **kwargs):
                return (
                    '{"normed_text": "大家都说越努力,越幸运这句话不适用于钉钉 app.", '
                    '"profiles": [{'
                    '"profile_id": 1, '
                    '"operation": "大家都说<pause act=\\"ins\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 apq.", '
                    '"st_inst": {'
                    '  "increase_pause": ["大家都说<pause act=\\"ins\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 apq."], '
                    '  "decrease_pause": ["大家都说<pause act=\\"red\\" level=\\"1\\"/> 越努力,越幸运这句话不适用于钉钉 apq."]'
                    '}, '
                    '"nl_inst": {'
                    '  "increase_pause": ["加停顿。"], '
                    '  "decrease_pause": ["减停顿。"]'
                    '}'
                    '}]}'
                )

        with pytest.raises(PauseProfileValidationError) as exc_info:
            generate_profiles_with_retries(
                plain_text="大家都说越努力,越幸运这句话不适用于钉钉 app.",
                duration_pool=[200, 500, 800],
                profile_targets=[
                    PauseProfileTarget(
                        profile_id=1,
                        boundary_types=["nonpunct"],
                        levels=[1],
                    )
                ],
                llm_client=_FakeLlm(),
                max_retries=2,
            )

        assert "Validation details:" in str(exc_info.value)
        assert "tag validation failed" in str(exc_info.value) or "text mismatch" in str(exc_info.value)
