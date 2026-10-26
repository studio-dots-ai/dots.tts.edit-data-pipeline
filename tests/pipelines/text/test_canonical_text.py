from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/text_edit/defaults")

import pytest

from speech_edit.pipelines.text.canonical_text import (
    canonicalize_text,
    project_tagged_instruction_to_surface,
    validate_canonical_sentence_edit,
)
from speech_edit.pipelines.text.generation_plan import (
    select_generation_specs,
    select_op_count,
)
from speech_edit.data.text_markup import (
    TaggedInstructionError,
    edit_segments,
    render_source,
    render_target,
)


@pytest.mark.parametrize(
    "text,canonical",
    [
        ("  hello   world ", "hello world"),
        ("你好 世界", "你好世界"),
        ("hello ， world", "hello，world"),
        ("你好,  world", "你好,world"),
        ("mix 中文 text", "mix中文text"),
        ("a  b   c", "a b c"),
    ],
)
def test_canonicalize_text_whitespace_policy(text, canonical):
    assert canonicalize_text(text).text == canonical


def test_project_repeated_word_uses_tag_position_not_search():
    surface = "go go  go go"
    canonical_st = "go go <sub targ=\"went\">go</sub> go"

    st = project_tagged_instruction_to_surface(canonical_st, surface)

    assert render_source(st) == surface
    assert render_target(st) == "go go went go"
    edit = edit_segments(st)[0]
    assert edit.source_start == len("go go  ")
    assert edit.source == "go"


def test_project_chinese_and_punctuation_ignores_optional_spaces():
    surface = "你好 ， 世界"
    canonical_st = "你好，<sub targ=\"朋友\">世界</sub>"

    st = project_tagged_instruction_to_surface(canonical_st, surface)

    assert st == "你好 ， <sub targ=\"朋友\">世界</sub>"
    assert render_source(st) == surface
    assert render_target(st) == "你好，朋友"


def test_project_rejects_source_that_does_not_canonicalize_equal():
    with pytest.raises(TaggedInstructionError):
        project_tagged_instruction_to_surface(
            "hello <sub targ=\"kind\">missing</sub>",
            "hello brave",
        )


def test_sentence_validation_requires_one_full_source_sub():
    validate_canonical_sentence_edit(
        "<sub targ=\"new sentence\">old sentence</sub>",
        "old sentence",
    )
    with pytest.raises(TaggedInstructionError):
        validate_canonical_sentence_edit(
            "old <sub targ=\"line\">sentence</sub>",
            "old sentence",
        )


def test_generation_sampling_zeros_illegal_operation_counts(monkeypatch):
    monkeypatch.setattr("random.choices", lambda choices, weights, k: [choices[-1]])

    assert select_op_count("sentence", {1: 0.0, 2: 1.0, 3: 1.0}) == 1
    assert select_op_count("clause", {1: 0.0, 2: 1.0, 3: 1.0}) == 2
    assert select_op_count("arbitrary", {1: 0.0, 2: 0.0, 3: 1.0}) == 3


def test_default_op_count_weights_prefer_multi_op_large_scale():
    assert PARAMETERS["op_count_weights"] == {1: 2.0, 2: 3.0, 3: 5.0}


def test_generation_specs_use_default_generation_ids(monkeypatch):
    values = iter(["word", 1, "phrase", 2, "sentence", 1])

    def fake_choices(choices, weights, k):
        return [next(values)]

    monkeypatch.setattr("random.choices", fake_choices)

    specs = select_generation_specs(num_generations=3, span_level_weights=PARAMETERS["span_level_weights"], op_count_weights=PARAMETERS["op_count_weights"], require_del_ins_prob=PARAMETERS["require_del_ins_prob"])

    assert [spec.generation_id for spec in specs] == ["000", "001", "002"]
    assert [spec.span_level for spec in specs] == ["word", "phrase", "sentence"]
    assert [spec.op_count for spec in specs] == [1, 2, 1]


def test_generation_specs_require_del_ins_probability_marks_first_non_sentence():
    specs = select_generation_specs(
        num_generations=2,
        span_level_weights={"word": 1.0},
        op_count_weights={1: 1.0},
        require_del_ins_prob=1.0,
    )

    assert specs[0].require_del_ins is True
    assert specs[1].require_del_ins is False
    assert specs[0].to_prompt_dict()["require_del_ins"] is True


def test_generation_specs_require_del_ins_probability_ignores_sentence():
    specs = select_generation_specs(
        num_generations=1,
        span_level_weights={"sentence": 1.0},
        op_count_weights={1: 1.0},
        require_del_ins_prob=1.0,
    )

    assert specs[0].span_level == "sentence"
    assert specs[0].require_del_ins is False
    assert "require_del_ins" not in specs[0].to_prompt_dict()


def test_generation_specs_require_del_ins_probability_zero_disables_requirement():
    specs = select_generation_specs(
        num_generations=1,
        span_level_weights={"phrase": 1.0},
        op_count_weights={2: 1.0},
        require_del_ins_prob=0.0,
    )

    assert specs[0].require_del_ins is False
