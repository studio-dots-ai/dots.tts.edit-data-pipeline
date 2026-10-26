import re

import numpy as np

from speech_edit.pipelines.text.audio_timeline import (
    build_target_condition_waveform,
)
from speech_edit.pipelines.text.span_planner import (
    alignment_answers_to_spans,
    build_alignment_queries_from_st,
    estimate_speech_rate,
    plan_spans_from_answers,
)
from speech_edit.data.text_markup import (
    edit_segments,
    render_source,
    render_target,
    reverse_tagged_instruction,
)


def _strip_query_tags(text: str) -> str:
    return re.sub(r"</?q>", "", text)


def test_estimate_duration_substitution_matches_letter_rate_semantics():
    source = "hello brave world"
    st = "hello <sub targ=\"kind\">brave</sub> world"
    edits = edit_segments(st)
    queries = build_alignment_queries_from_st(st)

    plan = plan_spans_from_answers(
        source_text=source,
        target_text="hello kind world",
        edit_segments=edits,
        queries=queries,
        answers=[{"start": 1.0, "end": 2.0}],
        source_duration=4.0,
        expand_radius=0.0,
        merge_gap=0.0,
    )

    speech_rate = estimate_speech_rate(source, 4.0)
    expected_fix = 4 / speech_rate
    assert plan.source_spans == [[1.0, 2.0]]
    assert plan.target_spans == [[1.0, round(1.0 + expected_fix, 5)]]
    assert plan.target_duration == round(1.0 + expected_fix + 2.0, 5)


def test_insertion_uses_anchor_query_and_interunit_duration():
    source = "hello world"
    st = "hello <ins>wide </ins>world"
    edits = edit_segments(st)
    queries = build_alignment_queries_from_st(st)

    assert queries[0].anchor_mode == "point"
    assert queries[0].query == "hello <q></q>world"

    plan = plan_spans_from_answers(
        source_text=source,
        target_text="hello wide world",
        edit_segments=edits,
        queries=queries,
        answers=[{"start": 0.52, "end": 0.52}],
        source_duration=2.0,
        expand_radius=0.0,
        merge_gap=0.0,
    )

    assert plan.source_spans == [[0.52, 0.52]]
    assert plan.target_spans[0][0] == 0.52
    assert plan.target_spans[0][1] > plan.source_spans[0][1]


def test_generated_alignment_uses_reverse_source_anchor_for_deleted_forward_text():
    st = "hello <del>wide </del>world"
    reverse_st = reverse_tagged_instruction(st)
    queries = build_alignment_queries_from_st(reverse_st)

    assert render_source(reverse_st) == render_target(st)
    assert queries[0].anchor_mode == "point"
    assert queries[0].query == "hello<q></q> world"
    spans = alignment_answers_to_spans(
        audio_duration=1.0,
        queries=queries,
        answers=[{"start": 0.4, "end": 0.45}],
    )
    assert spans == [[0.4, 0.4]]


def test_generated_alignment_queries_match_normalized_target_surface():
    st = "No, <ins> actually</ins>it's not fine."
    target_text = render_target(st)
    reverse_st = reverse_tagged_instruction(st)
    queries = build_alignment_queries_from_st(reverse_st)

    assert target_text == "No, actually it's not fine."
    assert render_source(reverse_st) == target_text
    assert len(queries) == 1
    assert queries[0].anchor_mode == "span"
    assert queries[0].query.count("<q>") == 1
    assert queries[0].query.count("</q>") == 1
    assert _strip_query_tags(queries[0].query) == target_text
    assert queries[0].text == "actually"


def test_generated_alignment_queries_match_surface_with_connector_edit():
    st = "No, <ins> actually</ins>don<ins>'</ins>t stop"
    target_text = render_target(st)
    reverse_st = reverse_tagged_instruction(st)
    queries = build_alignment_queries_from_st(reverse_st)

    assert target_text == "No, actually don't stop"
    assert render_source(reverse_st) == target_text
    assert len(queries) == 2
    assert [query.query.count("<q>") for query in queries] == [1, 1]
    assert [query.query.count("</q>") for query in queries] == [1, 1]
    assert [_strip_query_tags(query.query) for query in queries] == [
        target_text,
        target_text,
    ]
    assert [query.text for query in queries] == ["actually", "'"]


def test_alignment_queries_are_direction_transparent_for_complex_tagged_text():
    cases = [
        "hello <sub targ=\"kind\">brave</sub> world",
        "hello <ins>wide </ins>world",
        "hello <del>very </del>wide world",
        "a <sub targ=\"x\">one</sub> b <del>two</del> c <ins>three</ins>",
        "请<ins>认真</ins>讲话，<sub targ=\"朋友\">同学</sub><del>们</del>",
        "room<ins>，</ins>课文<sub targ=\"是\">sold out.</sub>",
        "那么你到外面去旅游,住酒店呢,也要预先订酒店房间<sub targ=\"making a reservation\">book a hotel room in advance book, a hotel room in advance</sub><ins>，</ins>课文的标题<sub targ=\"是\">sold out.</sub>",
        "No, <ins> actually</ins>don<ins>'</ins>t stop",
        "mix <sub targ=\"中文\">English</sub> here and <ins>now</ins>.",
        "score 3 - 2 <ins>today</ins>, then <sub targ=\"tomorrow\">later</sub>",
        "<ins>Start </ins>middle <del>old </del>end",
        "state<ins>-of-the-art</ins> model <sub targ=\"works\">failed</sub>",
    ]

    for st in cases:
        forward_plain = render_source(st)
        forward_queries = build_alignment_queries_from_st(st)
        forward_edits = edit_segments(st)
        assert len(forward_queries) == len(forward_edits)
        assert all(_strip_query_tags(query.query) == forward_plain for query in forward_queries)
        assert all(
            query.text == forward_plain[query.char_start : query.char_end]
            for query in forward_queries
        )

        reverse_st = reverse_tagged_instruction(st)
        reverse_plain = render_source(reverse_st)
        assert reverse_plain == render_target(st)
        reverse_queries = build_alignment_queries_from_st(reverse_st)
        assert len(reverse_queries) == len(forward_edits)
        assert all(_strip_query_tags(query.query) == reverse_plain for query in reverse_queries)


def test_reverse_alignment_queries_keep_standalone_punctuation_edit():
    st = (
        "那么你到外面去旅游,住酒店呢,也要预先订酒店房间"
        "<sub targ=\"making a reservation\">book a hotel room in advance book, a hotel room in advance</sub>"
        "<ins>，</ins>课文的标题<sub targ=\"是\">sold out.</sub>"
    )

    reverse_st = reverse_tagged_instruction(st)
    queries = build_alignment_queries_from_st(reverse_st)

    assert render_source(reverse_st) == render_target(st)
    assert queries[1].text == "，"
    assert "<q>，</q>" in queries[1].query


def test_expand_and_merge_adjusts_fix_duration_and_source_spans():
    source = "a b c"
    st = "<sub targ=\"x\">a</sub> <sub targ=\"y\">b</sub> c"
    edits = edit_segments(st)
    queries = build_alignment_queries_from_st(st)

    plan = plan_spans_from_answers(
        source_text=source,
        target_text="x y c",
        edit_segments=edits,
        queries=queries,
        answers=[{"start": 0.1, "end": 0.2}, {"start": 0.25, "end": 0.35}],
        source_duration=1.0,
        expand_radius=0.05,
        merge_gap=0.2,
    )

    assert plan.source_spans == [[0.05, 0.4]]
    assert len(plan.target_spans) == 1
    assert plan.target_spans[0][0] == 0.05


def test_expand_duration_does_not_apply_min_edit_clamp():
    source = "a b"
    st = "<del>a</del> b"
    edits = edit_segments(st)
    queries = build_alignment_queries_from_st(st)

    plan = plan_spans_from_answers(
        source_text=source,
        target_text=" b",
        edit_segments=edits,
        queries=queries,
        answers=[{"start": 0.0, "end": 0.0}],
        source_duration=1.0,
        expand_radius=0.0,
        merge_gap=0.0,
    )

    assert plan.source_spans == [[0.0, 0.01]]
    assert plan.target_spans == [[0.0, 0.0]]
    assert plan.target_duration == 0.99


def test_target_condition_waveform_copies_unedited_regions_on_target_timeline():
    source = np.arange(10, dtype=np.float32)
    target = build_target_condition_waveform(
        source,
        10,
        source_spans=[[0.2, 0.4]],
        target_spans=[[0.2, 0.6]],
        target_duration=1.2,
    )

    assert len(target) == 12
    np.testing.assert_array_equal(target[:2], source[:2])
    np.testing.assert_array_equal(target[2:6], np.zeros(4, dtype=np.float32))
    np.testing.assert_array_equal(target[6:12], source[4:10])
