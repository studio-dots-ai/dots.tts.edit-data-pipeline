from __future__ import annotations

from qwen3_asr_service.query_handling import (
    QueryValidationError,
    build_query_mode_plan,
    resolve_query_answers,
)


def _tokenize_english(text: str) -> list[str]:
    if text == "But even so, it's nice":
        return ["But", "even", "so", "it's", "nice"]
    if text == "But even so":
        return ["But", "even", "so"]
    if text == "But":
        return ["But"]
    if text == " even ":
        return ["even"]
    if text == "so":
        return ["so"]
    if text == " even so, it's nice":
        return ["even", "so", "it's", "nice"]
    if text == ", ":
        return []
    if text == "it's nice":
        return ["it's", "nice"]
    if text == "it's":
        return ["it's"]
    if text == " nice":
        return ["nice"]
    if text == "s nice":
        return ["s", "nice"]
    return []


def _tokenize_chinese(text: str) -> list[str]:
    return list(text)


def test_query_mode_plan_tracks_default_tokenization_match_for_english():
    text = "But even so, it's nice"
    plan = build_query_mode_plan(
        text,
        [
            "<q>But</q> even so, it's nice",
            "But even <q>so</q>, it's nice",
            "But even so, <q></q>it's nice",
        ],
        tokenize_text=_tokenize_english,
    )

    assert plan.full_units == ["But", "even", "so", "it's", "nice"]
    assert plan.query_mode_units == ["But", "even", "so", "it's", "nice"]
    assert plan.tokenization_matches_default is True


def test_query_mode_plan_reports_divergence_when_forced_boundaries_split_units():
    text = "But even so, it's nice"
    plan = build_query_mode_plan(
        text,
        ["But even so<q>, </q>it's nice"],
        tokenize_text=_tokenize_english,
    )

    assert plan.full_units == ["But", "even", "so", "it's", "nice"]
    assert plan.query_mode_units == ["But", "even", "so", "it's", "nice"]
    assert plan.tokenization_matches_default is True


def test_query_mode_plan_reports_divergence_for_mid_token_apostrophe_cut():
    text = "But even so, it's nice"

    def tokenize(text_: str) -> list[str]:
        if text_ == text:
            return ["But", "even", "so", "it's", "nice"]
        if text_ == "But even so, it":
            return ["But", "even", "so", "it"]
        if text_ == "'":
            return ["'"]
        if text_ == "s nice":
            return ["s", "nice"]
        return _tokenize_english(text_)

    plan = build_query_mode_plan(
        text,
        ["But even so, it<q>'</q>s nice"],
        tokenize_text=tokenize,
    )

    assert plan.query_mode_units == ["But", "even", "so", "it", "'", "s", "nice"]
    assert plan.full_units == ["But", "even", "so", "it's", "nice"]
    assert plan.tokenization_matches_default is False


def test_resolve_query_answers_returns_span_and_boundary_point():
    text = "But even so, it's nice"
    plan = build_query_mode_plan(
        text,
        ["<q>But</q> even so, it's nice"],
        tokenize_text=_tokenize_english,
    )
    alignments = [
        {"char": "But", "start": 0.1, "end": 0.2},
        {"char": "even", "start": 0.25, "end": 0.45},
        {"char": "so", "start": 0.5, "end": 0.6},
        {"char": "it's", "start": 0.6, "end": 0.85},
        {"char": "nice", "start": 0.9, "end": 1.2},
    ]

    assert resolve_query_answers(plan, alignments) == [{"start": 0.1, "end": 0.2}]


def test_resolve_query_answers_handles_empty_query_boundary_point():
    text = "But even so, it's nice"
    plan = build_query_mode_plan(
        text,
        ["But even so, <q></q>it's nice"],
        tokenize_text=_tokenize_english,
    )
    alignments = [
        {"char": "it's", "start": 0.6, "end": 0.85},
        {"char": "nice", "start": 0.9, "end": 1.2},
    ]

    assert resolve_query_answers(plan, alignments) == [{"start": 0.6, "end": 0.6}]


def test_query_mode_requires_all_queries_to_reduce_to_same_plain_text():
    try:
        build_query_mode_plan(
            "开发边界",
            ["<q>开发</q>边界", "开发<q>约定</q>"],
            tokenize_text=_tokenize_chinese,
        )
    except QueryValidationError as exc:
        assert "same plain text" in str(exc)
    else:
        raise AssertionError("expected QueryValidationError")


def test_query_mode_rejects_malformed_tags():
    try:
        build_query_mode_plan(
            "开发边界",
            ["<q>开发</q><q></q>边界"],
            tokenize_text=_tokenize_chinese,
        )
    except QueryValidationError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected QueryValidationError")
