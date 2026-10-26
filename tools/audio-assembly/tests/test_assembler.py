from __future__ import annotations

import pytest
import importlib.util
import sys

from pathlib import Path


def _load_assembler_module():
    module_path = Path(__file__).resolve().parents[1] / "assembler.py"
    spec = importlib.util.spec_from_file_location(
        "audio_assembly_assembler", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


assembler = _load_assembler_module()


def test_parse_edit_tagged_text_merges_adjacent_target_spans():
    base_text, pieces = assembler.parse_edit_tagged_text(
        "<target>Mon</target><target>day,</target> hello"
    )

    assert base_text == "Monday, hello"
    assert len(pieces) == 2
    assert pieces[0].from_target is True
    assert pieces[0].text == "Monday,"
    assert pieces[0].start_char == 0
    assert pieces[0].end_char == 7
    assert pieces[1].from_target is False
    assert pieces[1].text == " hello"


def test_parse_edit_tagged_text_rejects_nested_markup():
    try:
        assembler.parse_edit_tagged_text("prefix <target>a <bad>b</bad></target>")
    except ValueError as exc:
        assert "Nested or malformed" in str(exc)
    else:
        raise AssertionError("Expected ValueError for nested markup")


def test_build_queries_preserves_repeated_substring_identity():
    base_text, pieces = assembler.parse_edit_tagged_text(
        "again <target>again</target> again"
    )
    queries, piece_to_query_index = assembler.build_queries(base_text, pieces)

    assert base_text == "again again again"
    assert queries == ["again <q>again</q> again"]
    assert piece_to_query_index == {1: 0}


def test_parse_edit_tagged_text_passthrough_target_only():
    base_text, pieces = assembler.parse_edit_tagged_text(
        "<target>Why are you beating up my jukebox?</target>"
    )

    assert base_text == "Why are you beating up my jukebox?"
    assert len(pieces) == 1
    assert pieces[0].from_target is True
    assert pieces[0].start_char == 0
    assert pieces[0].end_char == len(base_text)


def test_call_aligner_forwards_explicit_qwen3_backend(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        text = "ok"

        def json(self):
            return {"answers": [{"start": 0.0, "end": 0.4}]}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(assembler.requests, "post", _fake_post)

    answers = assembler._call_aligner(
        assembler.encode_wav(assembler.np.zeros(100, dtype=assembler.np.float32), 100),
        base_text="hello world",
        queries=["<q>hello</q> world"],
        aligner_url="http://aligner.test",
        aligner_backend="qwen3",
        language="eng",
        label="source",
    )

    assert answers == [{"start": 0.0, "end": 0.4}]
    assert captured["url"] == "http://aligner.test/align"
    assert captured["json"]["backend"] == "qwen3"
    assert captured["json"]["language"] == "eng"
    assert captured["json"]["queries"] == ["<q>hello</q> world"]


def test_assemble_mixed_audio_uses_query_boundaries_for_untagged_middle_span():
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    base_text, pieces = assembler.parse_edit_tagged_text(
        "<target>AA</target>BB<target>CC</target>"
    )
    queries, piece_to_query_index = assembler.build_queries(base_text, pieces)
    assert queries == ["<q>AA</q>BBCC", "AABB<q>CC</q>"]

    assembled, splice_points = assembler._assemble_mixed_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        piece_to_query_index=piece_to_query_index,
        source_answers=[
            {"start": 0.1, "end": 0.2},
            {"start": 0.7, "end": 0.8},
        ],
        target_answers=[
            {"start": 1.1, "end": 1.2},
            {"start": 1.7, "end": 1.8},
        ],
        preserve_outer_boundaries=True,
    )

    assert len(splice_points) == 2
    assert splice_points[0].position_sec == 1.2
    assert splice_points[1].position_sec == 1.7
    assert len(assembled) == 1000


def test_prepare_manual_time_spans_accepts_valid_non_adjacent_layout():
    edit_tagged_text = "AA<target>BB</target>CC<target>DD</target>EE"
    _, pieces = assembler.parse_edit_tagged_text(edit_tagged_text)

    prepared = assembler._prepare_manual_time_spans(
        edit_tagged_text=edit_tagged_text,
        pieces=pieces,
        source_duration_sec=10.0,
        target_duration_sec=8.0,
        time_spans_from_source=[(0.0, 1.0), (3.0, 4.0), (7.0, 8.0)],
        time_spans_from_target=[(0.5, 1.5), (2.0, 2.8)],
    )

    assert prepared is not None
    (
        source_span_by_piece_index,
        target_span_by_piece_index,
        source_answers,
        target_answers,
    ) = prepared
    assert source_span_by_piece_index[0] == {"start": 0.0, "end": 1.0}
    assert source_span_by_piece_index[2] == {"start": 3.0, "end": 4.0}
    assert source_span_by_piece_index[4] == {"start": 7.0, "end": 8.0}
    assert target_span_by_piece_index[1] == {"start": 0.5, "end": 1.5}
    assert target_span_by_piece_index[3] == {"start": 2.0, "end": 2.8}
    assert source_answers == [
        {"start": 1.0, "end": 3.0},
        {"start": 4.0, "end": 7.0},
    ]
    assert target_answers == [
        {"start": 0.5, "end": 1.5},
        {"start": 2.0, "end": 2.8},
    ]


def test_prepare_manual_time_spans_rejects_whitespace_adjacent_targets():
    edit_tagged_text = "<target>AA</target>  \n\t <target>BB</target>"
    _, pieces = assembler.parse_edit_tagged_text(edit_tagged_text)

    with pytest.raises(ValueError):
        prepared = assembler._prepare_manual_time_spans(
            edit_tagged_text=edit_tagged_text,
            pieces=pieces,
            source_duration_sec=5.0,
            target_duration_sec=5.0,
            time_spans_from_source=[],
            time_spans_from_target=[(0.1, 0.4), (0.6, 0.9)],
        )


def test_prepare_manual_time_spans_rejects_invalid_counts():
    edit_tagged_text = "AA<target>BB</target>CC"
    _, pieces = assembler.parse_edit_tagged_text(edit_tagged_text)

    with pytest.raises(ValueError):
        prepared = assembler._prepare_manual_time_spans(
            edit_tagged_text=edit_tagged_text,
            pieces=pieces,
            source_duration_sec=5.0,
            target_duration_sec=5.0,
            time_spans_from_source=[(0.0, 1.0)],
            time_spans_from_target=[(0.1, 0.4), (0.6, 0.9)],
        )


def test_assemble_manual_spans_audio_tracks_splice_points():
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    base_text, pieces = assembler.parse_edit_tagged_text(
        "AA<target>BB</target>CC<target>DD</target>EE"
    )

    assembled, splice_points = assembler._assemble_manual_spans_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        source_span_by_piece_index={
            0: {"start": 0.0, "end": 1.0},
            2: {"start": 3.0, "end": 4.0},
            4: {"start": 7.0, "end": 8.0},
        },
        target_span_by_piece_index={
            1: {"start": 1.1, "end": 1.2},
            3: {"start": 1.7, "end": 1.8},
        },
        preserve_outer_boundaries=True,
    )

    assert len(splice_points) == 4
    assert [point.position_sec for point in splice_points] == [1.0, 1.1, 2.1, 2.2]
    assert len(assembled) == 520


def test_assemble_mixed_audio_preserve_outer_boundaries_flag_controls_first_and_last_segment():
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    base_text, pieces = assembler.parse_edit_tagged_text(
        "<target>AA</target>BB<target>CC</target>"
    )
    queries, piece_to_query_index = assembler.build_queries(base_text, pieces)
    assert queries == ["<q>AA</q>BBCC", "AABB<q>CC</q>"]

    preserved, _ = assembler._assemble_mixed_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        piece_to_query_index=piece_to_query_index,
        source_answers=[
            {"start": 0.1, "end": 0.2},
            {"start": 0.7, "end": 0.8},
        ],
        target_answers=[
            {"start": 1.1, "end": 1.2},
            {"start": 1.7, "end": 1.8},
        ],
        preserve_outer_boundaries=True,
    )
    trimmed, _ = assembler._assemble_mixed_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        piece_to_query_index=piece_to_query_index,
        source_answers=[
            {"start": 0.1, "end": 0.2},
            {"start": 0.7, "end": 0.8},
        ],
        target_answers=[
            {"start": 1.1, "end": 1.2},
            {"start": 1.7, "end": 1.8},
        ],
        preserve_outer_boundaries=False,
    )

    assert len(preserved) == 1000
    assert len(trimmed) == 70
    assert preserved[0] == 1000.0
    assert trimmed[0] == 1110.0
    assert preserved[-1] == 1999.0
    assert trimmed[-1] == 1179.0


def test_assemble_manual_spans_preserve_outer_boundaries_flag_controls_first_and_last_segment():
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    base_text, pieces = assembler.parse_edit_tagged_text(
        "<target>AA</target>BB<target>CC</target>"
    )

    preserved, _ = assembler._assemble_manual_spans_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        source_span_by_piece_index={1: {"start": 3.0, "end": 4.0}},
        target_span_by_piece_index={
            0: {"start": 1.1, "end": 1.2},
            2: {"start": 1.7, "end": 1.8},
        },
        preserve_outer_boundaries=True,
    )
    trimmed, _ = assembler._assemble_manual_spans_audio(
        source_audio=source_audio,
        target_audio=target_audio,
        sr=100,
        base_text=base_text,
        pieces=pieces,
        source_span_by_piece_index={1: {"start": 3.0, "end": 4.0}},
        target_span_by_piece_index={
            0: {"start": 1.1, "end": 1.2},
            2: {"start": 1.7, "end": 1.8},
        },
        preserve_outer_boundaries=False,
    )

    assert len(preserved) == 1050
    assert len(trimmed) == 120
    assert preserved[0] == 1000.0
    assert trimmed[0] == 1110.0
    assert preserved[-1] == 1999.0
    assert trimmed[-1] == 1179.0


def test_run_assembly_uses_manual_spans_without_calling_aligner(monkeypatch):
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    source_bytes = assembler.encode_wav(source_audio, 100)
    target_bytes = assembler.encode_wav(target_audio, 100)
    smoother_calls = []

    def _fail_aligner(*args, **kwargs):
        raise AssertionError("aligner should not be called")

    def _fake_smoother(**kwargs):
        smoother_calls.append(kwargs)
        return kwargs["audio"]

    monkeypatch.setattr(assembler, "TARGET_SR", 100)
    monkeypatch.setattr(assembler, "_call_aligner", _fail_aligner)
    monkeypatch.setattr(assembler, "_smooth_splice_points", _fake_smoother)

    result = assembler.run_assembly(
        source_bytes=source_bytes,
        target_bytes=target_bytes,
        edit_tagged_text="AA<target>BB</target>CC",
        time_spans_from_source=[(0.0, 1.0), (3.0, 4.0)],
        time_spans_from_target=[(1.2, 1.5)],
    )

    assert len(smoother_calls) == 1
    assert result.details["assembly_mode"] == "splice_and_smooth"
    assert result.source_answers == [{"start": 1.0, "end": 3.0}]
    assert result.target_answers == [{"start": 1.2, "end": 1.5}]
    assert len(result.splice_points) == 2
    assert result.target_spans[0]["source_answer"] == {"start": 1.0, "end": 3.0}
    assert result.target_spans[0]["target_answer"] == {"start": 1.2, "end": 1.5}


def test_smooth_splice_points_uses_smooth_only_mode(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200
        content = assembler.encode_wav(assembler.np.zeros(16, dtype=assembler.np.float32), 100)

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(assembler.requests, "post", _fake_post)
    monkeypatch.setattr(assembler, "decode_audio", lambda _: (assembler.np.zeros(16, dtype=assembler.np.float32), 100))
    monkeypatch.setattr(assembler, "resample_audio", lambda audio, orig_sr, target_sr: audio)

    splice_points = [
        assembler.SplicePoint(position_sec=1.2),
        assembler.SplicePoint(position_sec=1.7),
    ]

    assembler._smooth_splice_points(
        audio=assembler.np.zeros(300, dtype=assembler.np.float32),
        sr=100,
        base_text="AABB",
        splice_points=splice_points,
        smoother_url="http://localhost:8018",
        regen_margin_sec=0.2,
        f5_steps=16,
        f5_cfg_strength=2.0,
        f5_sway_sampling_coef=-1.0,
        f5_seed=42,
        debug=False,
        debug_dir=None,
        trace_id=None,
    )

    assert captured["url"] == "http://localhost:8018/smooth"
    assert captured["json"]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": 1.2,
            "regen_margin_sec": 0.2,
        },
        {
            "mode": "smooth_only",
            "position_sec": 1.7,
            "regen_margin_sec": 0.2,
        },
    ]


def test_run_assembly_rejects_invalid_manual_spans_before_alignment(monkeypatch):
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    source_bytes = assembler.encode_wav(source_audio, 100)
    target_bytes = assembler.encode_wav(target_audio, 100)
    aligner_calls = []

    def _fake_aligner(*args, **kwargs):
        aligner_calls.append(kwargs["label"])
        return [{"start": 0.1, "end": 0.2}]

    monkeypatch.setattr(assembler, "TARGET_SR", 100)
    monkeypatch.setattr(assembler, "_call_aligner", _fake_aligner)
    monkeypatch.setattr(
        assembler, "_smooth_splice_points", lambda **kwargs: kwargs["audio"]
    )

    with pytest.raises(ValueError, match="source count"):
        result = assembler.run_assembly(
            source_bytes=source_bytes,
            target_bytes=target_bytes,
            edit_tagged_text="AA<target>BB</target>CC",
            time_spans_from_source=[(0.0, 1.0)],
            time_spans_from_target=[(1.2, 1.5)],
        )
    assert aligner_calls == []


def test_run_assembly_preserve_outer_boundaries_false_trims_outer_segments(monkeypatch):
    source_audio = assembler.np.arange(0, 1000, dtype=assembler.np.float32)
    target_audio = assembler.np.arange(1000, 2000, dtype=assembler.np.float32)
    source_bytes = assembler.encode_wav(source_audio, 100)
    target_bytes = assembler.encode_wav(target_audio, 100)

    def _fake_aligner(*args, **kwargs):
        return [{"start": 0.1, "end": 0.2}, {"start": 0.7, "end": 0.8}]

    monkeypatch.setattr(assembler, "TARGET_SR", 100)
    monkeypatch.setattr(assembler, "_call_aligner", _fake_aligner)
    monkeypatch.setattr(
        assembler, "_smooth_splice_points", lambda **kwargs: kwargs["audio"]
    )

    result = assembler.run_assembly(
        source_bytes=source_bytes,
        target_bytes=target_bytes,
        edit_tagged_text="<target>AA</target>BB<target>CC</target>",
        preserve_outer_boundaries=False,
    )

    assert len(result.audio) == 70
    assert result.details["assembled_duration"] == 0.7
