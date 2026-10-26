from __future__ import annotations

import importlib.util
import tempfile
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _load_smoother_module():
    module_path = Path(__file__).resolve().parents[1] / "smoother.py"
    spec = importlib.util.spec_from_file_location("boundary_pause_smoother_core", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoother = _load_smoother_module()


def test_normalize_points_uses_interval_midpoint_for_position():
    points = [
        smoother.SmoothingPoint(
            mode="adjust",
            position_sec=9.9,
            pause_interval_sec=(1.0, 1.4),
            min_pause_sec=0.1,
            max_pause_sec=0.3,
            regen_margin_sec=0.2,
        )
    ]

    normalized = smoother._normalize_points(points, total_dur=5.0)

    assert len(normalized) == 1
    assert normalized[0].mode == "adjust"
    assert normalized[0].position_sec == pytest.approx(1.2)


def test_call_aligner_qwen3_payload_omits_ctc_only_options(monkeypatch):
    captured = {}

    class _FakeResponse:
        status_code = 200

        def json(self):
            return {"alignments": [{"text": "hello", "start": 0.0, "end": 0.5}]}

    def _fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(smoother.requests, "post", _fake_post)

    alignments = smoother._call_aligner(
        np.zeros(100, dtype=np.float32),
        100,
        "hello",
        "http://aligner.test",
        aligner_backend="qwen3",
        language="eng",
    )

    assert alignments == [{"text": "hello", "start": 0.0, "end": 0.5}]
    assert captured["url"] == "http://aligner.test/align"
    assert captured["json"]["backend"] == "qwen3"
    assert captured["json"]["language"] == "eng"
    assert captured["json"]["word_level"] is True
    assert "separate_punctuation" not in captured["json"]


def test_plan_pause_adjustments_expands_zero_length_interval_symmetrically():
    points = [
        smoother.SmoothingPoint(
            mode="adjust",
            pause_interval_sec=(1.0, 1.0),
            min_pause_sec=0.2,
            max_pause_sec=0.2,
            regen_margin_sec=0.1,
        )
    ]

    normalized = smoother._normalize_points(points, total_dur=3.0)
    plans = smoother._plan_pause_adjustments(
        normalized,
        sample_rate=100,
        audio_num_samples=300,
        total_dur=3.0,
        alignments=[],
        vad_silences=[],
    )

    assert plans[0].native_duration_samples == 0
    assert plans[0].target_duration_samples == 20
    assert plans[0].target_left_samples == 10
    assert plans[0].target_right_samples == 10


def test_build_pause_adjusted_audio_shrinks_around_position_anchor():
    audio = np.arange(0, 10, dtype=np.float32)
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.4,
        native_pause={"pause_start": 0.2, "pause_end": 0.8, "method": "provided"},
        native_start_idx=2,
        native_position_idx=4,
        native_end_idx=8,
        native_left_samples=2,
        native_right_samples=4,
        native_duration_samples=6,
        target_duration_samples=2,
        target_left_samples=1,
        target_right_samples=1,
        regen_margin_samples=1,
    )

    adjusted = smoother._build_pause_adjusted_audio(audio, [plan])

    assert adjusted.tolist() == [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]
    assert plan.adjusted_start_idx == 2
    assert plan.adjusted_position_idx == 3
    assert plan.adjusted_end_idx == 4


def test_build_pause_adjusted_audio_expands_around_position_anchor():
    audio = np.arange(0, 10, dtype=np.float32)
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.4,
        native_pause={"pause_start": 0.4, "pause_end": 0.4, "method": "provided"},
        native_start_idx=4,
        native_position_idx=4,
        native_end_idx=4,
        native_left_samples=0,
        native_right_samples=0,
        native_duration_samples=0,
        target_duration_samples=4,
        target_left_samples=2,
        target_right_samples=2,
        regen_margin_samples=1,
    )

    adjusted = smoother._build_pause_adjusted_audio(audio, [plan])

    assert adjusted.tolist() == [0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    assert plan.adjusted_start_idx == 4
    assert plan.adjusted_position_idx == 6
    assert plan.adjusted_end_idx == 8


def test_plan_pause_adjustments_rejects_overlapping_intervals():
    points = [
        smoother.SmoothingPoint(mode="adjust", pause_interval_sec=(1.0, 1.5)),
        smoother.SmoothingPoint(mode="adjust", pause_interval_sec=(1.4, 1.8)),
    ]
    normalized = smoother._normalize_points(points, total_dur=3.0)

    with pytest.raises(ValueError, match="overlap"):
        smoother._plan_pause_adjustments(
            normalized,
            sample_rate=100,
            audio_num_samples=300,
            total_dur=3.0,
            alignments=[],
            vad_silences=[],
        )


def test_resolve_native_pause_supports_mixed_explicit_and_auto_detect():
    points = [
        smoother.SmoothingPoint(mode="adjust", pause_interval_sec=(0.5, 0.7)),
        smoother.SmoothingPoint(mode="adjust", position_sec=1.5, max_pause_sec=0.0),
    ]
    normalized = smoother._normalize_points(points, total_dur=3.0)

    plans = smoother._plan_pause_adjustments(
        normalized,
        sample_rate=100,
        audio_num_samples=300,
        total_dur=3.0,
        alignments=[
            {"char": "A", "start": 1.0, "end": 1.3, "is_punctuation": False},
            {"char": "B", "start": 1.7, "end": 2.0, "is_punctuation": False},
        ],
        vad_silences=[{"start": 1.32, "end": 1.68, "duration": 0.36}],
    )

    assert plans[0].native_pause["method"] == "provided"
    assert plans[1].native_pause["method"] == "alignment+vad"


def test_normalize_adjust_position_only_preserves_pause_bounds():
    points = [
        smoother.SmoothingPoint(
            mode="adjust",
            position_sec=1.5,
            min_pause_sec=0.0,
            max_pause_sec=0.2,
            regen_margin_sec=0.1,
        )
    ]

    normalized = smoother._normalize_points(points, total_dur=3.0)

    assert normalized[0].mode == "adjust"
    assert normalized[0].position_sec == pytest.approx(1.5)
    assert normalized[0].pause_interval_sec is None
    assert normalized[0].min_pause_sec == pytest.approx(0.0)
    assert normalized[0].max_pause_sec == pytest.approx(0.2)


def test_smooth_only_mode_keeps_pause_fields_empty():
    points = [
        smoother.SmoothingPoint(position_sec=1.0, regen_margin_sec=0.2),
    ]

    normalized = smoother._normalize_points(points, total_dur=3.0)

    assert normalized[0].mode == "smooth_only"
    assert normalized[0].position_sec == pytest.approx(1.0)
    assert normalized[0].pause_interval_sec is None
    assert normalized[0].min_pause_sec is None
    assert normalized[0].max_pause_sec is None


def test_smooth_only_mode_rejects_pause_interval():
    points = [
        smoother.SmoothingPoint(position_sec=1.0, pause_interval_sec=(0.9, 1.1)),
    ]

    with pytest.raises(ValueError, match="smooth_only"):
        smoother._normalize_points(points, total_dur=3.0)


def test_build_smooth_only_details_and_spans():
    normalized = [
        smoother._NormalizedPoint(
            orig_idx=0,
            mode="smooth_only",
            position_sec=1.0,
            pause_interval_sec=None,
            min_pause_sec=None,
            max_pause_sec=None,
            regen_margin_sec=0.2,
        )
    ]

    details, spans, processed = smoother._build_smooth_only_details_and_spans(
        normalized,
        sample_rate=100,
        total_dur=3.0,
    )

    assert processed == 1
    assert spans == [[0.8, 1.2]]
    assert details[0]["mode"] == "smooth_only"
    assert details[0]["action"] == "smooth_only"


def test_audio_insert_uses_pause_boundaries_for_left_right_regen_spans():
    adjusted_audio = np.zeros(200, dtype=np.float32)
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.ones(40, dtype=np.float32),
    )

    details, spans, protected, processed = smoother._build_details_and_regen_spans(
        adjusted_audio, [plan], sample_rate=100
    )

    assert processed == 1
    assert spans == [[0.4, 0.5], [0.9, 1.0]]
    assert protected == [[0.5, 0.9]]
    assert plan.insert_start_idx == 50
    assert plan.insert_end_idx == 90
    assert details[0]["audio_insert_interval"]["duration"] == pytest.approx(0.4)
    assert details[0]["regen"]["spans"] == [
        {"start": 0.4, "end": 0.5, "duration": 0.1},
        {"start": 0.9, "end": 1.0, "duration": 0.1},
    ]


def test_audio_insert_intrusion_expands_regen_spans_into_insert_edges():
    adjusted_audio = np.zeros(200, dtype=np.float32)
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.ones(40, dtype=np.float32),
        intrusion_samples=8,
    )

    details, spans, protected, processed = smoother._build_details_and_regen_spans(
        adjusted_audio, [plan], sample_rate=100
    )

    assert processed == 1
    assert spans == [[0.4, 0.58], [0.82, 1.0]]
    assert protected == [[0.5, 0.9]]
    assert details[0]["audio_insert_intrusion"]["duration"] == pytest.approx(0.08)
    assert details[0]["regen"]["spans"] == [
        {"start": 0.4, "end": 0.58, "duration": 0.18},
        {"start": 0.82, "end": 1.0, "duration": 0.18},
    ]


def test_subtract_protected_intervals_removes_force_region_from_neighbor_masks():
    spans = [[0.0, 1.0], [1.2, 1.5]]
    protected = [[0.4, 0.6]]

    assert smoother._subtract_protected_intervals(spans, protected) == [
        [0.0, 0.4],
        [0.6, 1.0],
        [1.2, 1.5],
    ]


def test_audio_insert_sets_target_pause_from_insert_length():
    points = [
        smoother.SmoothingPoint(
            mode="adjust",
            pause_interval_sec=(1.0, 1.0),
            min_pause_sec=0.3,
            max_pause_sec=0.0,
            regen_margin_sec=0.1,
            audio_insert=np.ones(20, dtype=np.float32),
        )
    ]

    normalized = smoother._normalize_points(points, total_dur=3.0)
    plans = smoother._plan_pause_adjustments(
        normalized,
        sample_rate=100,
        audio_num_samples=300,
        total_dur=3.0,
        alignments=[],
        vad_silences=[],
    )
    adjusted = smoother._build_pause_adjusted_audio(
        np.zeros(300, dtype=np.float32),
        plans,
    )
    _, _, _, _ = smoother._build_details_and_regen_spans(
        adjusted, plans, sample_rate=100
    )
    inserted = smoother._apply_audio_inserts(adjusted, plans)

    assert plans[0].target_duration_samples == 20
    assert plans[0].insert_end_idx - plans[0].insert_start_idx == 20
    assert np.allclose(inserted[plans[0].insert_start_idx : plans[0].insert_end_idx], 1.0)


def test_audio_insert_f5_input_zero_fills_only_force_silence_interval():
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.full(40, 0.25, dtype=np.float32),
    )
    adjusted = np.ones(200, dtype=np.float32)

    _, _, _, _ = smoother._build_details_and_regen_spans(
        adjusted, [plan], sample_rate=100
    )
    f5_input = smoother._build_f5_input_audio(adjusted, [plan])

    assert np.allclose(f5_input[50:90], 0.0)
    assert np.allclose(f5_input[40:50], 1.0)
    assert np.allclose(f5_input[90:100], 1.0)
    assert np.allclose(f5_input[:50], 1.0)
    assert np.allclose(f5_input[100:], 1.0)


def test_audio_insert_intrusion_keeps_regen_edges_and_zeroes_full_insert_interval():
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.full(40, 0.25, dtype=np.float32),
        intrusion_samples=8,
    )
    adjusted = np.ones(200, dtype=np.float32)

    _, _, _, _ = smoother._build_details_and_regen_spans(
        adjusted, [plan], sample_rate=100
    )
    f5_input = smoother._build_f5_input_audio(adjusted, [plan])

    assert np.allclose(f5_input[40:50], 1.0)
    assert np.allclose(f5_input[50:90], 0.0)
    assert np.allclose(f5_input[90:100], 1.0)


def test_audio_insert_replaces_zero_filled_insert_interval_after_f5():
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.full(40, 0.25, dtype=np.float32),
    )
    adjusted = np.ones(200, dtype=np.float32)

    _, _, _, _ = smoother._build_details_and_regen_spans(
        adjusted, [plan], sample_rate=100
    )
    f5_input = smoother._build_f5_input_audio(adjusted, [plan])
    final_audio = smoother._apply_audio_inserts(f5_input, [plan])

    assert np.allclose(f5_input[plan.insert_start_idx : plan.insert_end_idx], 0.0)
    assert np.allclose(
        final_audio[plan.insert_start_idx : plan.insert_end_idx],
        0.25,
    )


def test_audio_insert_intrusion_still_overwrites_full_insert_interval_after_f5():
    plan = smoother._PausePlan(
        orig_idx=0,
        position_sec=0.5,
        native_pause={"pause_start": 0.5, "pause_end": 0.9, "method": "provided"},
        native_start_idx=50,
        native_position_idx=70,
        native_end_idx=90,
        native_left_samples=20,
        native_right_samples=20,
        native_duration_samples=40,
        target_duration_samples=40,
        target_left_samples=20,
        target_right_samples=20,
        regen_margin_samples=10,
        adjusted_start_idx=50,
        adjusted_position_idx=70,
        adjusted_end_idx=90,
        audio_insert=np.full(40, 0.25, dtype=np.float32),
        intrusion_samples=8,
    )
    adjusted = np.ones(200, dtype=np.float32)

    _, _, _, _ = smoother._build_details_and_regen_spans(
        adjusted, [plan], sample_rate=100
    )
    f5_input = smoother._build_f5_input_audio(adjusted, [plan])
    final_audio = smoother._apply_audio_inserts(f5_input, [plan])

    assert np.allclose(f5_input[40:50], 1.0)
    assert np.allclose(f5_input[50:90], 0.0)
    assert np.allclose(f5_input[90:100], 1.0)
    assert np.allclose(
        final_audio[plan.insert_start_idx : plan.insert_end_idx],
        0.25,
    )


def test_debug_saves_post_f5_and_post_insert_outputs_separately(monkeypatch):
    sample_rate = 100
    audio = np.linspace(-0.2, 0.2, 200, dtype=np.float32)
    insert_audio = np.full(20, 0.75, dtype=np.float32)

    def _fake_regenerate_regions(
        input_audio,
        sample_rate,
        _text,
        merged_spans,
        **_kwargs,
    ):
        output = input_audio.copy()
        for start_sec, end_sec in merged_spans:
            start_idx = int(round(start_sec * sample_rate))
            end_idx = int(round(end_sec * sample_rate))
            output[start_idx:end_idx] += 0.1
        return smoother._RegenerationResult(
            spliced_audio=output,
            f5_output_audio=np.full(80, 0.2, dtype=np.float32),
            f5_output_sample_rate=25,
        )

    monkeypatch.setattr(smoother, "_regenerate_regions", _fake_regenerate_regions)

    with tempfile.TemporaryDirectory() as tmpdir:
        result = smoother.smooth_audio(
            audio=audio,
            sample_rate=sample_rate,
            text="abc",
            points=[
                smoother.SmoothingPoint(
                    mode="adjust",
                    pause_interval_sec=(0.5, 0.9),
                    regen_margin_sec=0.1,
                    audio_insert=insert_audio,
                )
            ],
            debug=True,
            debug_dir=tmpdir,
        )

        f5_input_audio, _ = sf.read(
            str(Path(tmpdir) / "02_f5_input.wav"),
            dtype="float32",
        )
        mask_audio, _ = sf.read(
            str(Path(tmpdir) / "03_mask_visualization.wav"),
            dtype="float32",
        )
        raw_f5_audio, raw_f5_sr = sf.read(
            str(Path(tmpdir) / "04_f5_output.wav"),
            dtype="float32",
        )
        post_f5_audio, _ = sf.read(
            str(Path(tmpdir) / "05_smoothed.wav"),
            dtype="float32",
        )
        post_insert_audio, _ = sf.read(
            str(Path(tmpdir) / "06_audio_inserted.wav"),
            dtype="float32",
        )

        assert (Path(tmpdir) / "06_audio_inserted.wav").exists()
        assert raw_f5_sr == 25
        assert np.allclose(raw_f5_audio, 0.2, atol=1e-4)
        assert not np.allclose(post_f5_audio, result.audio, atol=1e-4)
        assert np.allclose(post_insert_audio, result.audio, atol=1e-4)

        detail = result.details[0]
        insert_start = int(round(detail["audio_insert_interval"]["start"] * sample_rate))
        insert_end = insert_start + len(insert_audio)
        assert np.allclose(f5_input_audio[40:50], audio[40:50], atol=1e-4)
        assert np.allclose(f5_input_audio[50:70], 0.0, atol=1e-4)
        assert not np.allclose(mask_audio, f5_input_audio, atol=1e-4)
        assert np.max(
            np.abs(mask_audio[insert_start - 5 : insert_start + 5] - f5_input_audio[insert_start - 5 : insert_start + 5])
        ) > 1e-4
        assert np.max(
            np.abs(mask_audio[insert_end - 5 : insert_end + 5] - f5_input_audio[insert_end - 5 : insert_end + 5])
        ) > 1e-4
        assert np.allclose(post_f5_audio[insert_start:insert_end], 0.0, atol=1e-4)
        assert np.allclose(post_insert_audio[insert_start:insert_end], insert_audio, atol=1e-4)


def test_default_aligner_payload_uses_auto_without_unsupported_options(monkeypatch):
    captured = {}
    class Response:
        status_code = 200
        def json(self):
            return {"alignments": []}
    def post(url, json, timeout):
        captured.update(json)
        return Response()
    monkeypatch.setattr(smoother.requests, "post", post)
    assert smoother._call_aligner(
        np.zeros(100, dtype=np.float32), 100, "hello", "http://aligner.test",
        debug=True, debug_dir="unused",
    ) == []
    assert captured["backend"] == "auto"
    assert captured["word_level"] is True
    assert not {"separate_punctuation", "debug", "debug_dir"} & captured.keys()
