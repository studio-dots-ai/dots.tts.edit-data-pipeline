from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/pause_edit/defaults")

import io

import numpy as np
import pytest
import soundfile as sf

from speech_edit.pipelines.pause.synthesizer import (
    _apply_pause_waveform_edits,
    synthesize_pause_edit,
)
from speech_edit.pipelines.pause.tag_utils import (
    PauseEditGroup,
    PauseEditProfile,
    parse_instruction_pause_tags,
    parse_pause_tags,
)


def _wav_bytes(duration_sec: float = 0.5, sr: int = 24000) -> bytes:
    buf = io.BytesIO()
    sf.write(
        buf,
        np.zeros(int(duration_sec * sr), dtype=np.float32),
        sr,
        format="WAV",
        subtype="PCM_16",
    )
    return buf.getvalue()


def _build_pause_insert_fixture():
    operation = (
        'She walked slowly <pause act="ins" level="1"/> through the garden.'
    )
    profile = PauseEditProfile(
        profile_id=1,
        operation=operation,
        st_inst={
            "increase_pause": [operation],
            "decrease_pause": ['She walked slowly <pause act="red" level="1"/> through the garden.'],
        },
        nl_inst={
            "increase_pause": ["Add a short pause between 'slowly' and 'through'."],
            "decrease_pause": ["Reduce the pause between 'slowly' and 'through'."],
        },
        tags=parse_pause_tags(operation, "She walked slowly through the garden."),
    )
    group = PauseEditGroup(
        normed_text="She walked slowly through the garden.",
        profiles=[profile],
        duration_pool=[200, 500, 800],
        num_operations=1,
        original_audio=_wav_bytes(),
    )
    return profile, group


def test_synthesize_pause_edit_sets_fix_duration_from_source_gap(tmp_path):
    profile, group = _build_pause_insert_fixture()
    captured = {}

    class _FakeAlignmentClient:
        def __init__(self):
            self.calls = []

        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            self.calls.append(
                {
                    "text": text,
                    "queries": queries,
                    "word_level": kwargs.get("word_level", False),
                }
            )
            if kwargs.get("word_level", False):
                return {
                    "alignments": [
                        {"word": "She", "start": 0.0, "end": 0.1},
                        {"word": "walked", "start": 0.1, "end": 0.2},
                    ],
                    "backend": "qwen3",
                }
            return {
                "answers": [{"start": 0.35, "end": 0.35}],
                "query_tokenization_matches_default": True,
                "alignments": [{"word": "She", "start": 0.0, "end": 0.2}],
            }

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            captured.update(kwargs)
            return kwargs["audio_bytes"]

    alignment_client = _FakeAlignmentClient()

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=alignment_client,
        output_dir=str(tmp_path),
     expand_radius=0.15, aligner_backend="qwen3", synth_method="dual_boundary", center_regen_second_pass=False, center_regen_margin_sec=None)

    assert alignment_client.calls == [
        {
            "text": "She walked slowly through the garden.",
            "queries": None,
            "word_level": True,
        },
        {
            "text": "She walked slowly through the garden.",
            "queries": ["She walked slowly<q></q> through the garden."],
            "word_level": False,
        }
    ]
    assert captured["text"] == "She walked slowly, through the garden."
    assert captured["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.35),
            "regen_margin_sec": pytest.approx(0.06),
        },
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.55),
            "regen_margin_sec": pytest.approx(0.06),
        },
    ]
    assert result.generated_text == "She walked slowly through the garden."
    assert result.pause_edit_plan["insert_position_sec"] == [pytest.approx(0.35)]
    assert result.pause_edit_plan["pause_interval_sec"] == [[pytest.approx(0.35), pytest.approx(0.55)]]
    assert result.pause_edit_plan["f5_text"] == "She walked slowly, through the garden."
    assert result.original_word_alignments == [
        {"word": "She", "start": 0.0, "end": 0.1},
        {"word": "walked", "start": 0.1, "end": 0.2},
    ]


def test_synthesize_pause_edit_requires_alignment_for_insert_duration():
    profile, group = _build_pause_insert_fixture()

    class _UnusedSmootherClient:
        def smooth(self, **kwargs):
            raise AssertionError("smooth should not be called without an aligner")

    with pytest.raises(ValueError, match="alignment client"):
        synthesize_pause_edit(
            profile=profile,
            group=group,
            smoother_client=_UnusedSmootherClient(),
            alignment_client=None,
         expand_radius=0.15, aligner_backend="qwen3", synth_method="dual_boundary", center_regen_second_pass=False, center_regen_margin_sec=None)


def test_synthesize_pause_edit_uses_group_original_audio(tmp_path):
    profile, group = _build_pause_insert_fixture()
    original_audio = _wav_bytes(0.8)
    group.original_audio = original_audio

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            assert audio_bytes == original_audio
            return {
                "answers": [{"start": 0.20, "end": 0.20}],
                "query_tokenization_matches_default": True,
                "alignments": [{"word": "She", "start": 0.0, "end": 0.2}],
            }

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            assert kwargs["audio_bytes"] != original_audio
            return kwargs["audio_bytes"]

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        output_dir=str(tmp_path),
     expand_radius=0.15, aligner_backend="qwen3", synth_method="dual_boundary", center_regen_second_pass=False, center_regen_margin_sec=None)

    assert result.original_audio_bytes == original_audio
    assert result.original_duration_sec == pytest.approx(0.8, abs=0.02)


def test_synthesize_pause_edit_optional_center_regen_second_pass(tmp_path):
    profile, group = _build_pause_insert_fixture()
    smoother_calls = []

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            return {
                "answers": [{"start": 0.35, "end": 0.35}],
                "query_tokenization_matches_default": True,
                "alignments": [{"word": "She", "start": 0.0, "end": 0.2}],
            }

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            smoother_calls.append(kwargs)
            return kwargs["audio_bytes"]

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        output_dir=str(tmp_path),
        center_regen_second_pass=True,
     expand_radius=0.15, aligner_backend="qwen3", synth_method="dual_boundary", center_regen_margin_sec=None)

    assert len(smoother_calls) == 2
    assert smoother_calls[0]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.35),
            "regen_margin_sec": pytest.approx(0.06),
        },
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.55),
            "regen_margin_sec": pytest.approx(0.06),
        },
    ]
    assert smoother_calls[1]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.45),
            "regen_margin_sec": pytest.approx(0.095),
        }
    ]
    assert result.pause_edit_plan["second_pass_center_regen"]["enabled"] is True


def test_synthesize_pause_edit_single_region_uses_single_region_smoothing(tmp_path):
    profile, group = _build_pause_insert_fixture()
    smoother_calls = []

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            return {
                "answers": [{"start": 0.35, "end": 0.35}],
                "query_tokenization_matches_default": True,
                "alignments": [{"word": "She", "start": 0.0, "end": 0.2}],
            }

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            smoother_calls.append(kwargs)
            return kwargs["audio_bytes"]

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        output_dir=str(tmp_path),
        synth_method="single_region",
        expand_radius=0.2,
     aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    assert len(smoother_calls) == 1
    assert smoother_calls[0]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.45),
            "regen_margin_sec": pytest.approx(0.3),
        },
    ]
    assert result.pause_edit_plan["mode"] == "single_region_smooth_only"
    assert result.pause_edit_plan["merged_regen_expected_span"] == [
        pytest.approx(0.15),
        pytest.approx(0.75),
    ]
    assert "second_pass_center_regen" not in result.pause_edit_plan


def test_synthesize_pause_edit_insert_only_skips_smoother(tmp_path):
    profile, group = _build_pause_insert_fixture()

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            return {
                "answers": [{"start": 0.35, "end": 0.35}],
                "query_tokenization_matches_default": True,
                "alignments": [{"word": "She", "start": 0.0, "end": 0.2}],
            }

    class _UnusedSmootherClient:
        def smooth(self, **kwargs):
            raise AssertionError("smooth should not be called in insert_only mode")

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_UnusedSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        output_dir=str(tmp_path),
        synth_method="insert_only",
     expand_radius=0.15, aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    assert result.pause_edit_plan["mode"] == "insert_only_explicit_silence"
    assert result.pause_edit_plan["smoothing_points"] == []
    assert result.metadata["effective_regen_margin_sec"] == 0.0


def test_pause_insert_operations_disambiguate_repeated_anchor():
    tagged = 'go slowly and slowly <pause act="ins" level="1"/> home'
    tags = parse_pause_tags(tagged, "go slowly and slowly home")
    assert tags[0].char_idx == len("go slowly and slowly")


def _build_pause_reduction_fixture(level: int = 2):
    operation = (
        f'She walked slowly <pause act="red" level="{level}"/> through the garden.'
    )
    profile = PauseEditProfile(
        profile_id=1,
        operation=operation,
        st_inst={"increase_pause": [], "decrease_pause": [operation]},
        nl_inst={"increase_pause": [], "decrease_pause": ["Reduce the pause."]},
        tags=parse_instruction_pause_tags(operation, "She walked slowly through the garden."),
    )
    group = PauseEditGroup(
        normed_text="She walked slowly through the garden.",
        profiles=[profile],
        duration_pool=[200, 500, 800],
        num_operations=1,
        original_audio=_wav_bytes(2.0),
    )
    return profile, group


def test_reduction_removes_centered_aligned_gap_and_smooths_stitch(tmp_path):
    profile, group = _build_pause_reduction_fixture(level=2)
    smoother_calls = []

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if kwargs.get("word_level"):
                return {"alignments": [], "backend": "qwen3"}
            assert queries and len(queries) == 2
            return {
                "answers": [
                    {"start": 0.2, "end": 0.4},
                    {"start": 1.2, "end": 1.4},
                ]
            }

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            smoother_calls.append(kwargs)
            return kwargs["audio_bytes"]

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        synth_method="single_region",
        expand_radius=0.16,
        output_dir=str(tmp_path),
     aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    assert result.generated_duration_sec == pytest.approx(1.5, abs=0.02)
    operation = result.pause_edit_plan["operations"][0]
    assert operation["action"] == "red"
    assert operation["source_gap_sec"] == [pytest.approx(0.4), pytest.approx(1.2)]
    assert operation["source_remove_interval_sec"] == [
        pytest.approx(0.55),
        pytest.approx(1.05),
    ]
    assert operation["edited_stitch_position_sec"] == pytest.approx(0.55)
    assert smoother_calls[0]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.55),
            "regen_margin_sec": pytest.approx(0.16),
        }
    ]
    assert smoother_calls[0]["text"] == group.normed_text


def test_reduction_caps_removal_to_available_gap_without_smoothing():
    profile, group = _build_pause_reduction_fixture(level=3)

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if kwargs.get("word_level"):
                return {"alignments": [], "backend": "qwen3"}
            return {
                "answers": [
                    {"start": 0.2, "end": 0.4},
                    {"start": 0.5, "end": 0.7},
                ]
            }

    class _UnusedSmootherClient:
        def smooth(self, **kwargs):
            raise AssertionError("insert_only must not call the smoother")

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_UnusedSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        synth_method="insert_only",
     expand_radius=0.15, aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    operation = result.pause_edit_plan["operations"][0]
    assert operation["requested_duration_sec"] == pytest.approx(0.8)
    assert operation["actual_duration_sec"] == pytest.approx(0.1)
    assert operation["source_remove_interval_sec"] == [
        pytest.approx(0.4),
        pytest.approx(0.5),
    ]
    assert result.pause_edit_plan["mode"] == "splice_only"


def test_waveform_edit_pass_preserves_all_unedited_samples():
    source = np.arange(20, dtype=np.float32)
    edited, plans = _apply_pause_waveform_edits(
        source,
        10,
        operations=[
            {"action": "ins", "tag_index": 0, "position_sec": 0.5, "duration_sec": 0.2},
            {
                "action": "red",
                "tag_index": 1,
                "gap_start_sec": 1.0,
                "gap_end_sec": 1.6,
                "duration_sec": 0.4,
            },
        ],
    )

    np.testing.assert_array_equal(
        edited,
        np.concatenate([source[:5], np.zeros(2, dtype=np.float32), source[5:11], source[15:]]),
    )
    assert [plan["action"] for plan in plans] == ["ins", "red"]


def test_multiple_reductions_use_original_alignment_and_edited_stitch_coordinates(
    tmp_path,
):
    operation = (
        'one<pause act="red" level="1"/> two three'
        '<pause act="red" level="2"/> four'
    )
    profile = PauseEditProfile(
        profile_id=1,
        operation=operation,
        st_inst={"increase_pause": [], "decrease_pause": [operation]},
        nl_inst={"increase_pause": [], "decrease_pause": ["Reduce pauses."]},
        tags=parse_instruction_pause_tags(operation, "one two three four"),
    )
    group = PauseEditGroup(
        normed_text="one two three four",
        profiles=[profile],
        duration_pool=[200, 500, 800],
        num_operations=2,
        original_audio=_wav_bytes(3.0),
    )
    aligned_gaps = iter(
        [
            [{"start": 0.2, "end": 0.4}, {"start": 0.8, "end": 1.0}],
            [{"start": 1.5, "end": 1.8}, {"start": 2.6, "end": 2.8}],
        ]
    )
    smoother_calls = []

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if kwargs.get("word_level"):
                return {"alignments": [], "backend": "qwen3"}
            return {"answers": next(aligned_gaps)}

    class _FakeSmootherClient:
        def smooth(self, **kwargs):
            smoother_calls.append(kwargs)
            return kwargs["audio_bytes"]

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_FakeSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        synth_method="single_region",
        expand_radius=0.16,
        output_dir=str(tmp_path),
     aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    operations = result.pause_edit_plan["operations"]
    assert [item["actual_duration_sec"] for item in operations] == [
        pytest.approx(0.2),
        pytest.approx(0.5),
    ]
    assert [item["edited_stitch_position_sec"] for item in operations] == [
        pytest.approx(0.5),
        pytest.approx(1.75),
    ]
    assert smoother_calls[0]["points"] == [
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(0.5),
            "regen_margin_sec": pytest.approx(0.16),
        },
        {
            "mode": "smooth_only",
            "position_sec": pytest.approx(1.75),
            "regen_margin_sec": pytest.approx(0.16),
        },
    ]


def test_zero_length_reduction_gap_does_not_regenerate_audio():
    profile, group = _build_pause_reduction_fixture(level=2)

    class _FakeAlignmentClient:
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if kwargs.get("word_level"):
                return {"alignments": [], "backend": "qwen3"}
            return {
                "answers": [
                    {"start": 0.2, "end": 0.4},
                    {"start": 0.4, "end": 0.7},
                ]
            }

    class _UnusedSmootherClient:
        def smooth(self, **kwargs):
            raise AssertionError("a zero-length edit must not call the smoother")

    result = synthesize_pause_edit(
        profile=profile,
        group=group,
        smoother_client=_UnusedSmootherClient(),
        alignment_client=_FakeAlignmentClient(),
        synth_method="single_region",
     expand_radius=0.15, aligner_backend="qwen3", center_regen_second_pass=False, center_regen_margin_sec=None)

    operation = result.pause_edit_plan["operations"][0]
    assert operation["actual_duration_sec"] == 0.0
    assert result.pause_edit_plan["smoothing_points"] == []
    original, original_sr = sf.read(io.BytesIO(group.original_audio), dtype="float32")
    generated, generated_sr = sf.read(
        io.BytesIO(result.generated_audio_bytes), dtype="float32"
    )
    assert generated_sr == original_sr
    np.testing.assert_array_equal(generated, original)
