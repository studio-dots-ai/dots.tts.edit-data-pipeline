import pytest

from speech_edit.pipelines.pause.evaluator import (
    evaluate_pause_accuracy,
    evaluate_pause_edit,
    evaluate_wdtw,
)
from speech_edit.pipelines.pause.synthesizer import SynthesizedPair
from speech_edit.pipelines.pause.tag_utils import PauseEditProfile, PauseTag


def _profile(*, act: str = "ins") -> PauseEditProfile:
    operation = f'hello<pause act="{act}" level="2"/> world'
    return PauseEditProfile(
        profile_id=1,
        operation=operation,
        st_inst={
            "increase_pause": [operation] if act == "ins" else [],
            "decrease_pause": [operation] if act == "red" else [],
        },
        nl_inst={
            "increase_pause": ["add a pause"] if act == "ins" else [],
            "decrease_pause": ["reduce the pause"] if act == "red" else [],
        },
        tags=[
            PauseTag(
                char_idx=5,
                act=act,
                level=2,
                delta_dur_ms=500,
                left_context="hello",
                right_context="world",
                punctuation=None,
            )
        ],
    )


def _synth_pair(*, act: str = "ins", original_word_alignments=None) -> SynthesizedPair:
    return SynthesizedPair(
        profile=_profile(act=act),
        normed_text="hello world",
        generated_text="hello world",
        original_audio_bytes=b"original-wav",
        generated_audio_bytes=b"generated-wav",
        original_duration_sec=1.0,
        generated_duration_sec=1.5,
        original_word_alignments=original_word_alignments,
    )


class _GapAligner:
    def __init__(self, *, original_gap: float, generated_gap: float):
        self.original_gap = original_gap
        self.generated_gap = generated_gap

    def align(self, audio_bytes, text, *, queries=None, **kwargs):
        del text, queries, kwargs
        gap = (
            self.original_gap
            if audio_bytes == b"original-wav"
            else self.generated_gap
        )
        return {
            "answers": [
                {"start": 0.0, "end": 0.2},
                {"start": 0.2 + gap, "end": 0.4 + gap},
            ]
        }


@pytest.mark.parametrize(
    ("act", "generated_gap", "expected_target", "expected_actual", "expected_correct"),
    [
        ("ins", 0.7, 500, 200.0, True),
        ("red", 0.2, -500, -300.0, True),
        ("ins", 0.3, 500, -200.0, False),
        ("red", 0.7, -500, 200.0, False),
        ("ins", 0.5, 500, 0.0, False),
        ("red", 0.5, -500, 0.0, False),
    ],
)
def test_evaluate_pause_accuracy_reports_signed_target_and_direction(
    act,
    generated_gap,
    expected_target,
    expected_actual,
    expected_correct,
):
    measurements = evaluate_pause_accuracy(
        _synth_pair(act=act),
        _GapAligner(original_gap=0.5, generated_gap=generated_gap),
    )

    assert len(measurements) == 1
    measurement = measurements[0]
    assert measurement["target_added_pause_ms"] == expected_target
    assert measurement["actual_added_pause_ms"] == pytest.approx(expected_actual)
    assert measurement["target_vs_actual_gap_error_ms"] == pytest.approx(
        abs(expected_actual - expected_target)
    )
    assert measurement["direction_correct"] is expected_correct


def test_evaluate_pause_accuracy_rejects_missing_gaps():
    class MissingGeneratedGapAligner(_GapAligner):
        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if audio_bytes == b"generated-wav":
                return {"answers": []}
            return super().align(
                audio_bytes,
                text,
                queries=queries,
                **kwargs,
            )

    with pytest.raises(ValueError, match='Cannot measure requested pause'):
        measurements = evaluate_pause_accuracy(
            _synth_pair(act="red"),
            MissingGeneratedGapAligner(original_gap=0.5, generated_gap=0.2),
        )


def test_evaluate_pause_edit_uses_service_wer_and_cached_original_alignment_for_wdtw():
    cached_alignments = [{"word": "hello", "start": 0.0, "end": 0.2}]
    synth_pair = _synth_pair(original_word_alignments=cached_alignments)

    class FakeASR:
        def transcribe(self, audio_bytes):
            return {"text": "hello world"}

    class FakeWER:
        def __init__(self):
            self.calls = []

        def compute(self, reference, hypothesis, *, language="auto", timeout=30):
            self.calls.append(
                {
                    "reference": reference,
                    "hypothesis": hypothesis,
                    "language": language,
                }
            )
            return {"error_rate": 0.0, "metric": "wer"}

    class FakeAligner:
        base_url = "http://aligner.test"

        def align(self, audio_bytes, text, *, queries=None, **kwargs):
            if audio_bytes == b"original-wav":
                return {
                    "answers": [
                        {"start": 0.0, "end": 0.2},
                        {"start": 0.25, "end": 0.45},
                    ]
                }
            return {
                "answers": [
                    {"start": 0.0, "end": 0.2},
                    {"start": 0.75, "end": 0.95},
                ]
            }

    class FakeSpkSim:
        def similarity(self, audio_a, audio_b, *, timeout=30):
            return {"similarity": 0.91, "inference_time_seconds": 0.5}

    class FakeWDTW:
        def __init__(self):
            self.calls = []

        def compute(self, **kwargs):
            self.calls.append(kwargs)
            return {"wdtw": 0.12, "mode": "preserved"}

    wer = FakeWER()
    wdtw = FakeWDTW()
    result = evaluate_pause_edit(
        synth_pair,
        asr_client=FakeASR(),
        alignment_client=FakeAligner(),
        wer_client=wer,
        spk_sim_client=FakeSpkSim(),
        wdtw_client=wdtw,
        sample_language="en",
    )

    assert len(wer.calls) == 2
    assert wer.calls[0]["language"] == "auto"
    assert result.original_wer == 0.0
    assert result.generated_wer == 0.0
    assert result.spk_sim == {"similarity": 0.91, "inference_time_seconds": 0.5}
    assert result.wdtw_result == {"wdtw": 0.12, "mode": "preserved"}
    assert len(wdtw.calls) == 1
    assert wdtw.calls[0]["source_alignments"] == cached_alignments
    assert wdtw.calls[0]["target_alignments"] is None
    assert "aligner_url" not in wdtw.calls[0]
    assert wdtw.calls[0]["language"] == "eng"
    assert (
        wdtw.calls[0]["preservation_st_inst"]
        == synth_pair.profile.operation
    )


def test_evaluate_wdtw_allows_service_alignment_without_cached_original_alignment():
    class FakeWDTW:
        def __init__(self):
            self.calls = []

        def compute(self, **kwargs):
            self.calls.append(kwargs)
            return {"status": "ok", "wdtw": 0.2}

    wdtw = FakeWDTW()
    result = evaluate_wdtw(
        _synth_pair(original_word_alignments=None),
        wdtw_client=wdtw,
        sample_language="en",
    )

    assert result == {"status": "ok", "wdtw": 0.2}
    assert wdtw.calls[0]["source_alignments"] is None
    assert wdtw.calls[0]["target_alignments"] is None
    assert wdtw.calls[0]["source_audio"] == b"original-wav"
    assert wdtw.calls[0]["target_audio"] == b"generated-wav"
    assert "aligner_url" not in wdtw.calls[0]


def test_wer_uses_shared_normalization_without_pipeline_preprocessing():
    from types import SimpleNamespace
    from speech_edit.pipelines.pause.evaluator import evaluate_wer
    from tools.wer_calc.wer_calc import compute_error_rate

    reference = "One, two, it's one, two, u."
    hypothesis = "One, two. It's one, two. You."
    calls = []
    def compute(ref, hyp, *, language):
        calls.append((ref, hyp, language))
        return compute_error_rate(ref, hyp, language=language).to_dict()

    result = evaluate_wer(reference, b"audio",
        SimpleNamespace(transcribe=lambda _: {"text": hypothesis}),
        SimpleNamespace(compute=compute))
    assert calls == [(reference, hypothesis, "auto")]
    assert result["wer"] == compute_error_rate(reference, hypothesis).error_rate
