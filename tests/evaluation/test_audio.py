import pytest

from speech_edit.evaluation.audio import EvalMetrics, PairEvalResult, evaluate_audio


class FakeASRClient:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def transcribe(self, _audio_bytes, language="English"):
        self.calls.append({"language": language})
        return {"text": self.text}


class FakeWERClient:
    def __init__(self, error_rate: float | None = None, exc: Exception | None = None):
        self.error_rate = error_rate
        self.exc = exc
        self.calls = []

    def compute(self, reference, hypothesis, *, language="auto", timeout=30):
        self.calls.append(
            {
                "reference": reference,
                "hypothesis": hypothesis,
                "language": language,
                "timeout": timeout,
            }
        )
        if self.exc is not None:
            raise self.exc
        return {"error_rate": self.error_rate}


def test_evaluate_audio_prefers_wer_service_over_local_whitespace_wer():
    ref = "Good heavens--what game?"
    hyp = "Good heavens! What game?"
    wer_client = FakeWERClient(error_rate=0.0)

    metrics = evaluate_audio(
        b"wav",
        ref,
        b"source-wav",
        asr_client=FakeASRClient(hyp),
        wer_client=wer_client,
        utmos_client=None,
        spk_sim_client=None,
        parallel=False,
    )

    assert metrics.asr_text == hyp
    assert metrics.wer == 0.0
    assert metrics.wer_pct == 0.0
    assert metrics.wer_backend == "wer_service"
    assert metrics.wer_metric == "mixed"
    assert wer_client.calls[0]["language"] == "auto"


def test_evaluate_audio_uses_chinese_asr_and_wer_language():
    asr_client = FakeASRClient("你好世界")
    wer_client = FakeWERClient(error_rate=0.0)

    metrics = evaluate_audio(
        b"wav",
        "你好世界",
        b"source-wav",
        asr_client=asr_client,
        wer_client=wer_client,
        utmos_client=None,
        spk_sim_client=None,
        parallel=False,
        sample_language="zh",
    )

    assert metrics.wer == 0.0
    assert metrics.wer_metric == "mixed"
    assert asr_client.calls[0]["language"] == "Chinese"
    assert wer_client.calls[0]["language"] == "auto"


def test_evaluate_audio_accepts_iso_language_codes():
    asr_client = FakeASRClient("你好世界")
    wer_client = FakeWERClient(error_rate=0.0)

    evaluate_audio(
        b"wav",
        "你好世界",
        b"source-wav",
        asr_client=asr_client,
        wer_client=wer_client,
        utmos_client=None,
        spk_sim_client=None,
        parallel=False,
        sample_language="zho",
    )

    assert asr_client.calls[0]["language"] == "Chinese"
    assert wer_client.calls[0]["language"] == "auto"


def test_evaluate_audio_chinese_propagates_service_error():
    with pytest.raises(RuntimeError, match='service down'):
        metrics = evaluate_audio(
            b"wav",
            "你好世界",
            b"source-wav",
            asr_client=FakeASRClient("你好"),
            wer_client=FakeWERClient(exc=RuntimeError("service down")),
            utmos_client=None,
            spk_sim_client=None,
            parallel=False,
            sample_language="zh",
        )


def test_evaluate_audio_fails_when_selected_wer_service_fails():
    ref = "Good heavens--what game?"
    hyp = "Good heavens! What game?"

    with pytest.raises(RuntimeError, match='service down'):
        metrics = evaluate_audio(
            b"wav",
            ref,
            b"source-wav",
            asr_client=FakeASRClient(hyp),
            wer_client=FakeWERClient(exc=RuntimeError("service down")),
            utmos_client=None,
            spk_sim_client=None,
            parallel=False,
        )


def test_pair_eval_result_serializes_nested_audio_and_pair_metrics():
    result = PairEvalResult(
        source_metrics=EvalMetrics(
            asr_text="source text",
            wer=0.10,
            wer_pct=10.0,
            wer_backend="wer_service",
            wer_metric="wer",
        ),
        target_metrics=EvalMetrics(
            asr_text="target text",
            wer=0.25,
            wer_pct=25.0,
            wer_backend="wer_service",
            wer_metric="wer",
            utmos_score=4.1,
            spk_similarity=0.96,
        ),
        eval_time_sec=1.5,
    )

    payload = result.to_dict()

    assert payload["src"]["transcript"] == "source text"
    assert payload["src"]["wer"] == {
        "error_rate": 0.10,
        "metric": "wer",
        "backend": "wer_service",
    }
    assert payload["tgt"]["transcript"] == "target text"
    assert payload["tgt"]["wer"]["error_rate"] == 0.25
    assert payload["tgt"]["utmos"] == {"score": 4.1}
    assert payload["spk_sim"] == {"similarity": 0.96}
    assert payload["eval_time_sec"] == 1.5
    assert "delta" not in payload
    assert "pair_eval_time_sec" not in payload


def test_uncomputed_metrics_are_omitted_and_delta_preserves_missing_sentinel():
    from speech_edit.evaluation.audio import delta_metrics

    metrics = evaluate_audio(b"audio", "hello", b"source", parallel=False)
    assert set(metrics.to_dict()) == {"eval_time_sec"}
    assert delta_metrics(metrics, EvalMetrics(wer=0.1)) == {"wer": -1.0, "wer_pct": -1.0}
    assert delta_metrics(EvalMetrics(wer=0.2), EvalMetrics(wer=0.1)) == {
        "wer": -0.1, "wer_pct": -10.0,
    }


def test_parallel_audio_metric_failure_fails_evaluation():
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match='unavailable'):
        metrics = evaluate_audio(
            b"audio", "hello", b"source",
            asr_client=SimpleNamespace(transcribe=fail),
            wer_client=FakeWERClient(error_rate=0.0),
            utmos_client=SimpleNamespace(predict=lambda _: {"score": 4.2}),
            spk_sim_client=SimpleNamespace(similarity=fail),
        )


def test_failed_http_wer_never_invokes_local_implementation(monkeypatch):
    from speech_edit.evaluation.audio import compute_wer
    from tools.wer_calc import wer_calc

    def unexpected_local(*args, **kwargs):
        pytest.fail("HTTP failure must not invoke the local implementation")

    monkeypatch.setattr(wer_calc, "compute_error_rate", unexpected_local)
    with pytest.raises(RuntimeError, match="unavailable"):
        compute_wer("hello world", "hello", wer_client=FakeWERClient(exc=RuntimeError("unavailable")))


def test_service_wer_preserves_insertions_above_one():
    from speech_edit.evaluation.audio import compute_wer
    assert compute_wer("a b", "c d e", wer_client=FakeWERClient(error_rate=1.5))["error_rate"] == 1.5


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("axis", ["asr", "wer", "utmos", "spk_sim"])
def test_every_enabled_metric_failure_propagates(axis, parallel):
    from types import SimpleNamespace

    def fail(*args, **kwargs):
        raise RuntimeError(f"{axis} failed")

    clients = {
        "asr_client": SimpleNamespace(transcribe=lambda *a, **k: {"text": "hello"}),
        "wer_client": SimpleNamespace(compute=lambda *a, **k: {"error_rate": 0, "metric": "mixed"}),
        "utmos_client": SimpleNamespace(predict=lambda *a, **k: {"score": 4}),
        "spk_sim_client": SimpleNamespace(similarity=lambda *a, **k: {"similarity": 0.8}),
    }
    method = {"asr": "transcribe", "wer": "compute", "utmos": "predict", "spk_sim": "similarity"}[axis]
    setattr(clients[axis + "_client"], method, fail)
    with pytest.raises(RuntimeError, match=f"{axis} failed"):
        evaluate_audio(b"audio", "hello", b"source", parallel=parallel, **clients)


def test_asr_evaluation_requires_wer_service_before_transcription():
    from unittest.mock import Mock

    asr = Mock()
    with pytest.raises(ValueError, match="WER service client is required"):
        evaluate_audio(b"audio", "hello", b"source", asr_client=asr)
    asr.transcribe.assert_not_called()
