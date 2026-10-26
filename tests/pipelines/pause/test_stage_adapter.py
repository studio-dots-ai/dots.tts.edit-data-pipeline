from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/pause_edit/defaults")

import io
import json

import numpy as np
import soundfile as sf

from speech_edit.runtime.models import (
    AdapterContext,
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
    StageSpec,
)
from speech_edit.pipelines.pause.evaluator import PauseEvalResult
from speech_edit.pipelines.pause.parquet_rows import PauseEditParquetRowProcessor
from speech_edit.pipelines.pause.stage_adapter import (
    PauseEvaluatedWorkItem,
    PauseEditStageAdapter,
    PauseSampleWorkItem,
    PauseProfileWorkItem,
    PauseSynthesizedWorkItem,
)
import speech_edit.pipelines.pause.stage_adapter as stage_adapter_module
from speech_edit.pipelines.pause.synthesizer import SynthesizedPair
from speech_edit.pipelines.pause.tag_utils import PauseEditProfile, PauseTag
from tests.service_graph_fixtures import direct_services


def _wav_bytes(duration_sec: float = 1.0, sample_rate: int = 24000) -> bytes:
    audio = np.zeros(int(duration_sec * sample_rate), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_pause_parquet_row_processor_builds_expected_row() -> None:
    operation = (
        'Hello<pause act="ins" level="2"/> world, '
        'again<pause act="ins" level="1"/> today.'
    )
    profile = PauseEditProfile(
        profile_id=1,
        operation=operation,
        st_inst={
            "increase_pause": [operation],
            "decrease_pause": [
                'Hello<pause act="red" level="2"/> world, '
                'again<pause act="red" level="1"/> today.'
            ],
        },
        nl_inst={
            "increase_pause": ["Add pauses after hello and again."],
            "decrease_pause": ["Reduce the inserted pauses."],
        },
        tags=[
            PauseTag(
                char_idx=5,
                act="ins",
                level=2,
                delta_dur_ms=500,
                left_context="Hello",
                right_context="world, again today.",
                punctuation=None,
            ),
            PauseTag(
                char_idx=18,
                act="ins",
                level=1,
                delta_dur_ms=200,
                left_context="Hello world, again",
                right_context="today.",
                punctuation=None,
            ),
        ],
    )
    work_item = PauseProfileWorkItem(
        sample_id="sample_pause",
        normed_text="Hello world, again today.",
        profile=profile,
        duration_pool=[200, 500, 800],
        num_operations=3,
        original_audio=_wav_bytes(1.0),
        metadata={
            "clip_id": "clip_001",
            "segment_clip_id": "clip_001",
            "language": "en",
            "sample_rate": 24000,
            "source_dataset": "Emilia",
        },
    )
    synth_pair = SynthesizedPair(
        profile=profile,
        normed_text=work_item.normed_text,
        generated_text=work_item.normed_text,
        original_audio_bytes=work_item.original_audio,
        generated_audio_bytes=_wav_bytes(1.3),
        original_duration_sec=1.0,
        generated_duration_sec=1.3,
        pause_edit_plan={"pause_duration_sec": [0.5, 0.2]},
        metadata={"synth_method": "single_region"},
    )
    eval_result = PauseEvalResult(
        profile_id=1,
        original_wer=0.01,
        generated_wer=0.03,
        content_passed=True,
        pause_measurements=[
            {
                "target_added_pause_ms": 500,
                "actual_added_pause_ms": 430,
                "target_vs_actual_gap_error_ms": 70,
            },
            {
                "target_added_pause_ms": 200,
                "actual_added_pause_ms": 150,
                "target_vs_actual_gap_error_ms": 50,
            },
        ],
        metadata={
            "original_transcription": "hello world again today",
            "generated_transcription": "hello world again today",
            "original_wer_result": {"error_rate": 0.01, "metric": "mixed"},
            "generated_wer_result": {"error_rate": 0.03, "metric": "mixed"},
        },
        spk_sim={"similarity": 0.91},
        wdtw_result={"wdtw": 0.12, "mode": "preserved"},
    )

    rows = PauseEditParquetRowProcessor().rows_from_payload(
        PauseEvaluatedWorkItem(
            sample_id="sample_pause",
            work_item=work_item,
            synth_pair=synth_pair,
            eval_result=eval_result,
        ),
        dataset_name="Emilia-PauseEdit-Test",
    )

    assert len(rows) == 2
    row = rows[0]
    reverse = rows[1]
    assert row["from_dataset"] == "Emilia-PauseEdit-Test"
    assert row["audio_id"] == "sample_pause--profile_1--original_to_generated"
    assert reverse["audio_id"] == "sample_pause--profile_1--generated_to_original"
    assert json.loads(row["tagged_texts"])["tgt"] == operation
    assert json.loads(row["instructions"])["st"] == operation
    assert json.loads(reverse["instructions"])["st"] == (
        'Hello<pause act="red" level="2"/> world, '
        'again<pause act="red" level="1"/> today.'
    )
    assert json.loads(row["pair_metadata"])["direction"] == "original_to_generated"
    assert json.loads(reverse["pair_metadata"])["direction"] == "generated_to_original"
    edit_metadata = json.loads(row["edit_metadata"])
    reverse_edit_metadata = json.loads(reverse["edit_metadata"])
    assert [item["act"] for item in edit_metadata["operations"]] == ["ins", "ins"]
    assert [item["level"] for item in edit_metadata["operations"]] == [2, 1]
    assert [item["act"] for item in reverse_edit_metadata["operations"]] == ["red", "red"]
    assert "num_pauses" not in row
    eval_results = json.loads(row["eval_results"])
    reverse_eval_results = json.loads(reverse["eval_results"])
    assert eval_results["src"]["wer"]["error_rate"] == 0.01
    assert eval_results["tgt"]["wer"]["error_rate"] == 0.03
    assert eval_results["spk_sim"] == {"similarity": 0.91}
    assert eval_results["wdtw"] == 0.12
    assert eval_results["wdtw_result"] == {"wdtw": 0.12, "mode": "preserved"}
    assert eval_results["src"]["transcript"] == "hello world again today"
    assert eval_results["pause"]["operations"] == [
        {
            "expected_tgt_minus_src_pause_gap_ms": 500.0,
            "measured_tgt_minus_src_pause_gap_ms": 430.0,
            "error_tgt_minus_src_pause_gap_ms": -70.0,
        },
        {
            "expected_tgt_minus_src_pause_gap_ms": 200.0,
            "measured_tgt_minus_src_pause_gap_ms": 150.0,
            "error_tgt_minus_src_pause_gap_ms": -50.0,
        },
    ]
    assert reverse_eval_results["src"]["wer"]["error_rate"] == 0.03
    assert reverse_eval_results["tgt"]["wer"]["error_rate"] == 0.01
    assert reverse_eval_results["spk_sim"] == {"similarity": 0.91}
    assert reverse_eval_results["wdtw"] == 0.12
    assert reverse_eval_results["pause"]["operations"][0][
        "expected_tgt_minus_src_pause_gap_ms"
    ] == -500.0
    assert reverse_eval_results["pause"]["operations"][0][
        "measured_tgt_minus_src_pause_gap_ms"
    ] == -430.0


def test_stage_adapter_plan_returns_profile_work_items(monkeypatch) -> None:
    adapter = PauseEditStageAdapter()
    payload = type(
        "SamplePayload",
        (),
        {
            "sample_id": "sample_001",
            "segment": None,
            "attempt_index": 1,
        },
    )()
    profile = PauseEditProfile(
        profile_id=1,
        operation='hello<pause act="ins" level="2"/> world',
        st_inst={"increase_pause": ['hello<pause act="ins" level="2"/> world'], "decrease_pause": ['hello<pause act="red" level="2"/> world']},
        nl_inst={"increase_pause": ["add a pause"], "decrease_pause": ["remove the pause"]},
        tags=[
            PauseTag(
                char_idx=5,
                act="ins",
                level=2,
                delta_dur_ms=500,
                left_context="hello",
                right_context="world",
                punctuation=None,
            )
        ],
    )
    fake_group = type(
        "FakeGroup",
        (),
        {
            "normed_text": "hello world",
            "profiles": [profile],
            "duration_pool": [200, 500, 800],
            "num_operations": 3,
            "original_audio": b"orig",
            "metadata": {"language": "en", "sample_rate": 24000},
        },
    )()
    monkeypatch.setattr(adapter, "_generate_group", lambda *args, **kwargs: fake_group)

    ctx = AdapterContext(
        run_config={"pipeline_config": {**load_parameters("pipeline/pause_edit/defaults"), }},
        worker_id="worker-a",
        stage=StageSpec(
            name="plan",
            input_queue="in",
            output_queue="out",
            worker_python_bin=DEFAULT_STAGE_WORKER_PYTHON_BIN,
        ),
        services=direct_services(llm="http://llm"),
    )

    result = adapter.process("plan", payload, ctx)

    assert len(result.outputs) == 1
    output = result.outputs[0]
    assert output.queue_name == "out"
    assert output.item_key == "sample_001:profile_1"
    assert output.payload.sample_id == "sample_001"
    assert output.payload.profile.profile_id == 1


def test_stage_adapter_samples_profile_targets_from_weights(monkeypatch) -> None:
    adapter = PauseEditStageAdapter()
    captured = {}
    profile = PauseEditProfile(
        profile_id=1,
        operation='hello,<pause act="ins" level="3"/> world',
        st_inst={
            "increase_pause": ['hello,<pause act="ins" level="3"/> world'],
            "decrease_pause": ['hello,<pause act="red" level="3"/> world'],
        },
        nl_inst={"increase_pause": ["add a pause"], "decrease_pause": ["reduce a pause"]},
        tags=[
            PauseTag(
                char_idx=6,
                act="ins",
                level=3,
                delta_dur_ms=800,
                left_context="hello,",
                right_context="world",
                punctuation=",",
            )
        ],
    )

    def fake_generate_profiles_with_retries(**kwargs):
        captured["profile_targets"] = kwargs["profile_targets"]
        return [profile], "hello, world"

    monkeypatch.setattr(
        stage_adapter_module,
        "generate_profiles_with_retries",
        fake_generate_profiles_with_retries,
    )
    segment = type(
        "Segment",
        (),
        {
            "text": "hello, world, again, today",
            "duration": 1.0,
            "metadata": {},
            "clip_id": "clip_001",
            "sample_rate": 24000,
            "audio_bytes": b"audio",
        },
    )()
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/pause_edit/defaults"),
                "num_profiles": 1,
                "num_operation_weights": {1: 0.0, 2: 0.0, 3: 1.0},
                "boundary_type_weights": {"punct": 1.0, "nonpunct": 0.0},
                "pause_level_weights": {1: 0.0, 2: 0.0, 3: 1.0},
            }
        },
        worker_id="worker-a",
        stage=StageSpec(
            name="plan",
            input_queue="in",
            output_queue="out",
            worker_python_bin=DEFAULT_STAGE_WORKER_PYTHON_BIN,
        ),
        services=direct_services(llm="http://llm"),
    )

    group = adapter._generate_group(
        PauseSampleWorkItem(
            sample_id="sample_001",
            segment=segment,
            attempt_index=123,
        ),
        ctx,
        llm_client=object(),
    )

    target = captured["profile_targets"][0]
    assert target.num_operations == 3
    assert target.boundary_types == ["punct", "punct", "punct"]
    assert target.levels == [3, 3, 3]
    assert group.metadata["profile_targets"][0]["num_operations"] == 3


def test_stage_adapter_eval_uses_only_canonical_skip_keys(monkeypatch) -> None:
    adapter = PauseEditStageAdapter()
    profile = PauseEditProfile(
        profile_id=1,
        operation='hello<pause act="ins" level="2"/> world',
        st_inst={"increase_pause": ['hello<pause act="ins" level="2"/> world'], "decrease_pause": []},
        nl_inst={"increase_pause": ["add a pause"], "decrease_pause": []},
        tags=[
            PauseTag(
                char_idx=5,
                act="ins",
                level=2,
                delta_dur_ms=500,
                left_context="hello",
                right_context="world",
                punctuation=None,
            )
        ],
    )
    work_item = PauseProfileWorkItem(
        sample_id="sample_001",
        normed_text="hello world",
        profile=profile,
        duration_pool=[200, 500, 800],
        num_operations=1,
        original_audio=b"orig",
        metadata={"language": "en"},
    )
    payload = PauseSynthesizedWorkItem(
        sample_id="sample_001",
        work_item=work_item,
        synth_pair=SynthesizedPair(
            profile=profile,
            normed_text="hello world",
            generated_text="hello world",
            original_audio_bytes=b"orig",
            generated_audio_bytes=b"gen",
            original_duration_sec=1.0,
            generated_duration_sec=1.5,
        ),
    )
    captured = {}

    def fake_evaluate_pause_edit(**kwargs):
        captured.update(kwargs)
        return PauseEvalResult(
            profile_id=1,
            original_wer=0.0,
            generated_wer=0.0,
            content_passed=True,
            pause_measurements=[],
        )

    monkeypatch.setattr(
        "speech_edit.pipelines.pause.evaluator.evaluate_pause_edit",
        fake_evaluate_pause_edit,
    )

    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/pause_edit/defaults"),
                "evaluate": True,
                "skip_utmos_eval": True,
                "skip_spk_sim_eval": False,
                "skip_wdtw_eval": False,
                "skip_spk_sim": True,
                "skip_wdtw": True,
            },
            "services": {
                "asr": {"url": "http://asr"},
                "aligner": {"url": "http://aligner"},
                "wer": {"url": "http://wer"},
                "utmos": {"url": "http://utmos"},
                "spk_sim": {"url": "http://spk"},
                "wdtw": {"url": "http://wdtw"},
            },
        },
        worker_id="worker-a",
        stage=StageSpec(
            name="eval",
            input_queue="in",
            output_queue="out",
            worker_python_bin=DEFAULT_STAGE_WORKER_PYTHON_BIN,
        ),
        services=direct_services({
            "asr": "http://asr",
            "aligner": "http://aligner",
            "wer": "http://wer",
            "utmos": "http://utmos",
            "spk_sim": "http://spk",
            "wdtw": "http://wdtw",
        }),
    )

    result = adapter.process("eval", payload, ctx)

    assert len(result.outputs) == 1
    assert captured["utmos_client"] is None
    assert captured["spk_sim_client"] is not None
    assert captured["wdtw_client"] is not None
