from __future__ import annotations
from speech_edit.pipelines import load_parameters
PARAMETERS = load_parameters("pipeline/emotion_style_edit/defaults")


import io
import json
import pickle

import numpy as np
import pytest
import soundfile as sf

from speech_edit.runtime.models import (
    AdapterContext,
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
    StageSpec,
)
from tests.service_graph_fixtures import direct_services
from speech_edit.data.parquet.reader import Segment
from speech_edit.pipelines.emotion import evaluator, input_generator
from speech_edit.pipelines.emotion.parquet_rows import (
    EmotionStyleParquetRowProcessor,
)
from speech_edit.pipelines.emotion.stage_adapter import (
    EmotionStyleStageAdapter,
)
from speech_edit.pipelines.emotion.synthesizer import (
    AssembledEditPair,
    SynthesizedAudio,
    SynthesisResult,
    _align_edit_pair_outputs,
)
from speech_edit.pipelines.emotion.tag_utils import (
    EditPair,
    TaggedTextGroup,
)


def _stage_spec(name: str, input_queue: str | None, output_queue: str | None) -> StageSpec:
    return StageSpec(name, input_queue, output_queue, DEFAULT_STAGE_WORKER_PYTHON_BIN)


def _service_urls(stage: str) -> dict[str, str]:
    if stage == "plan":
        return {"llm": "http://llm"}
    if stage.startswith("synth"):
        services = {
            "boundary_pause_smoother": "http://smoother",
            "audio_assembly": "http://assembly",
            "aligner": "http://aligner",
        }
        if stage == "synth_index":
            services["index_tts"] = "http://index-tts"
        elif stage == "synth_ted":
            services["ted_tts"] = "http://ted-tts"
        return services
    if stage == "eval":
        return {
            "asr": "http://asr",
            "wer": "http://wer",
            "utmos": "http://utmos",
            "spk_sim": "http://spk-sim",
            "wdtw": "http://wdtw",
        }
    return {}


def _wav_bytes(duration_sec: float = 1.0) -> bytes:
    audio = np.zeros(int(24000 * duration_sec), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _group() -> TaggedTextGroup:
    return TaggedTextGroup(

        sub_type="single",
        plain_text="Hello world",
        speaker_prompt_audio=_wav_bytes(),
        speaker_prompt_text="Hello",
        tagged_texts={
            "src": "<emo type=\"calm\">Hello world</emo>",
            "tgt": "<emo type=\"happy\">Hello world</emo>",
        },
        edit_pairs=[
            EditPair(
                source_id="src",
                target_id="tgt",
                nl_instruction="make it happy",
                st_instruction="<emo type=\"happy\">Hello world</emo>",
            )
        ],
        boundaries=[],
        metadata={
            "language": "en",
            "duration_control_mode": "off",
            "prompted_edit_mode": "type_change",
            "pair_change_summary": {"type_change": 1},
            "speaker_prompt_metadata": {"source_parquet_path": "speaker.parquet"},
            "synthesis_profile": {
                "synthesis_backend": "index",
                "synthesis_mode": "single",
                "emotion_condition_mode": "type_level",
                "emotion_control_mode": "alpha",
                "duration_control_mode": "off",
                "estimated_duration_sec": 0.44,
                "selected_text_window": {
                    "text": "Hello world",
                    "estimated_duration_sec": 0.44,
                },
            },
            "synthesis_backend": "index",
            "synthesis_mode": "single",
        },
    )


def _segment(
    clip_id: str,
    text: str,
    duration: float,
    *,
    language: str = "en",
) -> Segment:
    return Segment(
        clip_id=clip_id,
        text=text,
        audio_bytes=_wav_bytes(duration),
        duration=duration,
        sample_rate=24000,
        metadata={"language": language, "source_parquet_path": "source.parquet"},
    )


def _synthesis_result(*, with_alignment: bool = True) -> SynthesisResult:
    source_audio = _wav_bytes()
    target_audio = _wav_bytes(1.1)
    source_alignment = {
        "backend": "qwen3",
        "level": "word",
        "alignments": [{"char": "Hello", "start": 0.0, "end": 0.4}],
    }
    target_alignment = {
        "backend": "qwen3",
        "level": "word",
        "alignments": [{"char": "Hello", "start": 0.0, "end": 0.4}],
    }
    source_timestamps = [
        {
            "text": "Hello world",
            "tag": "calm",
            "tag_type": "emotion",
            "alpha": 0.9,
            "start_sec": 0.0,
            "end_sec": 1.0,
        }
    ]
    target_timestamps = [
        {
            "text": "Hello world",
            "tag": "happy",
            "tag_type": "emotion",
            "alpha": 0.93,
            "start_sec": 0.0,
            "end_sec": 1.0,
        }
    ]
    pair = AssembledEditPair(
        source_id="src",
        target_id="tgt",
        nl_instruction="make it happy",
        st_instruction="<emo type=\"happy\">Hello world</emo>",
        source_audio=source_audio,
        target_audio=target_audio,
        source_duration=1.0,
        target_duration=1.1,
        changed_segments=[
            {
                "segment_idx": 0,
                "segment_text": "Hello world",
                "source_tag": "calm",
                "target_tag": "happy",
            }
        ],
        assembly_method="direct",
        source_alignment=source_alignment if with_alignment else {},
        target_alignment=target_alignment if with_alignment else {},
        source_segment_timestamps=source_timestamps if with_alignment else [],
        target_segment_timestamps=target_timestamps if with_alignment else [],
    )
    return SynthesisResult(
        synthesized_audios={
            "src": SynthesizedAudio(
                text_id="src",
                tagged_text="<emo type=\"calm\">Hello world</emo>",
                audio_bytes=source_audio,
                sample_rate=24000,
                duration_sec=1.0,
                segment_timestamps=[],
                synthesis_mode="single",
            ),
            "tgt": SynthesizedAudio(
                text_id="tgt",
                tagged_text="<emo type=\"happy\">Hello world</emo>",
                audio_bytes=target_audio,
                sample_rate=24000,
                duration_sec=1.1,
                segment_timestamps=[],
                synthesis_mode="single",
            ),
        },
        edit_pairs=[pair],
        timing={},
    )


def _ted_description_synthesis_result() -> SynthesisResult:
    source_audio = _wav_bytes()
    target_audio = _wav_bytes(1.1)
    source_timestamps = [
        {
            "text": "Hello world",
            "tag": "calm",
            "tag_type": "emotion",
            "emotion_control_mode": "description",
            "emotion_type_ref": "calm",
            "start_sec": 0.0,
            "end_sec": 1.0,
        }
    ]
    target_timestamps = [
        {
            "text": "Hello world",
            "tag": "sad",
            "tag_type": "emotion",
            "emotion_control_mode": "description",
            "emotion_type_ref": "sad",
            "start_sec": 0.0,
            "end_sec": 1.0,
        }
    ]
    pair = AssembledEditPair(
        source_id="src",
        target_id="tgt",
        nl_instruction="make it sad",
        st_instruction='<emo desc="soft sadness" type_ref="sad">Hello world</emo>',
        source_audio=source_audio,
        target_audio=target_audio,
        source_duration=1.0,
        target_duration=1.1,
        changed_segments=[
            {
                "segment_idx": 0,
                "segment_text": "Hello world",
                "source_tag": "calm",
                "target_tag": "sad",
                "target_emotion_control_mode": "description",
                "target_emotion_type_ref": "sad",
            }
        ],
        assembly_method="direct",
        source_segment_timestamps=source_timestamps,
        target_segment_timestamps=target_timestamps,
    )
    return SynthesisResult(
        synthesized_audios={
            "src": SynthesizedAudio(
                text_id="src",
                tagged_text='<emo desc="calm delivery" type_ref="calm">Hello world</emo>',
                audio_bytes=source_audio,
                sample_rate=24000,
                duration_sec=1.0,
                segment_timestamps=[],
                synthesis_mode="single_ted",
                duration_control={"enabled": False, "backend": "ted"},
            ),
            "tgt": SynthesizedAudio(
                text_id="tgt",
                tagged_text='<emo desc="soft sadness" type_ref="sad">Hello world</emo>',
                audio_bytes=target_audio,
                sample_rate=24000,
                duration_sec=1.1,
                segment_timestamps=[],
                synthesis_mode="single_ted",
                duration_control={"enabled": False, "backend": "ted"},
            ),
        },
        edit_pairs=[pair],
        timing={},
    )


def test_emotion_stage_adapter_plan_does_not_pass_llm_backend(monkeypatch):
    captured = {}

    class FakeLLMClient:
        def __init__(self, url):
            captured["llm_url"] = url

    def fake_generate(**kwargs):
        captured["generate_kwargs"] = kwargs
        return _group()

    monkeypatch.setattr(
        "speech_edit.clients.services.LLMClient",
        FakeLLMClient,
    )
    monkeypatch.setattr(
        input_generator,
        "generate_tagged_text_group_attempt",
        fake_generate,
    )

    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "natural_language_instruction_mode": "template",
                "llm_backend": "must_not_be_forwarded",
                "llm_model": "must_not_be_forwarded",
            }
        },
        worker_id="worker-a",
        stage=_stage_spec("plan", "in", None),
        services=direct_services(_service_urls("plan")),
    )
    result = EmotionStyleStageAdapter().process(
        "plan",
        {
            "sample_id": "sample_1",
            "speaker_prompt": object(),
            "gen_segments": [],
            "language": "en",
            "stage1_seed": 123,
            "synthesis_profile": input_generator.normalize_synthesis_profile({"synthesis_mode": "single", "emotion_condition_mode": "type_level", "ted_estimated_duration_soft_limit_sec": PARAMETERS["ted_estimated_duration_soft_limit_sec"], "ted_max_generated_duration_sec": PARAMETERS["ted_max_generated_duration_sec"]}),
        },
        ctx,
    )

    kwargs = captured["generate_kwargs"]
    assert "llm_backend" not in kwargs
    assert "llm_model" not in kwargs
    assert kwargs["natural_language_instruction_mode"] == "template"
    assert kwargs["attempt"] == 1
    assert kwargs["validation_errors"] == ()
    assert result.outputs[0].queue_name == "emotion.plan_to_synth.index"


def test_emotion_stage_adapter_always_runs_pair_acoustic_eval(monkeypatch):
    captured = {}

    def fake_evaluate(synth_result, **kwargs):
        captured.update(kwargs)
        return {"pair_evaluations": []}

    monkeypatch.setattr(evaluator, "evaluate_synthesis_result", fake_evaluate)

    group = _group()
    group.metadata["synthesis_profile"]["synthesis_backend"] = "ted"
    group.metadata["synthesis_backend"] = "ted"
    ctx = AdapterContext(
        run_config={"pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults")}},
        worker_id="worker-a",
        stage=_stage_spec("eval", "in", "explicit.eval_to_save"),
        services=direct_services(_service_urls("eval")),
    )

    EmotionStyleStageAdapter().process(
        "eval",
        {
            "sample_id": "sample_1",
            "group": group,
            "synth_result": _ted_description_synthesis_result(),
        },
        ctx,
    )

    assert "skip_emotion" not in captured
    assert captured["wer_url"] == "http://wer"


def test_emotion_stage_adapter_plan_retries_with_validation_feedback(monkeypatch):
    captured = []

    class FakeLLMClient:
        def __init__(self, url):
            pass

    def fake_generate(**kwargs):
        captured.append(kwargs)
        if kwargs["attempt"] == 1:
            raise ValueError("invalid duration")
        return _group()

    monkeypatch.setattr("speech_edit.clients.services.LLMClient", FakeLLMClient)
    monkeypatch.setattr(
        input_generator,
        "generate_tagged_text_group_attempt",
        fake_generate,
    )
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "natural_language_instruction_mode": "template",
                "llm_max_retries": 2,
            }
        },
        worker_id="worker-a",
        stage=_stage_spec("plan", "in", "emotion.plan_to_synth"),
        services=direct_services(_service_urls("plan")),
    )

    result = EmotionStyleStageAdapter().process(
        "plan",
        {
            "sample_id": "sample_1",
            "speaker_prompt": object(),
            "gen_segments": [],
            "language": "en",
            "stage1_seed": 123,
            "synthesis_profile": input_generator.normalize_synthesis_profile({"synthesis_mode": "single", "emotion_condition_mode": "type_level", "ted_estimated_duration_soft_limit_sec": PARAMETERS["ted_estimated_duration_soft_limit_sec"], "ted_max_generated_duration_sec": PARAMETERS["ted_max_generated_duration_sec"]}),
        },
        ctx,
    )

    assert [call["attempt"] for call in captured] == [1, 2]
    assert captured[0]["validation_errors"] == ()
    assert "invalid duration" in captured[1]["validation_errors"][0]
    assert result.outputs[0].item_key == "sample_1"
    assert result.outputs[0].queue_name == "emotion.plan_to_synth.index"


def test_emotion_stage_adapter_plan_routes_by_derived_synthesis_backend(monkeypatch):
    class FakeLLMClient:
        def __init__(self, url):
            pass

    def fake_generate(**kwargs):
        group = _group()
        group.metadata["synthesis_profile"]["synthesis_backend"] = "ted"
        group.metadata["synthesis_profile"]["synthesis_mode"] = "multi_emo_frequent"
        group.metadata["synthesis_profile"]["emotion_condition_mode"] = "desc"
        group.metadata["synthesis_backend"] = "ted"
        group.metadata["synthesis_mode"] = "multi_emo_frequent"
        return group

    monkeypatch.setattr("speech_edit.clients.services.LLMClient", FakeLLMClient)
    monkeypatch.setattr(
        input_generator,
        "generate_tagged_text_group_attempt",
        fake_generate,
    )
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "natural_language_instruction_mode": "template",
                "llm_max_retries": 1,
            }
        },
        worker_id="worker-a",
        stage=_stage_spec("plan", "in", None),
        services=direct_services(_service_urls("plan")),
    )

    result = EmotionStyleStageAdapter().process(
        "plan",
        {
            "sample_id": "sample_ted",
            "speaker_prompt": object(),
            "gen_segments": [],
            "language": "en",
            "stage1_seed": 123,
            "synthesis_profile": input_generator.normalize_synthesis_profile({"synthesis_mode": "single", "emotion_condition_mode": "type_level", "ted_estimated_duration_soft_limit_sec": PARAMETERS["ted_estimated_duration_soft_limit_sec"], "ted_max_generated_duration_sec": PARAMETERS["ted_max_generated_duration_sec"]}),
        },
        ctx,
    )

    assert result.outputs[0].queue_name == "emotion.plan_to_synth.ted"
    assert result.outputs[0].item_key == "sample_ted"


def test_emotion_stage_adapter_synth_backend_filter_rejects_wrong_queue(monkeypatch):
    def fake_synthesize(*args, **kwargs):
        raise AssertionError("wrong-backend payload should be rejected before synthesis")

    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.synthesizer.synthesize_tagged_text_group",
        fake_synthesize,
    )
    group = _group()
    group.metadata["synthesis_profile"]["synthesis_backend"] = "ted"
    group.metadata["synthesis_backend"] = "ted"
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"), },
            "stages": {
                "synth_index": {"pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"), "synthesis_backend_filter": "index"}}
            },
        },
        worker_id="worker-synth",
        stage=_stage_spec("synth_index", "emotion.plan_to_synth.index", "emotion.synth_to_eval"),
        services=direct_services(_service_urls("synth_index")),
    )

    with pytest.raises(ValueError, match="only accepts backend=index"):
        EmotionStyleStageAdapter().process(
            "synth_index",
            {"sample_id": "sample_ted", "group": group},
            ctx,
        )


def test_emotion_stage_adapter_synth_backend_filter_accepts_matching_backend(monkeypatch):
    captured = {}

    def fake_synthesize(group, **kwargs):
        captured["group"] = group
        captured["kwargs"] = kwargs
        return _synthesis_result()

    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.synthesizer.synthesize_tagged_text_group",
        fake_synthesize,
    )
    group = _group()
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"), },
            "stages": {
                "synth_index": {"pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"), "synthesis_backend_filter": "index"}}
            },
        },
        worker_id="worker-synth",
        stage=_stage_spec("synth_index", "emotion.plan_to_synth.index", "emotion.synth_to_eval"),
        services=direct_services(_service_urls("synth_index")),
    )

    result = EmotionStyleStageAdapter().process(
        "synth_index",
        {"sample_id": "sample_index", "group": group},
        ctx,
    )

    assert captured["group"] is group
    assert result.outputs[0].queue_name == "emotion.synth_to_eval"
    assert result.outputs[0].payload["sample_id"] == "sample_index"


def test_emotion_stage_adapter_sample_uses_shortest_segment_without_speaker_source(
    monkeypatch,
):
    segments = [
        _segment("long", "long text", 4.0),
        _segment("short", "short prompt", 2.0),
        _segment("mid", "mid text", 3.0),
    ]
    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.stage_adapter._segments_from_parquet_payload",
        lambda payload, cfg: segments,
    )
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "utterance_batch_size": 3,
                "seed": 11,
            }
        },
        worker_id="worker-a",
        stage=_stage_spec("sample", "source_items", "emotion.sample_to_plan"),
        services=direct_services(_service_urls("sample")),
    )

    result = EmotionStyleStageAdapter().process("sample", {"parquet_path": "x"}, ctx)

    assert len(result.outputs) == 1
    payload = result.outputs[0].payload
    assert payload["speaker_prompt"].clip_id == "short"
    assert payload["gen_segments"]
    assert all(
        segment.clip_id.split("#", 1)[0] in {"long", "mid"}
        for segment in payload["gen_segments"]
    )
    assert payload["synthesis_profile"]["synthesis_backend"] in {"index", "ted"}
    assert payload["synthesis_profile"]["emotion_condition_mode"] in {
        "type_level",
        "desc",
    }
    assert payload["synthesis_profile"]["synthesis_mode"] in {
        "single",
        "multi_emo_normal",
        "multi_emo_frequent",
    }
    assert payload["language"] == "en"


def test_emotion_stage_adapter_sample_uses_language_speaker_source(monkeypatch):
    gen_segments = [
        _segment("gen1", "hello", 2.5, language="en"),
        _segment("gen2", "there", 2.6, language="en"),
        _segment("gen3", "friend", 2.7, language="en"),
    ]
    speaker = _segment("speaker", "speaker prompt", 2.0, language="en")

    class FakeSpeakerSampler:
        def sample(self):
            return speaker

    captured = {}

    def fake_init(sources, *, seed, log_prefix=""):
        captured["sources"] = sources
        captured["seed"] = seed
        return {"en": FakeSpeakerSampler()}

    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.stage_adapter._segments_from_parquet_payload",
        lambda payload, cfg: gen_segments,
    )
    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.stage_adapter.initialize_language_speaker_samplers",
        fake_init,
    )
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "utterance_batch_size": 2,
                "speaker_parquet_sources": ["speaker_timbre_library/en/parquet_files.list"],
                "seed": 23,
            }
        },
        worker_id="worker-b",
        stage=_stage_spec("sample", "source_items", "emotion.sample_to_plan"),
        services=direct_services(_service_urls("sample")),
    )

    result = EmotionStyleStageAdapter().process("sample", {"parquet_path": "x"}, ctx)

    assert captured == {
        "sources": ["speaker_timbre_library/en/parquet_files.list"],
        "seed": 23,
    }
    payload = result.outputs[0].payload
    assert payload["speaker_prompt"].clip_id == "speaker"
    assert [segment.clip_id.split("#", 1)[0] for segment in payload["gen_segments"]] == [
        "gen1"
    ]
    assert "synthesis_profile" in payload
    assert payload["language"] == "en"
    assert [output.payload["speaker_prompt"].clip_id for output in result.outputs] == [
        "speaker",
        "speaker",
        "speaker",
    ]
    assert len({output.item_key for output in result.outputs}) == 3
    assert len({output.payload["sample_id"] for output in result.outputs}) == 3


def test_emotion_stage_adapter_sample_keeps_unique_keys_with_cyclic_speaker(
    monkeypatch,
):
    gen_segments = [
        _segment("gen1", "hello", 2.5, language="en"),
        _segment("gen2", "there", 2.6, language="en"),
    ]
    speaker = _segment("speaker", "speaker prompt", 2.0, language="en")

    class FakeSpeakerSampler:
        def sample(self):
            return speaker

    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.stage_adapter._segments_from_parquet_payload",
        lambda payload, cfg: gen_segments,
    )
    monkeypatch.setattr(
        "speech_edit.pipelines.emotion.stage_adapter.initialize_language_speaker_samplers",
        lambda sources, *, seed, log_prefix="": {"en": FakeSpeakerSampler()},
    )
    ctx = AdapterContext(
        run_config={
            "pipeline_config": {**load_parameters("pipeline/emotion_style_edit/defaults"),
                "utterance_batch_size": 2,
                "speaker_parquet_sources": "speaker_timbre_library/en/parquet_files.list",
                "seed": 23,
            }
        },
        worker_id="worker-b",
        stage=_stage_spec("sample", "source_items", "emotion.sample_to_plan"),
        services=direct_services(_service_urls("sample")),
    )

    result = EmotionStyleStageAdapter().process("sample", {"parquet_path": "x"}, ctx)

    assert [output.payload["speaker_prompt"].clip_id for output in result.outputs] == [
        "speaker",
        "speaker",
    ]
    assert len({output.item_key for output in result.outputs}) == 2
    assert len({output.payload["sample_id"] for output in result.outputs}) == 2


def test_template_instruction_mode_replaces_llm_nl_and_st():
    pairs = input_generator._build_edit_pairs_from_diff(
        _group().tagged_texts,
        {"nl_inst": {"src->tgt": "llm text"}, "st_inst": {"src->tgt": "llm st"}},
        natural_language_instruction_mode="template",
    )
    pairs = [pair for pair in pairs if pair.source_id == "src" and pair.target_id == "tgt"]

    assert pairs[0].nl_instruction.startswith("Please change")
    assert pairs[0].st_instruction == '<emo type="happy" alpha="0.93">Hello world</emo>'


def test_synth_alignment_cache_uses_qwen_word_level_batch():
    class FakeAligner:
        def __init__(self):
            self.calls = []

        def align_batch(self, items, backend="auto"):
            self.calls.append({"items": items, "backend": backend})
            assert backend == "qwen3"
            assert len(items) == 2
            assert all(item["word_level"] is True for item in items)
            assert all("separate_punctuation" not in item for item in items)
            return [
                {
                    "backend": "qwen3",
                    "level": "word",
                    "alignments": [{"char": "Hello", "start": 0.0, "end": 0.4}],
                    "answers": [{"start": 0.0, "end": 1.0}],
                },
                {
                    "backend": "qwen3",
                    "level": "word",
                    "alignments": [{"char": "Hello", "start": 0.0, "end": 0.4}],
                    "answers": [{"start": 0.0, "end": 1.0}],
                },
            ]

    synth_result = _synthesis_result(with_alignment=False)
    pair = synth_result.edit_pairs[0]
    cache = _align_edit_pair_outputs(
        edit_pairs=synth_result.edit_pairs,
        synthesized=synth_result.synthesized_audios,
        aligner_client=FakeAligner(),
        aligner_backend="qwen3",
        aligner_language="eng",
    )

    assert pair.source_alignment["level"] == "word"
    assert pair.target_alignment["level"] == "word"
    assert pair.source_segment_timestamps[0]["tag"] == "calm"
    assert pair.target_segment_timestamps[0]["tag"] == "happy"
    assert cache["pairs"]["src->tgt"]["source"]["alignment"]["backend"] == "qwen3"
    queued_result = pickle.loads(pickle.dumps(synth_result))
    queued_pair = queued_result.edit_pairs[0]
    assert queued_pair.source_alignment == pair.source_alignment
    assert queued_pair.target_alignment == pair.target_alignment
    assert queued_pair.source_segment_timestamps == pair.source_segment_timestamps
    assert queued_pair.target_segment_timestamps == pair.target_segment_timestamps


@pytest.fixture
def wer_service(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    client = SimpleNamespace(compute=Mock(return_value={"error_rate": 1.25, "metric": "mixed"}))
    factory = Mock(return_value=client)
    monkeypatch.setattr(evaluator, "WERClient", factory)
    return factory, client


def test_eval_reuses_alignment_and_supplies_audio_for_wdtw_f0(monkeypatch, wer_service):
    calls = []

    class FakeASR:
        def __init__(self, url):
            pass

        def transcribe_batch(self, audio_list, languages=None, timeout=300):
            return [{"text": "Hello world"} for _ in audio_list]

    class FakeUTMOS:
        def __init__(self, url):
            pass

        def predict(self, audio_bytes):
            return {"score": 4.0, "duration_seconds": 1.0}

    class FakeSpkSim:
        def __init__(self, url):
            pass

        def similarity(self, source_audio, target_audio, timeout=30):
            return {"similarity": 0.9, "inference_time_seconds": 0.1}

    class FakeWDTW:
        def __init__(self, url):
            pass

        def compute(self, **kwargs):
            calls.append(kwargs)
            return {"wdtw": 0.25, "mode": "preserved"}

    monkeypatch.setattr(evaluator, "ASRClient", FakeASR)
    monkeypatch.setattr(evaluator, "UTMOSClient", FakeUTMOS)
    monkeypatch.setattr(evaluator, "SpkSimClient", FakeSpkSim)
    monkeypatch.setattr(evaluator, "WDTWClient", FakeWDTW)

    synthesis = _synthesis_result()
    result = evaluator.evaluate_synthesis_result(
        synthesis,
        wer_url="http://wer.test",
        wdtw_url="http://wdtw.test",
        aligner_language="en",
        parallel=False,
    )

    pair_eval = result["pair_evaluations"][0]["eval_results"]
    assert pair_eval["wdtw"] == 0.25
    wer_factory, wer_client = wer_service
    wer_factory.assert_called_once_with("http://wer.test")
    wer_client.compute.assert_called_with("Hello world", "Hello world", language="auto")
    assert pair_eval["src"]["wer"]["backend"] == "wer_service"
    assert pair_eval["src"]["wer"]["error_rate"] == 1.25
    assert pair_eval["wdtw_result"]["mode"] == "preserved"
    assert calls[0]["source_alignments"] == [
        {"char": "Hello", "start": 0.0, "end": 0.4}
    ]
    assert calls[0]["target_alignments"] == [
        {"char": "Hello", "start": 0.0, "end": 0.4}
    ]
    assert calls[0]["source_audio"] == synthesis.edit_pairs[0].source_audio
    assert calls[0]["target_audio"] == synthesis.edit_pairs[0].target_audio
    assert "aligner_url" not in calls[0]


def test_eval_uses_cached_alignment_for_pair_acoustic_effect(monkeypatch, wer_service):
    acoustic_calls = []

    class FakeASR:
        def __init__(self, url):
            pass

        def transcribe_batch(self, audio_list, languages=None, timeout=300):
            return [{"text": "Hello world"} for _ in audio_list]

    monkeypatch.setattr(evaluator, "ASRClient", FakeASR)

    def fake_acoustic_effect(**kwargs):
        acoustic_calls.append(kwargs)
        return {"version": "v1", "status": "ok", "score": 0.75, "regions": []}

    monkeypatch.setattr(
        evaluator,
        "compute_emotion_acoustic_effect",
        fake_acoustic_effect,
    )

    result = evaluator.evaluate_synthesis_result(
        _synthesis_result(),
        wer_url="http://wer.test",
        skip_quality=True,
        skip_spk_sim=True,
        skip_wdtw=True,
        parallel=False,
    )

    assert len(acoustic_calls) == 1
    assert acoustic_calls[0]["source_alignment"]["level"] == "word"
    assert acoustic_calls[0]["target_alignment"]["level"] == "word"
    assert "target_tag" not in acoustic_calls[0]
    pair_eval = result["pair_evaluations"][0]["eval_results"]
    assert pair_eval["version"] == "v3"
    assert pair_eval["emotion_acoustic_effect_v1"]["score"] == 0.75
    assert "emotion" not in pair_eval["src"]
    assert "emotion" not in pair_eval["tgt"]


def test_eval_records_structured_acoustic_error_without_classifier_fields(
    monkeypatch, wer_service,
):
    class FakeASR:
        def __init__(self, url):
            pass

        def transcribe_batch(self, audio_list, languages=None, timeout=300):
            return [{"text": "Hello world"} for _ in audio_list]

    monkeypatch.setattr(evaluator, "ASRClient", FakeASR)
    monkeypatch.setattr(
        evaluator,
        "compute_emotion_acoustic_effect",
        lambda **kwargs: {
            "version": "v1",
            "status": "error",
            "error": "alignment token mismatch",
            "regions": [],
        },
    )

    result = evaluator.evaluate_synthesis_result(
        _synthesis_result(),
        wer_url="http://wer.test",
        skip_quality=True,
        skip_spk_sim=True,
        skip_wdtw=True,
        parallel=False,
    )

    payload = result["pair_evaluations"][0]["eval_results"]
    assert payload["emotion_acoustic_effect_v1"]["status"] == "error"
    assert payload["emotion_acoustic_effect_v1"]["error"] == (
        "alignment token mismatch"
    )
    serialized = json.dumps(payload)
    assert "clsp" not in serialized.lower()
    assert "whisper_ser" not in serialized.lower()


def test_eval_rejects_missing_synthesis_alignment(monkeypatch, wer_service):
    calls = []

    class FakeASR:
        def __init__(self, url):
            pass

        def transcribe_batch(self, audio_list, languages=None, timeout=300):
            return [{"text": "Hello world"} for _ in audio_list]

    class FakeWDTW:
        def __init__(self, url):
            pass

        def compute(self, **kwargs):
            calls.append(kwargs)
            return {"wdtw": 0.5, "mode": "preserved"}

    monkeypatch.setattr(evaluator, "ASRClient", FakeASR)
    monkeypatch.setattr(evaluator, "WDTWClient", FakeWDTW)
    with pytest.raises(ValueError, match="Missing cached segment timestamps"):
        result = evaluator.evaluate_synthesis_result(
            _synthesis_result(with_alignment=False),
            wer_url="http://wer.test",
            wdtw_url="http://wdtw.test",
            skip_quality=True,
            skip_spk_sim=True,
            parallel=False,
        )
    assert calls == []


def test_emotion_parquet_rows_write_compact_v3_eval_and_raw_alignment():
    group = _group()
    group.metadata["speaker_prompt_metadata"]["source_file_idx"] = np.int64(7)
    eval_result = {
        "pair_evaluations": [
            {
                "source_id": "src",
                "target_id": "tgt",
                "eval_results": {
                    "src": {
                        "transcript": "Hello world",
                        "emotion": {"clsp_distribution": {"happy": 1.0}},
                        "wer": {"error_rate": 0.0},
                        "edit_time_spans": [
                            {"text": "Hello world", "span": [0.0, 1.0]}
                        ],
                    },
                    "tgt": {
                        "transcript": "Hello world",
                        "wer": {"error_rate": 0.0},
                        "edit_time_spans": [
                            {"text": "Hello world", "span": [0.0, 1.0]}
                        ],
                    },
                    "emotion_acoustic_effect_v1": {
                        "version": "v1",
                        "status": "ok",
                        "score": 0.8,
                        "regions": [],
                    },
                    "spk_sim": {"similarity": 0.9},
                    "wdtw": 0.25,
                    "wdtw_result": {"wdtw": 0.25},
                    "eval_time_sec": 1.0,
                    "overall_pass": True,
                    "source_id": "legacy-src",
                },
            }
        ]
    }
    rows = EmotionStyleParquetRowProcessor().rows_from_payload(
        {
            "group": group,
            "synth_result": _synthesis_result(),
            "eval_result": eval_result,
            "sample_id": "sample_1",
        },
        dataset_name="Emilia-Emotion-Test",
    )

    eval_payload = json.loads(rows[0]["eval_results"])
    assert eval_payload["version"] == "v3"
    assert eval_payload["emotion_acoustic_effect_v1"]["status"] == "ok"
    assert "emotion" not in eval_payload["src"]
    assert "emotion" not in eval_payload["tgt"]
    assert "source_id" not in eval_payload
    assert "target_id" not in eval_payload
    assert "eval_results" not in eval_payload
    assert "overall_pass" not in eval_payload
    assert eval_payload["spk_sim"] == {"similarity": 0.9}
    assert eval_payload["wdtw_result"] == {"wdtw": 0.25}
    assert eval_payload["wdtw"] == 0.25
    segments = json.loads(rows[0]["segments"])
    for segment in segments:
        assert segment["edit_time_spans"] == [{"text": "Hello world", "span": [0.0, 1.0]}]
    assert segments[0]["alignment"]["backend"] == "qwen3"
    assert segments[0]["alignment"]["alignments"][0]["char"] == "Hello"
    pair_metadata = json.loads(rows[0]["pair_metadata"])
    assert pair_metadata["sample_id"] == "sample_1"
    assert pair_metadata["language"] == "en"
    assert pair_metadata["synthesis_backend"] == "index"
    assert pair_metadata["synthesis_mode"] == "single"
    assert pair_metadata["emotion_condition_mode"] == "type_level"
    assert pair_metadata["prompted_edit_mode"] == "type_change"
    assert pair_metadata["estimated_duration_sec"] == 0.44
    assert pair_metadata["selected_text_window"]["text"] == "Hello world"
    assert pair_metadata["pair_change_summary"] == {"type_change": 1}
    assert pair_metadata["speaker_prompt_metadata"]["source_parquet_path"] == "speaker.parquet"
    assert pair_metadata["speaker_prompt_metadata"]["source_file_idx"] == 7


@pytest.mark.parametrize("parallel", [False, True])
def test_emotion_wer_service_failure_propagates(monkeypatch, wer_service, parallel):
    from types import SimpleNamespace

    monkeypatch.setattr(evaluator, "ASRClient", lambda _: SimpleNamespace(
        transcribe_batch=lambda audio_list, **kwargs: [{"text": "Hello world"} for _ in audio_list]
    ))
    _, client = wer_service
    client.compute.side_effect = RuntimeError("WER service unavailable")
    with pytest.raises(RuntimeError, match="WER service unavailable"):
        evaluator.evaluate_synthesis_result(
            _synthesis_result(), wer_url="http://wer.test",
            skip_quality=True, skip_spk_sim=True, skip_wdtw=True, parallel=parallel,
        )
    assert client.compute.called
