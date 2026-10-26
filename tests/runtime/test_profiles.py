from __future__ import annotations

from pathlib import Path

import pytest

from speech_edit.runtime.config import load_run_config
from speech_edit.runtime.models import (
    AdapterContext,
    DEFAULT_STAGE_WORKER_PYTHON_BIN,
)
from speech_edit.runtime.registry import get_adapter
from speech_edit.runtime.services.resolver import resolve_service_graph
from speech_edit.runtime.services.single_node_workers import materialize_worker_service_graph
from speech_edit.runtime.submit_run import prepare_run_config


TEXT_EDIT_DEFAULTS = Path(
    "configs/pipeline/text_edit/defaults.yaml"
)
PIPELINE_DEFAULTS = Path("configs/pipeline/defaults.yaml")
PROSODY_DEFAULTS = Path(
    "configs/pipeline/prosody_edit/"
    "defaults.yaml"
)
EMOTION_DEFAULTS = Path(
    "configs/pipeline/emotion_style_edit/"
    "defaults.yaml"
)
PAUSE_DEFAULTS = Path(
    "configs/pipeline/pause_edit/defaults.yaml"
)


def _resolved(stage_spec):
    return resolve_service_graph(stage_spec.services)


@pytest.mark.parametrize(
    "config_path",
    [TEXT_EDIT_DEFAULTS, EMOTION_DEFAULTS, PROSODY_DEFAULTS, PAUSE_DEFAULTS],
)
def test_distributed_eval_keeps_qwen3_asr_ephemeral_compile_cache(
    config_path: Path,
):
    loaded = load_run_config(config_path)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
    graph = _resolved(specs["eval"])

    gateway = graph.roots["asr"]
    qwen = gateway.dependencies["qwen3_asr"]
    assert gateway.startup_timeout_sec == 1200
    assert "--backend-startup-timeout 1200" in gateway.command
    assert qwen.startup_timeout_sec == 1200
    from speech_edit.runtime.paths import get_artifact_paths
    assert str(get_artifact_paths().worktree / "cache/qwen3_asr") in qwen.command
    assert "mktemp -d" in qwen.command
    assert not any(name.startswith("ASR_VLLM_") for name in qwen.env)
    assert 'trap cleanup EXIT' in qwen.command
    assert 'rm -rf "$cache_root"' in qwen.command
    assert "compile_cache.py" not in qwen.command
    assert "checkpoints/runtime-cache/qwen3-asr" not in qwen.command


@pytest.mark.parametrize(
    ("config_path", "aligner_is_direct"),
    [
        (TEXT_EDIT_DEFAULTS, False),
        (EMOTION_DEFAULTS, False),
        (PROSODY_DEFAULTS, False),
        (PAUSE_DEFAULTS, True),
    ],
)
def test_eval_stage_resolves_wdtw_dependency_without_leaking_it_to_worker(
    config_path: Path,
    aligner_is_direct: bool,
):
    loaded = load_run_config(config_path)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
    graph = _resolved(specs["eval"])

    wdtw = graph.roots["wdtw"]
    nested_aligner = wdtw.dependencies["aligner"]
    nested_f0 = wdtw.dependencies["f0"]
    assert f"--aligner-url {nested_aligner.url}" in wdtw.command
    assert "--aligner-backend qwen3" in wdtw.command
    assert f"--f0-url {nested_f0.url}" in wdtw.command
    assert ("aligner" in graph.direct_services) is aligner_is_direct
    assert [spec["instance_key"] for spec in graph.command_specs()].count(
        nested_aligner.instance_key
    ) == 1
    assert [spec["instance_key"] for spec in graph.command_specs()].count(
        nested_f0.instance_key
    ) == 1


def test_dist_pipeline_source_configs_use_plural_producers():
    for config_path in (TEXT_EDIT_DEFAULTS, PROSODY_DEFAULTS, EMOTION_DEFAULTS):
        text = config_path.read_text(encoding="utf-8")
        assert "\n      producer:" not in text
        assert "\n      producers:" in text


def test_pipeline_config_files_do_not_embed_runtime_commands():
    for path in Path("configs/pipeline").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "service_profiles" not in text
        assert "stage_profiles" not in text
        assert "command_template" not in text
        assert "wait_urls" not in text
        assert "service_urls" not in text


@pytest.mark.parametrize("config_path", sorted(Path("configs/pipeline").glob("*/*.yaml")))
def test_run_config_entries_are_complete(config_path: Path) -> None:
    loaded = load_run_config(config_path)
    adapter = get_adapter(loaded["pipeline"], loaded)
    specs = adapter.stage_specs(loaded)
    assert specs["sample"].input_queue == loaded["source"]["queue_name"]
    assert specs["save"].parquet_output.enabled
    assert loaded["source"]["inputs"]


def test_service_config_files_do_not_use_custom_placeholders():
    forbidden = (
        "{project_root}",
        "{run_dir}",
        "{gpu_ids}",
        "{dependencies.",
        "{port}",
        "{model}",
        "{max_",
    )
    for path in Path("configs/service").glob("**/*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "command_template" not in text
        if "command:" in text:
            assert "python_bin:" in text, f"{path} command must declare python_bin"
            assert "${.python_bin}" in text, f"{path} command must start from python_bin"
        for token in forbidden:
            assert token not in text, f"{path} contains legacy placeholder {token!r}"


def test_runtime_config_parser_preserves_composed_yaml_values():
    config_text = Path("speech_edit/runtime/config.py").read_text(
        encoding="utf-8"
    )
    adapter_text = Path("speech_edit/runtime/adapters.py").read_text(
        encoding="utf-8"
    )

    assert "apply_stage_overrides" not in config_text
    assert "apply_stage_overrides" not in adapter_text
    assert "_fill_default_parquet_output" not in config_text
    assert "_default_parquet_output_dir" not in config_text
    assert "_default_parquet_dataset_name" not in config_text
    assert "output_dataset_name" not in config_text
    assert "flush_interval_sec) * 2.0" not in config_text


def test_prepare_run_config_resolves_service_log_root_to_effective_run_dir(tmp_path):
    run_dir = tmp_path / "prosody-run"
    loaded, _ = prepare_run_config(
        run_config_path=str(PROSODY_DEFAULTS),
        run_id="cfgref",
        run_dir=str(run_dir),
    )
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
    command = _resolved(specs["plan"]).command_specs()[0]["command"]

    assert loaded["runtime"]["service_log_root"] == f"{run_dir.resolve()}/logs"
    assert f"--gemma4-mtp-log-dir {run_dir.resolve()}/logs/llm_gemma_mtp" in command


def test_pipeline_configs_inherit_shared_pipeline_defaults():
    shared = PIPELINE_DEFAULTS.read_text(encoding="utf-8")
    assert "worker_python_bin:" not in shared
    assert "run_dir: ${artifact:worktree}/distributed_pipeline_runs/default_run_name" in shared
    assert "service_log_root: ${runtime.run_dir}/logs" in shared
    save_stage = Path(
        "configs/stage/save/parquet.yaml"
    ).read_text(encoding="utf-8")
    stage_defaults = Path(
        "configs/stage/defaults.yaml"
    ).read_text(encoding="utf-8")
    assert f"worker_python_bin: {DEFAULT_STAGE_WORKER_PYTHON_BIN}" in stage_defaults
    assert "lease_seconds: 1200" in save_stage
    assert "output_dir: ${runtime.run_dir}/parquet" in save_stage
    assert "dataset_name: ${pipeline_config.output_dataset_name}" in save_stage

    text_defaults = Path(
        "configs/pipeline/text_edit/defaults.yaml"
    ).read_text(encoding="utf-8")
    prosody_defaults = PROSODY_DEFAULTS.read_text(encoding="utf-8")
    emotion_defaults = EMOTION_DEFAULTS.read_text(encoding="utf-8")
    assert "/pipeline/defaults@_global_" in text_defaults
    assert "/pipeline/defaults@_global_" in prosody_defaults
    assert "/pipeline/defaults@_global_" in emotion_defaults

    text_loaded = load_run_config(TEXT_EDIT_DEFAULTS)
    prosody_loaded = load_run_config(PROSODY_DEFAULTS)
    emotion_loaded = load_run_config(EMOTION_DEFAULTS)
    assert "worker_python_bin" not in text_loaded
    for loaded in (text_loaded, prosody_loaded, emotion_loaded):
        specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
        assert specs
        assert all(
            spec.worker_python_bin == DEFAULT_STAGE_WORKER_PYTHON_BIN
            for spec in specs.values()
        )
    assert text_loaded["source"]["queue_name"] == "source_items"
    assert text_loaded["run_dir"].endswith(
        "distributed_pipeline_runs/text_edit/defaults"
    )
    assert prosody_loaded["source"]["rows_per_item"] == 128
    assert prosody_loaded["runtime"]["service_log_root"].endswith(
        "distributed_pipeline_runs/prosody_edit/"
        "defaults/logs"
    )
    assert emotion_loaded["source"]["rows_per_item"] == 128
    assert emotion_loaded["runtime"]["service_log_root"].endswith(
        "distributed_pipeline_runs/emotion_style_edit/"
        "defaults/logs"
    )

def test_text_edit_uses_generic_stage_configs():
    text = Path(
        "configs/pipeline/text_edit/defaults.yaml"
    ).read_text(encoding="utf-8")
    assert "/stage/sample/defaults@_global_.stages.sample" in text
    assert "/stage/plan/gemma4_mtp_tp2@_global_.stages.plan" in text
    assert "/stage/synth/text_edit@_global_.stages.synth" in text
    assert "/stage/eval/text_edit@_global_.stages.eval" in text
    assert "/stage/save/parquet@_global_.stages.save" in text
    assert "/stage/text_edit/sample" not in text
    assert "/stage/text_edit/plan" not in text
    assert "/stage/text_edit/synth" not in text
    assert "/stage/text_edit/eval" not in text
    assert "/stage/text_edit/save" not in text


def test_prosody_uses_generic_stage_configs():
    text = PROSODY_DEFAULTS.read_text(encoding="utf-8")
    assert "/stage/sample/defaults@_global_.stages.sample" in text
    assert "/stage/plan/gemma4_mtp_tp2@_global_.stages.plan" in text
    assert "/stage/synth/prosody@_global_.stages.synth" in text
    assert "/stage/eval/prosody@_global_.stages.eval" in text
    assert "/stage/save/parquet@_global_.stages.save" in text


def test_emotion_style_uses_generic_stage_configs():
    text = EMOTION_DEFAULTS.read_text(encoding="utf-8")
    assert "/stage/sample/defaults@_global_.stages.sample" in text
    assert "/stage/plan/gemma4_mtp_tp2@_global_.stages.plan" in text
    assert "/stage/synth/emotion_style_index@_global_.stages.synth_index" in text
    assert "/stage/synth/emotion_style_ted@_global_.stages.synth_ted" in text
    assert "/stage/eval/emotion_style@_global_.stages.eval" in text
    assert "/stage/save/parquet@_global_.stages.save" in text


def test_emotion_style_hydra_configs_compose_stage_services_and_topology():
    for config_path in (
        EMOTION_DEFAULTS,
        Path("configs/pipeline/emotion_style_edit/emilia_en.yaml"),
        Path("configs/pipeline/emotion_style_edit/emilia_zh.yaml"),
    ):
        loaded = load_run_config(config_path)
        adapter = get_adapter(str(loaded["pipeline"]), loaded)
        specs = adapter.stage_specs(loaded)

        assert list(specs) == [
            "sample",
            "plan",
            "synth_index",
            "synth_ted",
            "eval",
            "save",
        ]
        assert specs["sample"].input_queue == "source_items"
        assert specs["sample"].output_queue == "emotion.sample_to_plan"
        assert specs["plan"].input_queue == "emotion.sample_to_plan"
        assert specs["plan"].output_queue is None
        assert specs["synth_index"].input_queue == "emotion.plan_to_synth.index"
        assert specs["synth_index"].output_queue == "emotion.synth_to_eval"
        assert specs["synth_ted"].input_queue == "emotion.plan_to_synth.ted"
        assert specs["synth_ted"].output_queue == "emotion.synth_to_eval"
        assert specs["eval"].input_queue == "emotion.synth_to_eval"
        assert specs["eval"].output_queue == "emotion.eval_to_save"
        assert specs["save"].input_queue == "emotion.eval_to_save"
        assert specs["save"].output_queue is None
        assert specs["save"].parquet_output.enabled is True
        assert specs["save"].parquet_output.dataset_name == loaded["pipeline_config"][
            "output_dataset_name"
        ]
        assert "llm_backend" not in loaded["pipeline_config"]
        assert "llm_model" not in loaded["pipeline_config"]
        assert "aligner" in specs["synth_index"].services
        assert "aligner" in specs["synth_ted"].services
        assert "aligner" not in specs["eval"].services
        assert "wdtw" in specs["eval"].services


def test_emilia_v2_uses_strict_planner_and_language_speaker_libraries():
    expected_manifests = {
        "emilia_en.yaml": (
            "/path/to/speakers/en/parquet_files.list"
        ),
        "emilia_zh.yaml": (
            "/path/to/speakers/zh/parquet_files.list"
        ),
    }
    for filename, expected_manifest in expected_manifests.items():
        loaded = load_run_config(Path("configs/pipeline/emotion_style_edit") / filename)
        cfg = loaded["pipeline_config"]

        assert cfg["strict_emotion_plan"] is True
        assert cfg["speaker_parquet_sources"] == [expected_manifest]
        assert cfg["emotion_edit_mode_probs"]["type_level"] == {
            "type_change": 1.0,
            "intensity_change": 0.0,
        }
        assert cfg["emotion_edit_mode_probs"]["desc"] == {
            "type_change": 1.0,
            "intensity_change": 0.0,
        }
        assert cfg["emotion_tag_weights"]["calm"] == 0.05
        assert cfg["emotion_tag_weights"]["disgusted"] == 0.05
        assert str(loaded["run_dir"]).endswith("_v2")


def test_emilia_worker_counts_for_production_stages():
    for filename in ("emilia_en.yaml", "emilia_zh.yaml"):
        loaded = load_run_config(Path("configs/pipeline/emotion_style_edit") / filename)
        specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

        assert specs["plan"].max_workers == 4
        assert specs["synth_index"].max_workers == 32
        assert specs["synth_ted"].max_workers == 16
        assert specs["eval"].max_workers == 8
        assert specs["save"].max_workers == 3
        assert specs["save"].num_clients_per_worker == 1


def test_emotion_synth_workers_split_main_and_auxiliary_services_across_two_gpus():
    loaded = load_run_config(EMOTION_DEFAULTS)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

    for stage_name, main_instance in (
        ("synth_index", "index_tts/defaults:8002"),
        ("synth_ted", "ted_tts/defaults:8035"),
    ):
        spec = specs[stage_name]
        assert spec.num_clients_per_worker == 3

        worker_zero = resolve_service_graph(materialize_worker_service_graph(
            spec.services, worker_index=0, port_stride=spec.port_stride,
        ))
        worker_one = resolve_service_graph(materialize_worker_service_graph(
            spec.services, worker_index=1, port_stride=spec.port_stride,
        ))

        assert worker_zero.instances[main_instance].gpu_ids == "0"
        assert worker_zero.instances["f5_tts/defaults:8001"].gpu_ids == "1"
        assert worker_zero.instances["qwen3_aligner/defaults:18009"].gpu_ids == "1"
        assert worker_one.instances[
            main_instance.replace(":8002", ":8102").replace(":8035", ":8135")
        ].gpu_ids == "0"
        assert worker_one.instances["f5_tts/defaults:8101"].gpu_ids == "1"
        assert worker_one.instances["qwen3_aligner/defaults:18109"].gpu_ids == "1"

    ted_graph = _resolved(specs["synth_ted"])
    assert ted_graph.instances["ted_tts/defaults:8035"].startup_timeout_sec == 1800


def test_emotion_style_service_profiles_render_without_legacy_placeholders():
    loaded = load_run_config(EMOTION_DEFAULTS)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
    synth_index_commands = {
        command["name"]: command
        for command in _resolved(specs["synth_index"]).command_specs()
    }
    synth_ted_commands = {
        command["name"]: command
        for command in _resolved(specs["synth_ted"]).command_specs()
    }
    eval_commands = {
        command["name"]: command for command in _resolved(specs["eval"]).command_specs()
    }

    assert "index_tts" in synth_index_commands
    assert "ted_tts" not in synth_index_commands
    index_command = synth_index_commands["index_tts"]["command"]
    assert "--accel" in index_command
    assert "--torch-compile" in index_command
    assert "--cuda-kernel" in index_command
    assert "fish_speech" not in synth_index_commands
    assert "ted_tts" in synth_ted_commands
    assert "index_tts" not in synth_ted_commands
    assert "fish_speech" not in synth_ted_commands
    assert "boundary_pause_smoother" in synth_index_commands
    assert "audio_assembly" in synth_index_commands
    assert "aligner" in synth_index_commands
    assert "boundary_pause_smoother" in synth_ted_commands
    assert "audio_assembly" in synth_ted_commands
    assert "aligner" in synth_ted_commands
    assert "aligner" not in eval_commands
    assert {"asr", "utmos", "spk_sim", "wdtw"}.issubset(set(eval_commands))
    assert "clsp" not in eval_commands
    assert "whisper_ser" not in eval_commands
    for command in [
        *synth_index_commands.values(),
        *synth_ted_commands.values(),
        *eval_commands.values(),
    ]:
        assert "${." not in command["command"]


def test_prosody_hydra_config_composes_stage_services_and_topology():
    loaded = load_run_config(PROSODY_DEFAULTS)
    adapter = get_adapter(str(loaded["pipeline"]), loaded)
    specs = adapter.stage_specs(loaded)

    assert list(specs) == ["sample", "plan", "synth", "eval", "save"]
    assert specs["sample"].input_queue == "source_items"
    assert specs["sample"].output_queue == "prosody.sample_to_plan"
    assert specs["plan"].input_queue == "prosody.sample_to_plan"
    assert specs["plan"].output_queue == "prosody.plan_to_synth"
    assert specs["synth"].input_queue == "prosody.plan_to_synth"
    assert specs["synth"].output_queue == "prosody.synth_to_eval"
    assert specs["eval"].input_queue == "prosody.synth_to_eval"
    assert specs["eval"].output_queue == "prosody.eval_to_save"
    assert specs["save"].input_queue == "prosody.eval_to_save"
    assert specs["save"].output_queue is None

    plan_ctx = AdapterContext(
        run_config=loaded,
        worker_id="worker-a",
        stage=specs["plan"],
        services=_resolved(specs["plan"]).direct_services,
    )
    assert specs["plan"].max_workers == 8
    assert specs["plan"].num_clients_per_worker == 16
    assert "llm_backend" not in plan_ctx.pipeline_config
    assert "llm_model" not in plan_ctx.pipeline_config
    assert plan_ctx.services.url("llm") == "http://127.0.0.1:8012"

    synth_ctx = AdapterContext(
        run_config=loaded,
        worker_id="worker-a",
        stage=specs["synth"],
        services=_resolved(specs["synth"]).direct_services,
    )
    assert specs["synth"].max_workers == 16
    assert specs["synth"].num_clients_per_worker == 1
    assert "qwen3_tts" not in synth_ctx.services
    assert "index_tts" not in synth_ctx.services
    assert synth_ctx.services.url("aligner") == "http://127.0.0.1:8029"
    assert synth_ctx.services.url("pitch") == "http://127.0.0.1:8027"
    assert synth_ctx.services.url("rate") == "http://127.0.0.1:8004"
    assert synth_ctx.services.url("f5") == "http://127.0.0.1:8001"
    assert "qwen3_aligner" not in synth_ctx.services

    eval_ctx = AdapterContext(
        run_config=loaded,
        worker_id="worker-a",
        stage=specs["eval"],
        services=_resolved(specs["eval"]).direct_services,
    )
    assert specs["eval"].max_workers == 16
    assert specs["eval"].num_clients_per_worker == 1
    assert eval_ctx.services.url("asr") == "http://127.0.0.1:8010"
    assert eval_ctx.services.url("wer") == "http://127.0.0.1:8011"
    assert eval_ctx.services.url("utmos") == "http://127.0.0.1:8017"
    assert eval_ctx.services.url("spk_sim") == "http://127.0.0.1:8019"
    assert eval_ctx.services.url("f0") == "http://127.0.0.1:8034"
    assert specs["save"].parquet_output.dataset_name == "Emilia-ProsodyEdit-V1"
    policy = plan_ctx.pipeline_config["parameter_policy"]
    assert policy["values_per_wave"] == 3
    assert policy["pitch"]["buckets"]["normal_down"]["values"] == [-6, -5, -4, -3, -2]
    assert policy["rate"]["decimals"] == 2
    assert policy["rate"]["buckets"]["wide_fast"]["ranges"] == [[1.34, 2.0]]


def test_prosody_stage_services_are_worker_managed(tmp_path, monkeypatch):
    checkpoints = str(tmp_path / "models")
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setenv("CHECKPOINTS_DIR", checkpoints)
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_root))
    expected_model_env = {
        "CHECKPOINTS_DIR": checkpoints,
        "HF_HOME": checkpoints,
        "HF_HUB_CACHE": checkpoints,
        "TRANSFORMERS_CACHE": checkpoints,
        "TORCH_HOME": checkpoints + "/torch",
        "HF_MODULES_CACHE": str(artifact_root / "shared/cache/hf_modules"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    loaded = load_run_config(PROSODY_DEFAULTS)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

    plan_graph = _resolved(specs["plan"])
    synth_graph = _resolved(specs["synth"])
    eval_graph = _resolved(specs["eval"])
    plan_commands = plan_graph.command_specs()
    synth_commands = synth_graph.command_specs()
    eval_commands = eval_graph.command_specs()
    assert len(plan_commands) == 1
    assert "--default-backend gemma4_mtp" in plan_commands[0]["command"]

    synth_command_map = {command["instance_key"]: command for command in synth_commands}
    assert list(synth_command_map) == [
        "qwen3_aligner/defaults:18009",
        "aligner_gateway/qwen3:8029",
        "qwen3_tts/defaults:8007",
        "index_tts/defaults:8002",
        "pitch_conversion/defaults:8027",
        "speech_rate_conversion/defaults:8004",
        "f5_tts/defaults:8001",
    ]
    assert synth_graph.roots["pitch"].dependencies["qwen3_tts"].url == "http://127.0.0.1:8007"
    assert synth_graph.roots["pitch"].dependencies["index_tts"].url == "http://127.0.0.1:8002"
    assert synth_graph.roots["rate"].dependencies["qwen3_tts"].url == "http://127.0.0.1:8007"
    assert synth_graph.roots["rate"].dependencies["index_tts"].url == "http://127.0.0.1:8002"
    assert "--qwen3tts-url http://127.0.0.1:8007" in synth_command_map["pitch_conversion/defaults:8027"]["command"]
    assert "--indextts-url http://127.0.0.1:8002" in synth_command_map["pitch_conversion/defaults:8027"]["command"]
    assert "--qwen3tts-url http://127.0.0.1:8007" in synth_command_map["speech_rate_conversion/defaults:8004"]["command"]
    assert "--indextts-url http://127.0.0.1:8002" in synth_command_map["speech_rate_conversion/defaults:8004"]["command"]
    assert synth_command_map["qwen3_tts/defaults:8007"]["env"] == {
        **expected_model_env,
        "PYTHONPATH": "tools/qwen3-tts:.",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert synth_command_map["index_tts/defaults:8002"]["env"] == {
        **expected_model_env,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    assert synth_command_map["pitch_conversion/defaults:8027"]["env"] == {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
    }
    assert all(
        "${." not in command_spec["command"]
        for command_spec in synth_command_map.values()
    )

    eval_command_map = {command["instance_key"]: command["command"] for command in eval_commands}
    assert list(eval_command_map) == [
        "qwen3_asr/defaults:8033",
        "asr_service/qwen3:8010",
        "wer_calc/defaults:8011",
        "utmos/defaults:8017",
        "spk_sim_eval/defaults:8019",
            "qwen3_aligner/defaults:18009",
            "aligner_gateway/qwen3:8029",
            "f0_eval/defaults:8034",
            "wdtw/defaults:8032",
        ]
    assert all("${." not in command for command in eval_command_map.values())

    assert plan_graph.health_urls == ["http://127.0.0.1:8012/health"]
    assert synth_graph.health_urls == [
        "http://127.0.0.1:18009/health",
        "http://127.0.0.1:8029/health",
        "http://127.0.0.1:8007/health",
        "http://127.0.0.1:8002/health",
        "http://127.0.0.1:8027/health",
        "http://127.0.0.1:8004/health",
        "http://127.0.0.1:8001/health",
    ]
    assert eval_graph.health_urls == [
        "http://127.0.0.1:8033/health",
        "http://127.0.0.1:8010/health",
        "http://127.0.0.1:8011/health",
        "http://127.0.0.1:8017/health",
        "http://127.0.0.1:8019/health",
        "http://127.0.0.1:18009/health",
        "http://127.0.0.1:8029/health",
        "http://127.0.0.1:8034/health",
        "http://127.0.0.1:8032/health",
    ]


def test_standalone_pitch_and_rate_service_profiles_render_without_placeholders():
    loaded = load_run_config(PROSODY_DEFAULTS)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)
    commands = {
        command["name"]: command
        for command in _resolved(specs["synth"]).command_specs()
    }
    assert "--qwen3tts-url http://127.0.0.1:8007" in commands["pitch"]["command"]
    assert "--indextts-url http://127.0.0.1:8002" in commands["pitch"]["command"]
    assert "--qwen3tts-url http://127.0.0.1:8007" in commands["rate"]["command"]
    assert "--indextts-url http://127.0.0.1:8002" in commands["rate"]["command"]
    graph = _resolved(specs["synth"])
    assert graph.roots["pitch"].dependencies["qwen3_tts"].url == "http://127.0.0.1:8007"
    assert graph.roots["pitch"].dependencies["index_tts"].url == "http://127.0.0.1:8002"
    assert graph.roots["rate"].dependencies["qwen3_tts"].url == "http://127.0.0.1:8007"
    assert graph.roots["rate"].dependencies["index_tts"].url == "http://127.0.0.1:8002"
    assert all(
        "{" not in command_spec["command"] and "}" not in command_spec["command"]
        for command_spec in commands.values()
    )


def test_service_commands_dedupe_shared_dependencies_by_instance_key():
    graph = resolve_service_graph(
        {
            "parent_a": {
                "config_key": "parent_a/defaults",
                "instance_key": "parent_a/defaults:9001",
                "port": 9001,
                "command": "python parent_a.py",
                "dependencies": {
                    "shared": {
                        "config_key": "shared/defaults",
                        "instance_key": "shared/defaults:9000",
                        "port": 9000,
                        "command": "python shared.py",
                    }
                },
            },
            "parent_b": {
                "config_key": "parent_b/defaults",
                "instance_key": "parent_b/defaults:9002",
                "port": 9002,
                "command": "python parent_b.py",
                "dependencies": {
                    "shared": {
                        "config_key": "shared/defaults",
                        "instance_key": "shared/defaults:9000",
                        "port": 9000,
                        "command": "python shared.py",
                    }
                },
            },
        }
    )
    commands = graph.command_specs()

    instance_keys = [command["instance_key"] for command in commands]
    assert instance_keys.count("shared/defaults:9000") == 1
    assert instance_keys == [
        "shared/defaults:9000",
        "parent_a/defaults:9001",
        "parent_b/defaults:9002",
    ]
    assert graph.health_urls == [
        "http://127.0.0.1:9000/health",
        "http://127.0.0.1:9001/health",
        "http://127.0.0.1:9002/health",
    ]


def test_service_commands_raise_on_post_dedupe_port_conflict():
    raw_services = (
        {
            "service_a": {
                "config_key": "service_a/defaults",
                "instance_key": "service_a/defaults:9001",
                "port": 9001,
                "command": "python svc_a.py --port 9001",
            },
            "service_b": {
                "config_key": "service_b/defaults",
                "instance_key": "service_b/defaults:9001",
                "port": 9001,
                "command": "python svc_b.py --port 9001",
            },
        }
    )

    with pytest.raises(ValueError, match="same port 9001"):
        resolve_service_graph(raw_services)


def test_service_commands_raise_on_same_instance_command_conflict():
    raw_services = (
        {
            "service_a": {
                "config_key": "shared/defaults",
                "instance_key": "shared/defaults:9001",
                "port": 9001,
                "command": "python svc_a.py --port 9001",
            },
            "service_b": {
                "config_key": "shared/defaults",
                "instance_key": "shared/defaults:9001",
                "port": 9001,
                "command": "python svc_b.py --port 9001",
            },
        }
    )

    with pytest.raises(ValueError, match="same service instance"):
        resolve_service_graph(raw_services)


def test_resolved_health_urls_use_declared_service_config():
    graph = resolve_service_graph(
        {
            "service_a": {
                "config_key": "service_a/defaults",
                "instance_key": "service_a/defaults:9123",
                "port": 9123,
                "command": "python svc.py --port 9123",
            }
        }
    )

    urls = graph.health_urls
    assert urls == ["http://127.0.0.1:9123/health"]

    commands = graph.command_specs()
    assert commands[0]["command"] == "python svc.py --port 9123"


def test_adapter_context_services_use_declared_service_config():
    graph = resolve_service_graph(
        {
            "service_a": {
                "config_key": "service_a/defaults",
                "instance_key": "service_a/defaults:9123",
                "port": 9123,
                "command": "python svc.py --port 9123",
            }
        }
    )
    stage = AdapterContext(
        run_config={
            "project_root": "/repo",
            "run_dir": "/run",
            "pipeline": "unit",
        },
        worker_id="worker-a",
        stage=type("Stage", (), {"name": "test", "services": {}})(),
        services=graph.direct_services,
    )
    assert stage.services.as_urls() == {"service_a": "http://127.0.0.1:9123"}


def test_service_commands_use_hydra_resolved_dependency_urls():
    graph = resolve_service_graph(
        {
            "parent": {
                "config_key": "parent/defaults",
                "instance_key": "parent/defaults:9003",
                "port": 9003,
                "dependencies": {
                    "child": {
                        "config_key": "child/defaults",
                        "instance_key": "child/defaults:8123",
                        "port": 8123,
                        "url": "http://127.0.0.1:8123",
                        "command": "python child.py --port 8123",
                    }
                },
                "command": "python parent.py --dep http://127.0.0.1:8123",
            }
        }
    )
    commands = graph.command_specs()
    assert [command["command"] for command in commands] == [
        "python child.py --port 8123",
        "python parent.py --dep http://127.0.0.1:8123",
    ]


@pytest.mark.parametrize("cli_override", [None, "explicit-run"])
def test_prepare_artifact_paths_resolve_before_path_join(tmp_path, monkeypatch, cli_override):
    from speech_edit.runtime.paths import get_artifact_paths

    artifact_dir = tmp_path / "external-artifacts"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    config_path = tmp_path / "run.yaml"
    config_path.write_text(
        "pipeline: text_edit\nrun_id: pathcheck\n"
        "run_dir: ${artifact:worktree}/configured-run\n"
        "runtime:\n  service_log_root: ${runtime.run_dir}/logs\n"
    )
    loaded, effective = prepare_run_config(
        run_config_path=str(config_path), run_dir=cli_override
    )
    expected = get_artifact_paths().worktree / (cli_override or "configured-run")
    assert loaded["artifact_dir"] == str(artifact_dir)
    assert loaded["run_dir"] == str(expected)
    assert loaded["runtime"]["run_dir"] == str(expected)
    assert loaded["runtime"]["service_log_root"] == str(expected / "logs")
    assert loaded["artifact_root"] == str(expected / "artifacts")
    assert effective == expected / "run_config.yaml"
    # Effective configs carry the submitting environment's root across processes.
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "other-artifacts"))
    assert load_run_config(effective)["run_dir"] == str(expected)
    assert load_run_config(effective)["artifact_dir"] == str(artifact_dir)


def test_prepare_run_config_rejects_output_override_outside_artifact_dir(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "allowed"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    target = tmp_path / "not-allowed"
    with pytest.raises(ValueError):
        prepare_run_config(run_config_path=str(PROSODY_DEFAULTS), run_dir=str(target))
    assert not target.exists()


def test_relative_service_output_is_normalized_before_command_expansion(tmp_path, monkeypatch):
    from speech_edit.runtime.config import resolve_run_config
    from speech_edit.runtime.paths import get_artifact_paths

    artifact_dir = tmp_path / "root"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    loaded = resolve_run_config({
        "run_dir": "run",
        "service": {
            "params": {"output_dir": "service-results"},
            "command": "python service.py --output-dir ${.params.output_dir}",
        },
    })
    expected = get_artifact_paths().worktree / "service-results"
    assert loaded["service"]["params"]["output_dir"] == str(expected)
    assert loaded["service"]["command"] == f"python service.py --output-dir {expected}"
    assert not artifact_dir.exists()


@pytest.mark.parametrize("pipeline_config", [TEXT_EDIT_DEFAULTS, PROSODY_DEFAULTS, EMOTION_DEFAULTS, PAUSE_DEFAULTS])
def test_service_output_arguments_preserve_space_and_quote_paths(tmp_path, monkeypatch, pipeline_config):
    import shlex

    artifact_dir = tmp_path / "artifact root's files"
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    config = load_run_config(pipeline_config)
    assert Path(config["run_dir"]).is_relative_to(artifact_dir)
    # Check every materialized command carrying a generated output path.
    def inspect_commands(value):
        if isinstance(value, dict):
            command = value.get("command", "")
            if command and "--gemma4-mtp-log-dir" in command:
                argv = shlex.split(command)
                assert argv[argv.index("--gemma4-mtp-log-dir") + 1] == str(Path(config["runtime"]["service_log_root"]) / "llm_gemma_mtp")
            for child in value.values():
                inspect_commands(child)
        elif isinstance(value, list):
            for child in value:
                inspect_commands(child)
    inspect_commands(config)
    name = {TEXT_EDIT_DEFAULTS: "text_edit", PROSODY_DEFAULTS: "prosody_edit", EMOTION_DEFAULTS: "emotion_style_edit", PAUSE_DEFAULTS: "pause_edit"}[pipeline_config]
    service = load_run_config(f"service/{name}_pipeline/defaults")["service"][f"{name}_pipeline"]
    argv = shlex.split(service["command"])
    assert argv[argv.index("--output-dir") + 1] == service["params"]["output_dir"]
    assert Path(service["params"]["output_dir"]).is_relative_to(artifact_dir)


def test_prepare_from_outside_checkout_uses_source_project_root(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / PROSODY_DEFAULTS
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    loaded, _ = prepare_run_config(run_config_path=str(config_path), run_dir="outside-cwd-run")
    assert loaded["project_root"] == str(project_root)
    assert not (tmp_path / "outside-cwd-run").exists()


def test_yaml_inputs_and_resources_are_not_overridden_by_environment(monkeypatch):
    for name in ("SOURCE_MANIFEST", "SPEAKER_EN_MANIFEST"):
        monkeypatch.setenv(name, "unrelated-session")
    loaded = load_run_config("pipeline/emotion_style_edit/emilia_en")
    assert loaded["source"]["inputs"] == [{"path": "/path/to/data/emilia"}]
    assert loaded["pipeline_config"]["speaker_parquet_sources"] == ["/path/to/speakers/en/parquet_files.list"]
    loaded["source"]["inputs"] = [{"path": "/path/to/custom-emilia"}]
    from speech_edit.runtime.config import resolve_run_config
    resolved = resolve_run_config(loaded)
    assert resolved["source"]["inputs"] == loaded["source"]["inputs"]
