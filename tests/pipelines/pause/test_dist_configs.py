from pathlib import Path

import pytest

from speech_edit.runtime.config import load_run_config
from speech_edit.runtime.models import AdapterContext
from speech_edit.runtime.registry import get_adapter
from speech_edit.runtime.services.resolver import resolve_service_graph


CONFIG_DIR = Path(
    "configs/pipeline/pause_edit"
)


@pytest.mark.parametrize(
    "config_path",
    [
        CONFIG_DIR / "defaults.yaml",
        Path("configs/pipeline/pause_edit/emilia_en.yaml"),
        Path("configs/pipeline/pause_edit/emilia_zh.yaml"),
    ],
)
def test_pause_dist_configs_compose(config_path: Path) -> None:
    loaded = load_run_config(config_path)
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

    assert list(specs) == ["sample", "plan", "synth", "eval", "save"]


def test_pause_default_synth_uses_boundary_pause_smoother_service_key() -> None:
    loaded = load_run_config(CONFIG_DIR / "defaults.yaml")
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

    synth_graph = resolve_service_graph(specs["synth"].services)
    synth_ctx = AdapterContext(
        run_config=loaded,
        worker_id="worker-a",
        stage=specs["synth"],
        services=synth_graph.direct_services,
    )

    assert synth_ctx.services.url("boundary_pause_smoother") == "http://127.0.0.1:8018"
    assert synth_ctx.services.url("aligner") == "http://127.0.0.1:8029"
    assert [spec["instance_key"] for spec in synth_graph.command_specs()].count(
        synth_graph.roots["aligner"].instance_key
    ) == 1
    assert (
        synth_graph.roots["boundary_pause_smoother"].dependencies["aligner"].url
        == "http://127.0.0.1:8029"
    )


def test_pause_default_eval_includes_utmos_service() -> None:
    loaded = load_run_config(CONFIG_DIR / "defaults.yaml")
    specs = get_adapter(str(loaded["pipeline"]), loaded).stage_specs(loaded)

    eval_graph = resolve_service_graph(specs["eval"].services)
    eval_ctx = AdapterContext(
        run_config=loaded,
        worker_id="worker-a",
        stage=specs["eval"],
        services=eval_graph.direct_services,
    )

    assert loaded["pipeline_config"]["skip_utmos_eval"] is False
    assert loaded["pipeline_config"]["skip_spk_sim_eval"] is False
    assert loaded["pipeline_config"]["skip_wdtw_eval"] is False
    assert "skip_quality_eval" not in loaded["pipeline_config"]
    assert "skip_spk_sim" not in loaded["pipeline_config"]
    assert "skip_wdtw" not in loaded["pipeline_config"]
    assert eval_ctx.services.url("asr") == "http://127.0.0.1:8010"
    assert eval_ctx.services.url("aligner") == "http://127.0.0.1:8029"
    assert eval_ctx.services.url("wer") == "http://127.0.0.1:8011"
    assert eval_ctx.services.url("utmos") == "http://127.0.0.1:8017"
    assert eval_ctx.services.url("spk_sim") == "http://127.0.0.1:8019"
    assert eval_ctx.services.url("wdtw") == "http://127.0.0.1:8032"
