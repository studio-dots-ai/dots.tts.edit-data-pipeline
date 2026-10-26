import json
import os
import signal
import sys
from pathlib import Path

import pytest

from speech_edit.runtime.services import resolve_service_graph
from speech_edit.runtime.services import __main__ as launcher


@pytest.mark.parametrize("name", [
    "text_edit_pipeline", "emotion_style_edit_pipeline",
    "prosody_edit_pipeline", "pause_edit_pipeline",
])
@pytest.mark.parametrize("evaluate", [False, True])
def test_pipeline_service_dependencies(name, evaluate, tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    profile = "evaluate" if evaluate else "defaults"
    config = launcher.service_config(f"service/{name}/{profile}", env_root=tmp_path / "envs")
    graph = resolve_service_graph(config["service"])
    root = graph.roots[name]
    import shlex
    from speech_edit.pipelines import load_parameters
    args = shlex.split(root.command)
    defaults = args[args.index("--pipeline-defaults") + 1]
    overrides = json.loads(args[args.index("--pipeline-parameters") + 1])
    assert defaults == f"pipeline/{name.removesuffix('_pipeline')}/defaults"
    assert load_parameters(defaults, overrides) == load_parameters(f"service/{name}/{profile}")
    assert ("--init-evaluate-services" in root.command) is evaluate
    assert ("wdtw" in root.dependencies) is evaluate
    assert ("wer" in root.dependencies) is evaluate
    if evaluate:
        assert f"--wer-url {root.dependencies['wer'].url}" in root.command
        wdtw = root.dependencies["wdtw"]
        assert f"--wdtw-url {wdtw.url}" in root.command
        assert f"--f0-url {wdtw.dependencies['f0'].url}" in wdtw.command
        if name == "prosody_edit_pipeline":
            assert f"--f0-url {root.dependencies['f0'].url}" in root.command
            assert root.dependencies["f0"] is wdtw.dependencies["f0"]
    for spec in graph.command_specs():
        assert "envs/pipe-" in spec["command"]
        assert str(tmp_path / "envs") in spec["command"]


def test_pause_launch_controls_are_forwarded(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config = launcher.service_config("service/pause_edit_pipeline/defaults", [
        "service.pause_edit_pipeline.params.pipeline.center_regen_second_pass=true",
        "service.pause_edit_pipeline.params.pipeline.center_regen_margin_sec=0.23",
    ])
    command = resolve_service_graph(config["service"]).roots["pause_edit_pipeline"].command
    import shlex
    args = shlex.split(command)
    parameters = json.loads(args[args.index("--pipeline-parameters") + 1])
    assert parameters["center_regen_second_pass"] is True
    assert parameters["center_regen_margin_sec"] == 0.23


def test_signal_during_startup_terminates_children(tmp_path, monkeypatch):
    from speech_edit.runtime.services import processes
    import shlex

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config_path = tmp_path / "services.json"
    config_path.write_text(json.dumps({"service": {"test": {
        "command": f"{shlex.quote(sys.executable)} -c 'import time; time.sleep(120)'",
        "port": 19999, "instance_key": "cancel-test:19999", "config_key": "cancel-test",
    }}}))
    monkeypatch.setattr(sys, "argv", ["services", "--config", str(config_path)])
    monkeypatch.setattr(processes, "_write_startup_snapshot", lambda *a, **kw: None)
    children = []

    def interrupt_wait(urls, **kwargs):
        children.extend(value["process"] for value in kwargs["diagnostics"].values())
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(processes, "wait_for_urls", interrupt_wait)
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with pytest.raises(KeyboardInterrupt, match="cancelled"):
            launcher.main()
        assert children
        assert all(child.poll() is not None for child in children)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def test_cleanup_terminates_group_after_shell_leader_has_exited(tmp_path):
    import shlex
    import subprocess
    import time
    from speech_edit.runtime.services.processes import terminate_processes

    pid_file = tmp_path / "child.pid"
    script = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(120)"
    )
    leader = subprocess.Popen(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} &",
        shell=True, start_new_session=True,
    )
    try:
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        terminate_processes([leader], grace_sec=0.2)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = Path(f"/proc/{child_pid}/stat")
            if not status.exists() or status.read_text().split()[2] == "Z":
                break
            time.sleep(0.02)
        else:
            pytest.fail("Child survived after its shell leader exited")
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_cleanup_allows_child_to_finish_after_leader_exits(tmp_path):
    import shlex
    import subprocess
    import time
    from speech_edit.runtime.services.processes import terminate_processes

    ready = tmp_path / "ready"
    finished = tmp_path / "finished"
    script = f"""import signal, time
from pathlib import Path

def stop(*_):
    time.sleep(0.1)
    Path({str(finished)!r}).write_text("done")
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
Path({str(ready)!r}).touch()
time.sleep(120)
"""
    leader = subprocess.Popen(
        f"{shlex.quote(sys.executable)} -c {shlex.quote(script)} &",
        shell=True, start_new_session=True,
    )
    try:
        leader.wait(timeout=5)
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        terminate_processes([leader], grace_sec=2)
        assert finished.read_text() == "done"
    finally:
        try:
            os.killpg(leader.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.parametrize("pipeline,service,overrides", [
    ("text_edit", "text_edit_pipeline", {"num_generations": 1}),
    ("emotion_style_edit", "emotion_style_edit_pipeline", {"max_candidates": 3}),
    ("prosody_edit", "prosody_edit_pipeline", {}),
    ("pause_edit", "pause_edit_pipeline", {"center_regen_second_pass": False}),
])
def test_agent_and_distributed_parameters_share_yaml(pipeline, service, overrides):
    from speech_edit.pipelines import load_parameters
    from speech_edit.runtime.config import load_run_config

    parameters = load_parameters(f"pipeline/{pipeline}/defaults")
    distributed = load_run_config(f"pipeline/{pipeline}/defaults", resolve=False)
    assert distributed["pipeline_config"] == parameters
    assert load_parameters(f"service/{service}/defaults") == parameters | overrides


@pytest.mark.parametrize("kind,pipeline", [
    ("text", "text_edit"), ("emotion", "emotion_style_edit"),
    ("prosody", "prosody_edit"), ("pause", "pause_edit"),
])
@pytest.mark.parametrize("profile", ["defaults", "evaluate"])
def test_agent_cli_loads_same_parameters_as_service_profile(kind, pipeline, profile, tmp_path, monkeypatch):
    import importlib
    import shlex
    from speech_edit.pipelines import load_parameters

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    name = f"{pipeline}_pipeline"
    key = f"service/{name}/{profile}"
    config = launcher.service_config(key)
    command = shlex.split(resolve_service_graph(config["service"]).roots[name].command)
    module = importlib.import_module(f"speech_edit.pipelines.{kind}.service.api_service")
    monkeypatch.setattr(module, "RUNTIME_CONFIG", module.ServiceRuntimeConfig())
    monkeypatch.setattr(sys, "argv", command[2:])
    calls = []
    monkeypatch.setattr(module.uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
    module.main()
    assert len(calls) == 1
    assert module.RUNTIME_CONFIG.pipeline_parameters == load_parameters(key)
    assert module.RUNTIME_CONFIG.init_evaluate_services is (profile == "evaluate")


@pytest.mark.parametrize("profile", ["wer_calc/defaults", "vad/defaults", "f0_eval/defaults", "llm_service/gemini_2_5pro"])
def test_services_without_local_models_do_not_require_checkpoints(profile, tmp_path, monkeypatch):
    monkeypatch.delenv("CHECKPOINTS_DIR", raising=False)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config = launcher.service_config(f"service/{profile}")
    graph = resolve_service_graph(config["service"])
    assert graph.command_specs()
    for spec in graph.command_specs():
        assert "CHECKPOINTS_DIR" not in spec["env"]
        assert "HF_HUB_CACHE" not in spec["env"]


@pytest.mark.parametrize("profile", [
    "f5_tts/defaults", "index_tts/defaults", "ted_tts/defaults", "qwen3_aligner/defaults",
    "qwen3_asr/defaults", "qwen3_tts/defaults", "spk_sim_eval/defaults", "utmos/defaults",
    "asr_service/qwen3_transformers", "llm_service/gemma4_mtp_tp2",
    "llm_service/qwen3_omni_30b_a3b_instruct",
])
def test_local_model_configs_require_and_pin_checkpoints(profile, tmp_path, monkeypatch):
    from omegaconf.errors import InterpolationResolutionError
    from speech_edit.runtime.config import resolve_run_config

    monkeypatch.delenv("CHECKPOINTS_DIR", raising=False)
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    with pytest.raises(InterpolationResolutionError, match="CHECKPOINTS_DIR"):
        launcher.service_config(f"service/{profile}")
    original = str(tmp_path / "model-inputs")
    monkeypatch.setenv("CHECKPOINTS_DIR", original)
    resolved = launcher.service_config(f"service/{profile}")
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path / "other-host-models"))
    remote = resolve_run_config(resolved)
    graph = resolve_service_graph(remote["service"])
    for spec in graph.command_specs():
        assert spec["env"]["CHECKPOINTS_DIR"] == original
        assert spec["env"]["HF_HUB_CACHE"] == original
        assert spec["env"]["TORCH_HOME"] == original + "/torch"


@pytest.mark.parametrize("override", [False, True])
def test_standalone_service_uses_one_log_root(tmp_path, monkeypatch, override):
    import shlex
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("CHECKPOINTS_DIR", str(tmp_path / "models"))
    custom = tmp_path / "artifacts/custom-service-logs"
    args = [f"runtime.service_log_root={custom}"] if override else []
    config = launcher.service_config("service/llm_service/gemma4_mtp_tp2", args)
    log_root = Path(config["runtime"]["service_log_root"])
    if override:
        assert log_root == custom
    else:
        assert log_root.parent.name == "service_logs"
        assert log_root.is_relative_to(tmp_path / "artifacts")
    command = shlex.split(resolve_service_graph(config["service"]).roots["llm_service"].command)
    assert command[command.index("--gemma4-mtp-log-dir") + 1] == str(log_root / "llm_gemma_mtp")


def test_standalone_service_rejects_logs_outside_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    with pytest.raises(ValueError, match="inside ARTIFACT_DIR"):
        launcher.service_config("service/wer_calc/defaults", [f"runtime.service_log_root={tmp_path / 'outside'}"])


def test_environment_root_applies_to_all_services_and_preserves_explicit_paths(tmp_path, monkeypatch):
    import shlex

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    env_root = tmp_path / "shared environments"
    explicit = tmp_path / "specific environment/bin/python"
    cfg = launcher.service_config(
        "service/emotion_style_edit_pipeline/evaluate",
        [f"service.emotion_style_edit_pipeline.python_bin={explicit}"],
        env_root=env_root,
    )
    graph = resolve_service_graph(cfg["service"])
    root = graph.roots["emotion_style_edit_pipeline"]
    assert shlex.split(root.command)[0] == str(explicit)
    interpreters = set()
    for node in graph.instances.values():
        if node is root:
            continue
        if node.config_key == "qwen3_asr/defaults":
            assert shlex.split(node.command)[-1] == str(env_root / "pipe-asr/.venv/bin/python")
            continue
        interpreter = Path(shlex.split(node.command)[0])
        assert interpreter.is_relative_to(env_root), node.command
        interpreters.add(interpreter.relative_to(env_root).parts[0])
    assert interpreters == {"pipe-cpu", "pipe-gpu", "pipe-asr", "pipe-indextts"}
    llm_cfg = launcher.service_config("service/llm_service/gemini_2_5pro", env_root=env_root)
    assert shlex.split(llm_cfg["service"]["llm_service"]["command"])[0] == str(env_root / "pipe-llm/.venv/bin/python")
