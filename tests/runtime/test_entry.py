import pytest

from speech_edit.runtime import entry
from speech_edit.runtime.models import StageSpec


@pytest.mark.parametrize("fail", [False, True])
def test_entry_uses_current_worker_config_and_cleans_services(tmp_path, monkeypatch, fail):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config = {"artifact_dir": str(tmp_path), "project_root": str(tmp_path), "run_dir": str(tmp_path / "run")}
    spec = StageSpec(name="s", input_queue="q", output_queue=None, worker_python_bin="python", services={})
    events = []
    class Group:
        def __enter__(self):
            events.append("start")
            return self
        def __exit__(self, *_):
            events.append("stop")
    monkeypatch.setenv("DIST_PIPELINE_WORKER_INDEX", "3")
    monkeypatch.setattr(entry, "_install_termination_handlers", lambda: None)
    monkeypatch.setattr(entry, "load_run_config", lambda path: config)
    monkeypatch.setattr(entry, "_load_stage_spec", lambda *args: spec)
    monkeypatch.setattr(entry, "start_services", lambda **kwargs: Group())
    def run(**kwargs):
        assert kwargs["worker_id"] == "worker"
        assert kwargs["stage_name"] == "s"
        events.append("execute")
        if fail:
            raise RuntimeError("stage failed")
        return 0
    monkeypatch.setattr(entry, "run_worker", run)
    args = ["--run-config", "run.yaml", "--pipeline", "example", "--stage", "s", "--worker-id", "worker"]
    if fail:
        with pytest.raises(RuntimeError, match="stage failed"):
            entry.main(args)
    else:
        assert entry.main(args) == 0
    assert events == ["start", "execute", "stop"]


def test_local_runner_completion_and_failure_cleanup(tmp_path, monkeypatch):
    import pytest
    from speech_edit.runtime import local_runner

    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    config = {"artifact_dir": str(tmp_path), "pipeline": "example", "project_root": str(tmp_path),
              "run_dir": str(tmp_path), "run_id": "example"}
    events = []
    class Store:
        def get_run_state(self):
            return {"desired_state": "running"}
        def completion_stats(self):
            return {"run_complete": int(complete), "queue_dead": 0}
        def close(self):
            events.append("close")
    class Process:
        returncode = 0
        def poll(self):
            return self.returncode
    monkeypatch.setattr(local_runner, "load_run_config", lambda path: config)
    monkeypatch.setattr(local_runner, "QueueStore", lambda **kwargs: Store())
    monkeypatch.setattr(local_runner, "build_local_worker_specs", lambda cfg: [object()])
    monkeypatch.setattr(local_runner, "_launch_worker", lambda **kwargs: Process())
    monkeypatch.setattr(local_runner, "_terminate", lambda processes: events.append("terminate"))
    complete = True
    assert local_runner.run_local(run_config_path="run.yaml")["run_complete"] == 1
    assert events == ["terminate", "close"]
    events.clear()
    complete = False
    Process.returncode = 1
    with pytest.raises(RuntimeError, match="All workers exited"):
        local_runner.run_local(run_config_path="run.yaml")
    assert events == ["terminate", "close"]
