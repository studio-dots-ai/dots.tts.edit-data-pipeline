import os
import signal
import subprocess
import time
from pathlib import Path


SERVICE_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = SERVICE_DIR / "scripts" / "launch_gemma4_mtp_vllm.sh"


def test_gemma4_mtp_launcher_shell_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_gemma4_mtp_launcher_auto_detects_visible_gpus(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nprintf '0\\n1\\n'\n")
    (fake_bin / "curl").write_text(
        "#!/usr/bin/env bash\n"
        "printf '{\"data\":[{\"id\":\"gemma-4-31b-it-mtp-tp2-32k\"}]}'\n"
    )
    for path in fake_bin.iterdir():
        path.chmod(0o755)

    model_path = tmp_path / "model"
    draft_model_path = tmp_path / "draft"
    log_dir = tmp_path / "logs"
    model_path.mkdir()
    draft_model_path.mkdir()
    log_dir.mkdir()
    (log_dir / "gemma4_mtp_vllm_18112.meta").write_text(
        "\n".join(
            [
                f"pid={os.getpid()}",
                "log_file=/repo/log.log",
                "base_url=http://127.0.0.1:18112/v1",
                "port=18112",
                "served_model_name=gemma-4-31b-it-mtp-tp2-32k",
                "gpus=0,1",
                "gpu_count=2",
                "tensor_parallel_size=2",
                "data_parallel_size=1",
                "data_parallel_size_local=1",
                "max_model_len=32768",
                "max_num_batched_tokens=32768",
                "max_num_seqs=8",
                "gpu_memory_utilization=0.92",
                f"model_path={model_path}",
                f"draft_model_path={draft_model_path}",
            ]
        )
        + "\n"
    )

    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--model-path",
            str(model_path),
            "--draft-model-path",
            str(draft_model_path),
            "--vllm-python",
            "/bin/true",
            "--log-dir",
            str(log_dir),
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "BASE_URL=http://127.0.0.1:18112/v1" in result.stdout
    assert "STATUS=reused" in result.stdout


def test_gemma4_mtp_launcher_refuses_mismatched_reuse_metadata(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text(
        "#!/usr/bin/env bash\n"
        "printf '{\"data\":[{\"id\":\"gemma-4-31b-it-mtp-tp2-32k\"}]}'\n"
    )
    for path in fake_bin.iterdir():
        path.chmod(0o755)

    model_path = tmp_path / "model"
    draft_model_path = tmp_path / "draft"
    log_dir = tmp_path / "logs"
    model_path.mkdir()
    draft_model_path.mkdir()
    log_dir.mkdir()
    (log_dir / "gemma4_mtp_vllm_18112.meta").write_text(
        "\n".join(
            [
                f"pid={os.getpid()}",
                "served_model_name=gemma-4-31b-it-mtp-tp2-32k",
                "gpus=0,1",
                "gpu_count=2",
                "tensor_parallel_size=2",
                "data_parallel_size=1",
                "data_parallel_size_local=1",
                "max_model_len=32768",
                "max_num_batched_tokens=32768",
                "max_num_seqs=8",
                "gpu_memory_utilization=0.92",
                f"model_path={model_path}",
                f"draft_model_path={draft_model_path}",
            ]
        )
        + "\n"
    )

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--gpus",
            "0",
            "--model-path",
            str(model_path),
            "--draft-model-path",
            str(draft_model_path),
            "--vllm-python",
            "/bin/true",
            "--log-dir",
            str(log_dir),
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 3
    assert "Refusing to reuse" in result.stderr


def test_gemma4_mtp_launcher_cleans_started_process_on_readiness_failure(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "curl").write_text("#!/usr/bin/env bash\nexit 22\n")
    for path in fake_bin.iterdir():
        path.chmod(0o755)

    model_path = tmp_path / "model"
    draft_model_path = tmp_path / "draft"
    log_dir = tmp_path / "logs"
    fake_python = tmp_path / "fake_vllm_python.sh"
    child_pid_file = tmp_path / "child.pid"
    build_tool = tmp_path / "environment-build-tool"
    build_tool.write_text("#!/usr/bin/env bash\nprintf 'available'\n")
    build_tool.chmod(0o755)
    build_tool_result = tmp_path / "build-tool-result"
    model_path.mkdir()
    draft_model_path.mkdir()
    log_dir.mkdir()
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"echo $$ > {child_pid_file}\n"
        f"environment-build-tool > '{build_tool_result}'\n"
        "trap 'exit 0' TERM INT HUP\n"
        "while true; do sleep 1; done\n"
    )
    fake_python.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        [
            str(LAUNCHER),
            "--gpus",
            "0",
            "--model-path",
            str(model_path),
            "--draft-model-path",
            str(draft_model_path),
            "--vllm-python",
            str(fake_python),
            "--log-dir",
            str(log_dir),
            "--timeout",
            "1",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=20,
    )

    assert result.returncode == 1
    assert build_tool_result.read_text() == "available"
    metadata_file = log_dir / "gemma4_mtp_vllm_18112.meta"
    assert not metadata_file.exists()
    child_pid = int(child_pid_file.read_text().strip())
    for _ in range(20):
        if subprocess.run(["kill", "-0", str(child_pid)]).returncode != 0:
            break
        time.sleep(0.1)
    assert subprocess.run(["kill", "-0", str(child_pid)]).returncode != 0


def test_launcher_cleans_workers_when_leader_exits_before_readiness(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text("#!/usr/bin/env bash\nexit 22\n")
    curl.chmod(0o755)
    worker = tmp_path / "worker.sh"
    worker.write_text("#!/usr/bin/env bash\ntrap '' TERM\nwhile true; do sleep 1; done\n")
    worker.chmod(0o755)
    leader_pid_file = tmp_path / "leader.pid"
    child_pid_file = tmp_path / "child.pid"
    fake_python = tmp_path / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f"echo $$ > '{leader_pid_file}'\n"
        f"'{worker}' &\n"
        f"echo $! > '{child_pid_file}'\n"
        "exit 1\n"
    )
    fake_python.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    env = dict(os.environ, PATH=f"{fake_bin}:{os.environ['PATH']}")
    try:
        result = subprocess.run(
            [str(LAUNCHER), "--gpus", "0", "--model-path", str(model),
             "--draft-model-path", str(model), "--vllm-python", str(fake_python),
             "--log-dir", str(tmp_path / "logs"), "--timeout", "10"],
            capture_output=True, text=True, env=env, timeout=25,
        )
        assert result.returncode == 1, result.stderr
        status = Path(f"/proc/{int(child_pid_file.read_text())}/status")
        for _ in range(20):
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(0.1)
        assert not status.exists() or "State:\tZ" in status.read_text()
    finally:
        if leader_pid_file.exists():
            try:
                os.killpg(int(leader_pid_file.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
