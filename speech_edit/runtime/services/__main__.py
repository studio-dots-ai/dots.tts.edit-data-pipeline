"""Launch a configured service graph with the shared runtime process lifecycle."""

import argparse
import json
import signal
import threading
import uuid
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from speech_edit.runtime.config import load_run_config, resolve_run_config
from speech_edit.runtime.services import resolve_service_graph, start_services


def service_config(config_name, overrides=(), *, env_root=None):
    cfg = OmegaConf.merge(
        {"runtime": {"service_log_root": f"${{artifact:worktree}}/service_logs/{uuid.uuid4().hex[:10]}"}},
        load_run_config(config_name, resolve=False),
    )
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    if env_root is not None:
        root = Path(env_root).expanduser().resolve()

        def use_environments(value):
            if not isinstance(value, DictConfig):
                return
            for key, child in list(value.items_ex(resolve=False)):
                if key == "python_bin" and str(child).startswith("envs/"):
                    value[key] = str(root / str(child).removeprefix("envs/"))
                else:
                    use_environments(child)

        use_environments(cfg)
    return resolve_run_config(OmegaConf.to_container(cfg, resolve=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-root", help="Existing directory containing pipe-* environments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("overrides", nargs="*", help="Dotted config overrides: key=value")
    args = parser.parse_args()
    config = service_config(args.config, args.overrides, env_root=args.env_root)
    graph = resolve_service_graph(config["service"])
    if args.dry_run:
        print(json.dumps(graph.command_specs(), indent=2))
        return
    stopped = threading.Event()
    def cancel_startup(*_):
        raise KeyboardInterrupt("Service startup cancelled")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, cancel_startup)
    with start_services(
        graph=graph,
        project_root=Path(config["project_root"]),
        log_dir=Path(config["runtime"]["service_log_root"]),
    ) as group:
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stopped.set())
        while not stopped.wait(1):
            for process in group.processes:
                if process.poll() is not None:
                    raise RuntimeError(f"Service process {process.pid} exited: {process.returncode}")


if __name__ == "__main__":
    main()
