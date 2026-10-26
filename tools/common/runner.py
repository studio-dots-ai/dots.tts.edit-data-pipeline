"""
Shared service runner for tool services.

Provides ``ServiceRunner`` — a lightweight helper that eliminates the
repetitive argparse + tempdir + uvicorn boilerplate present in every
service's ``main()`` function.

Usage example (minimal)::

    from tools.common.runner import ServiceRunner

    runner = ServiceRunner(
        app=app,
        service_name="Volume Normalizer",
        default_port=8006,
    )

    if __name__ == "__main__":
        runner.run()

Usage example (with temp dir and model loading)::

    runner = ServiceRunner(
        app=app,
        service_name="VAD",
        default_port=8005,
        temp_dir_prefix="vad_service_",
    )

    if __name__ == "__main__":
        runner.run()
        # After run() returns, runner.temp_dir is set if temp_dir_prefix was given.

Usage example (GPU service with model loading)::

    def load(args):
        load_model(args.device)

    runner = ServiceRunner(
        app=app,
        service_name="UTMOS",
        default_port=8017,
        needs_device_arg=True,
        on_startup=load,
    )

    if __name__ == "__main__":
        runner.run()

Usage example (service-specific arguments)::

    def add_args(parser):
        parser.add_argument("--model-id", type=str, default="some/model")
        parser.add_argument("--workers", type=int, default=1)

    def startup(args):
        load_model(args.model_id, args.device)

    runner = ServiceRunner(
        app=app,
        service_name="Whisper SER",
        default_port=8016,
        needs_device_arg=True,
        extra_args=add_args,
        on_startup=startup,
    )

    if __name__ == "__main__":
        runner.run()
"""

import argparse
import logging
import tempfile
from typing import Callable, Optional

import uvicorn
from fastapi import FastAPI


logger = logging.getLogger(__name__)


class ServiceRunner:
    """Encapsulates the common service startup boilerplate.

    Handles:
    - ``argparse`` setup with ``--host``, ``--port``, ``--log-level`` (and
      optionally ``--device``)
    - Optional temp-directory creation
    - Optional pre-start callback (e.g. model loading)
    - ``uvicorn.run()`` invocation

    Attributes:
        temp_dir: Path to the temp directory created on ``run()``, or ``None``
            if *temp_dir_prefix* was not set.
        args: Parsed ``argparse.Namespace`` after ``run()`` is called.
    """

    def __init__(
        self,
        app: FastAPI,
        service_name: str,
        default_port: int,
        *,
        temp_dir_prefix: Optional[str] = None,
        needs_device_arg: bool = False,
        extra_args: Optional[Callable[[argparse.ArgumentParser], None]] = None,
        on_startup: Optional[Callable[[argparse.Namespace], None]] = None,
        log_level_arg: bool = True,
    ) -> None:
        """
        Args:
            app: The FastAPI application instance.
            service_name: Human-readable service name (used in argparse help
                and startup log messages).
            default_port: Default port for ``--port``.
            temp_dir_prefix: If set, a temporary directory is created with this
                prefix and stored in ``self.temp_dir``.  The calling service
                should reference ``runner.temp_dir`` (or its own module-level
                ``TEMP_DIR`` global that it sets from the runner).
            needs_device_arg: If ``True``, adds a ``--device`` argument
                (default ``"cuda"``).
            extra_args: Optional callback to add service-specific arguments
                to the parser.  Receives the ``ArgumentParser`` instance.
            on_startup: Optional callback invoked after argument parsing and
                temp-dir creation but before ``uvicorn.run()``.  Receives the
                parsed ``argparse.Namespace``.  Use this for model loading.
            log_level_arg: If ``True`` (default), adds ``--log-level`` argument
                and passes it to ``uvicorn.run()``.
        """
        self.app = app
        self.service_name = service_name
        self.default_port = default_port
        self.temp_dir_prefix = temp_dir_prefix
        self.needs_device_arg = needs_device_arg
        self.extra_args_fn = extra_args
        self.on_startup = on_startup
        self.log_level_arg = log_level_arg

        self.temp_dir: Optional[str] = None
        self.args: Optional[argparse.Namespace] = None

    def _build_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description=f"{self.service_name} Service")
        parser.add_argument(
            "--host",
            type=str,
            default="0.0.0.0",
            help="Host to bind to",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=self.default_port,
            help=f"Port to bind to (default: {self.default_port})",
        )

        if self.log_level_arg:
            parser.add_argument(
                "--log-level",
                type=str,
                default="info",
                choices=["debug", "info", "warning", "error"],
                help="Uvicorn log level",
            )

        if self.needs_device_arg:
            parser.add_argument(
                "--device",
                type=str,
                default="cuda",
                help="Compute device (default: cuda)",
            )

        if self.extra_args_fn is not None:
            self.extra_args_fn(parser)

        return parser

    def run(self) -> None:
        """Parse arguments, set up temp dir, run startup hook, launch uvicorn."""
        parser = self._build_parser()
        self.args = parser.parse_args()

        # Temp directory
        if self.temp_dir_prefix:
            self.temp_dir = tempfile.mkdtemp(prefix=self.temp_dir_prefix)
            logger.info(f"Temp directory: {self.temp_dir}")

        # Pre-start callback (model loading, etc.)
        if self.on_startup is not None:
            self.on_startup(self.args)

        # Launch
        log_level = getattr(self.args, "log_level", "info")
        logger.info(
            f"Starting {self.service_name} on {self.args.host}:{self.args.port}"
        )

        uvicorn_kwargs = dict(
            host=self.args.host,
            port=self.args.port,
            log_level=log_level,
        )

        # Support optional workers arg (e.g. wer_calc)
        if hasattr(self.args, "workers") and self.args.workers > 1:
            uvicorn_kwargs["workers"] = self.args.workers

        uvicorn.run(self.app, **uvicorn_kwargs)
