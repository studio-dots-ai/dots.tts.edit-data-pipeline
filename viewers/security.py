"""Browser access boundaries for the operator console."""

from fastapi import FastAPI, Request
from fastapi.responses import Response


def configure_access(app: FastAPI, host: str) -> None:
    allowed_hosts = {"localhost", "127.0.0.1", "::1", host.lower()}

    @app.middleware("http")
    async def check_access(request: Request, call_next):
        try:
            hostname = request.url.hostname
        except ValueError:
            return Response("Invalid host", status_code=400)
        if host not in {"0.0.0.0", "::"} and hostname not in allowed_hosts:
            return Response("Invalid host", status_code=400)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin")
            expected_origin = f"{request.url.scheme}://{request.url.netloc}"
            if (
                request.headers.get("sec-fetch-site") == "cross-site"
                or (origin is not None and origin != expected_origin)
            ):
                return Response("Cross-origin writes are not allowed", status_code=403)
        return await call_next(request)
