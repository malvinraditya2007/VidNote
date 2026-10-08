"""Keamanan localhost (plan 5.1).

Website jahat di tab lain bisa menyuruh browser user mengirim request ke
localhost:PORT (CSRF) atau membaca balasan lewat DNS rebinding. Mitigasi:
- Host header hanya 127.0.0.1:PORT / localhost:PORT (lawan DNS rebinding).
- Origin header pada request yang mengubah state (POST) harus same-origin.
- Token acak per sesi di header X-VidNote-Token untuk endpoint API.
- Tanpa CORS (frontend disajikan origin yang sama).
"""
from __future__ import annotations

import secrets
from urllib.parse import urlsplit

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

SESSION_TOKEN = secrets.token_urlsafe(32)
TOKEN_HEADER = "x-vidnote-token"
# Path yang tidak butuh token (dibuka langsung di browser / health check).
# "/" dan static disajikan same-origin; token disuntik ke HTML, bukan di-gate di sini.
PUBLIC_PATHS = {"/", "/health", "/favicon.ico", "/style.css", "/app.js"}


def allowed_hosts(port: int) -> set[str]:
    return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}


def allowed_origins(port: int) -> set[str]:
    hosts = allowed_hosts(port)
    return {f"http://{h}" for h in hosts} | {f"https://{h}" for h in hosts}


def host_ok(host_header: str | None, port: int) -> bool:
    return bool(host_header) and host_header.lower() in allowed_hosts(port)


def origin_ok(origin: str | None, port: int) -> bool:
    """Origin harus ada dan same-origin. Request lintas situs selalu mengirim Origin."""
    if not origin:
        return False
    try:
        p = urlsplit(origin)
    except ValueError:
        return False
    netloc = p.netloc.lower()
    return p.scheme in ("http", "https") and netloc in allowed_hosts(port)


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS


def token_ok(header_value: str | None, expected: str = SESSION_TOKEN) -> bool:
    return bool(header_value) and secrets.compare_digest(header_value, expected)


class LocalhostSecurity(BaseHTTPMiddleware):
    """Terapkan cek Host, Origin, dan token sebelum request sampai ke endpoint."""

    def __init__(self, app, port: int, token: str = SESSION_TOKEN):
        super().__init__(app)
        self.port = port
        self.token = token

    async def dispatch(self, request: Request, call_next):
        def deny(status: int, msg: str) -> JSONResponse:
            return JSONResponse({"detail": msg}, status_code=status)

        if not host_ok(request.headers.get("host"), self.port):
            return deny(400, "Host header tidak diizinkan")

        method = request.method.upper()
        path = request.url.path

        # Request yang mengubah state wajib same-origin.
        if method not in ("GET", "HEAD", "OPTIONS"):
            if not origin_ok(request.headers.get("origin"), self.port):
                return deny(403, "Origin tidak diizinkan (kemungkinan request lintas situs)")

        # Token untuk semua endpoint kecuali yang publik.
        # Header diutamakan; query ?token= hanya untuk GET (EventSource/<video> tak bisa
        # set header). Query token aman di sini karena same-origin sudah dijaga Host/Origin.
        if not is_public(path):
            tok = request.headers.get(TOKEN_HEADER)
            if tok is None and method in ("GET", "HEAD"):
                tok = request.query_params.get("token")
            if not token_ok(tok, self.token):
                return deny(401, "token sesi tidak valid atau tidak ada")

        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; media-src 'self' blob:; "
            "style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp
