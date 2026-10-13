"""Minimal redirect server for the legacy Render Web Service."""

import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


TARGET = os.environ.get("REDIRECT_TARGET", "https://vics-ir.onrender.com").rstrip("/")


class RedirectHandler(BaseHTTPRequestHandler):
    def _redirect(self) -> None:
        self.send_response(302)
        self.send_header("Location", f"{TARGET}{self.path}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    do_GET = _redirect
    do_HEAD = _redirect


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), RedirectHandler)
    print(f"Redirecting http://0.0.0.0:{port} to {TARGET}")
    server.serve_forever()
