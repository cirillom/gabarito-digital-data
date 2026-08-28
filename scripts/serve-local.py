"""Serve the checked-out data catalog and source PDFs with browser CORS."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class CorsHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parent.parent
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port),
        lambda *handler_args, **kwargs: CorsHandler(
            *handler_args,
            directory=str(repository),
            **kwargs,
        ),
    )
    print(f"Serving {repository} at http://127.0.0.1:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
