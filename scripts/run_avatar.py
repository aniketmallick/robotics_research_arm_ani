#!/usr/bin/env python
"""ARM-ANI's face. Read-only — it never touches the arm, the camera or a gate.

    python scripts/run_avatar.py                  # localhost
    python scripts/run_avatar.py --host 0.0.0.0   # + phones on the same wifi
    python scripts/run_avatar.py --port 9001

Two screens ship, and they do different jobs:

    run_dashboard.py  the PROOF     gates, confidence, the audit trail
    run_avatar.py     the DELIGHT   the mascot, mirroring what it is doing

This one serves ``armani/data/avatar.html`` and the state the agent publishes to
``logs/ui_state.json``. Same shape as the dashboard server: standard library
only, localhost by default, no dependency to install on a demo morning.

The page also runs standalone with ``?demo=1``, which drives the animation from
a canned timeline — useful for showing the face with no agent running.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TESTS_DIR = REPO_ROOT / "tests"
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import _bootstrap  # noqa: E402,F401  (verifies the interpreter, exits if wrong)

from armani import config, uistate  # noqa: E402
from armani.logutil import get_logger  # noqa: E402

log = get_logger("run_avatar")

PAGE = REPO_ROOT / "armani" / "data" / "avatar.html"


def lan_address() -> str | None:
    """This machine's LAN IP, for the URL a phone can actually open."""
    try:
        # No packet is sent; connect() on a UDP socket just picks the interface
        # the kernel would route through, which is the address we want to print.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.settimeout(0.2)
            probe.connect(("8.8.8.8", 80))
            return probe.getsockname()[0]
    except OSError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=config.AVATAR_PORT)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="bind address. Localhost by default; pass 0.0.0.0 to let a phone or "
             "a second screen on the same wifi open it.",
    )
    parser.add_argument("--dry-run", action="store_true", help="check the sources and exit")
    args = parser.parse_args()

    print("=== ARM-ANI avatar ===")
    print(f"page  : {PAGE}")
    print(f"state : {config.UI_STATE_PATH}")

    if not PAGE.is_file():
        print(f"\nmissing page: {PAGE}", file=sys.stderr)
        return 1

    state = uistate.current()
    print(f"now   : {state['state']}" + ("  (stale — no agent running)" if state.get("stale") else ""))

    if args.dry_run:
        print(f"[dry-run] would serve on http://localhost:{args.port}")
        return 0

    server = ThreadingHTTPServer((args.host, args.port), _handler())
    print(f"\n  OPEN:  http://localhost:{args.port}")
    if args.host == "0.0.0.0":
        address = lan_address()
        if address:
            print(f"  PHONE: http://{address}:{args.port}   (same wifi)")
        else:
            print("  (could not work out this machine's LAN address)")
    print(f"\n  Preview with no agent: http://localhost:{args.port}/?demo=1")
    print("  Read-only: this never touches the arm. Ctrl-C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


def _handler():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            route = self.path.split("?", 1)[0]
            try:
                if route in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif route == "/state":
                    payload = json.dumps(uistate.current()).encode("utf-8")
                    self._send(payload, "application/json", cache=False)
                else:
                    self._send(b"not found", "text/plain", status=404)
            except BrokenPipeError:
                pass  # a phone locked its screen mid-poll
            except Exception as exc:
                log.error("avatar request %s failed: %s", route, exc)
                try:
                    self._send(str(exc).encode("utf-8"), "text/plain", status=500)
                except Exception:
                    pass

        def _send(self, body: bytes, content_type: str, status: int = 200, cache: bool = True) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if not cache:
                self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            # The page polls 8x/second; the default access log would bury the
            # operator's console.
            del fmt, args

    return Handler


if __name__ == "__main__":
    raise SystemExit(main())
