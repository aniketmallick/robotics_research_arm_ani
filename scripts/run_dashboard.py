#!/usr/bin/env python
"""ARM-ANI's mind, on a screen. Read-only — it never touches the arm.

    python scripts/run_dashboard.py                    # live: tail the decision log
    python scripts/run_dashboard.py --replay           # replay the live log's picks
    python scripts/run_dashboard.py --replay logs/decisions_dev.jsonl
    python scripts/run_dashboard.py --port 9000

Then open http://localhost:8770 and put it on the projector.

Deliberately built on the standard library's http.server: fastapi and flask are
NOT installed in this env, and the last thing anyone should do the morning of a
demo is pip-install a web framework. Zero new dependencies, nothing to go wrong.

The page shows the frame perception last looked at, the marked zones, the
current gated pick with its confidence against the approval line, which gate
fired, and a scrolling audit trail. All of it derives from the decision log, so
it can only ever show what actually happened.

--replay is the insurance policy: if the venue's wifi or the Gemini quota dies
mid-pitch, this tells the whole trust story from a log recorded earlier.
"""

from __future__ import annotations

import argparse
import json
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

from armani import config, dashboard  # noqa: E402
from armani.logutil import get_logger  # noqa: E402

log = get_logger("run_dashboard")

PAGE = (REPO_ROOT / "armani" / "data" / "dashboard.html")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=config.DASHBOARD_PORT)
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="bind address. Localhost by default: the projector is this machine, and "
             "binding 0.0.0.0 would serve the decision log to the whole venue wifi. "
             "Pass 0.0.0.0 deliberately if you want it on other devices.",
    )
    parser.add_argument(
        "--replay", nargs="?", const=str(config.DECISION_LOG), metavar="LOG",
        help="cycle through a log's picks instead of following the live tail",
    )
    parser.add_argument("--interval", type=float, default=4.0, help="replay seconds per pick")
    parser.add_argument("--dry-run", action="store_true", help="check the sources and exit")
    args = parser.parse_args()

    source = dashboard.Source(
        path=Path(args.replay) if args.replay else config.DECISION_LOG,
        replay=bool(args.replay),
        interval_s=args.interval,
    )

    print("=== ARM-ANI dashboard ===")
    print(f"source : {source.path}  ({source.label()})")
    print(f"frame  : {config.LAST_FRAME_PATH}")

    if not source.path.is_file():
        print(f"\nno decision log at {source.path}", file=sys.stderr)
        if not args.dry_run:
            print("Run the agent or a smoke test first, or pass --replay <log>.", file=sys.stderr)
            return 1

    records = dashboard.read_records(source)
    picks = [r for r in records if r.get("kind") == "gated_pick"]
    print(f"records: {len(records)} in the tail, {len(picks)} gated picks")
    if not PAGE.is_file():
        print(f"\nmissing page template: {PAGE}", file=sys.stderr)
        return 1

    if args.dry_run:
        state = dashboard.build_state(source)
        print(f"[dry-run] state builds: pick={state['pick'] and state['pick']['headline']!r}, "
              f"{len(state['zones'])} zones, {len(state['feed'])} feed rows")
        print(f"[dry-run] would serve on http://localhost:{args.port}")
        return 0

    server = ThreadingHTTPServer((args.host, args.port), _handler_for(source))
    print(f"\n  OPEN:  http://localhost:{args.port}   (bound to {args.host})\n")
    print("  Read-only: this never touches the arm. Ctrl-C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        server.server_close()
    return 0


def _handler_for(source: dashboard.Source):
    """A request handler bound to one log source."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            route = self.path.split("?", 1)[0]
            try:
                if route in ("/", "/index.html"):
                    self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
                elif route == "/api/state":
                    payload = json.dumps(dashboard.build_state(source)).encode("utf-8")
                    self._send(payload, "application/json", cache=False)
                elif route == "/api/frame.jpg":
                    self._send_frame()
                else:
                    self._send(b"not found", "text/plain", status=404)
            except BrokenPipeError:
                pass  # the browser navigated away mid-response; not our problem
            except Exception as exc:
                log.error("dashboard request %s failed: %s", route, exc)
                try:
                    self._send(str(exc).encode("utf-8"), "text/plain", status=500)
                except Exception:
                    pass

        def _send_frame(self) -> None:
            try:
                data = config.LAST_FRAME_PATH.read_bytes()
            except OSError:
                # No frame yet is normal before the first look. A 204 lets the
                # page keep whatever it is showing instead of flashing a broken
                # image at the audience.
                self._send(b"", "image/jpeg", status=204, cache=False)
                return
            self._send(data, "image/jpeg", cache=False)

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
            # The default handler prints a line per poll; at 2 Hz that buries
            # the console the operator is also using.
            del fmt, args

    return Handler


if __name__ == "__main__":
    raise SystemExit(main())
