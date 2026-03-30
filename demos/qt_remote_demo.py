"""
Qt thin-client demo — view remote Matplotlib plots via WebSocket.

This script connects to a running mpl_fastapi server and displays
one or more plots in native Qt windows.  All rendering happens on the
server; the client only displays images and forwards user interactions.

Prerequisites
-------------
1. Start the demo server (from the repo root)::

       uvicorn demos.demo_server:app --reload

2. Run this script::

       python demos/qt_remote_demo.py

   Or with custom parameters::

       python demos/qt_remote_demo.py --url ws://localhost:8000/plots \\
           --plot sine --frequency 3.0 --amplitude 2.0

   Or with update parameters (applied after init)::

       python demos/qt_remote_demo.py --plot interactive_sine \\
           --frequency 2.0 --amplitude 1.5 --update phase=1.57

What you should see
-------------------
- One or more Qt windows appear, each showing a server-rendered plot.
- Pan, zoom, home, back/forward buttons all work — they send commands
  to the server, which re-renders and streams back the updated image.
- Mouse coordinates appear in the toolbar status bar.
- The zoom rubberband rectangle is drawn locally from server messages.
- Closing a window disconnects that figure's WebSocket cleanly.
"""

from __future__ import annotations

import argparse
import os

from mpl_fastapi.remote.backend_qtremote import (
    run_qt_app,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qt thin-client viewer for remote mpl_fastapi plots",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/plots",
        help="Server WebSocket base URL (default: ws://localhost:8000/plots)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help=(
            "Plot name to open (e.g. 'sine', 'cosine', 'lissajous').  "
            "If omitted, opens several demo plots."
        ),
    )
    # Allow arbitrary extra keyword arguments forwarded as init_params
    parser.add_argument(
        "params",
        nargs="*",
        metavar="KEY=VALUE",
        help="Init parameters as key=value pairs (e.g. frequency=3.0)",
    )
    parser.add_argument(
        "--update",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Update parameters as key=value pairs (e.g. phase=1.57)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Authentication token. Defaults to the MPL_FASTAPI_TOKEN "
            "environment variable if set."
        ),
    )
    args = parser.parse_args()

    # Resolve auth token: CLI arg > environment variable
    token: str | None = args.token
    if token is None:
        token = os.environ.get("MPL_FASTAPI_TOKEN", "").strip() or None

    # Parse key=value pairs into a dict
    init_params: dict[str, str] = {}
    for kv in args.params:
        if "=" not in kv:
            parser.error(f"Parameters must be KEY=VALUE, got: {kv!r}")
        key, value = kv.split("=", 1)
        init_params[key] = value

    update_params: dict[str, str] = {}
    for kv in args.update:
        if "=" not in kv:
            parser.error(f"Update parameters must be KEY=VALUE, got: {kv!r}")
        key, value = kv.split("=", 1)
        update_params[key] = value

    if args.plot:
        # Open a single named plot
        specs = [
            (args.url, args.plot, init_params or None, update_params or None)
        ]
    else:
        # Open several demo plots to show multi-figure support
        specs = [
            (args.url, "sine", {"frequency": "2.0", "amplitude": "1.5"}),
            (args.url, "interactive_sine", {"frequency": "2.0", "amplitude": "1.5"}),
            (args.url, "cosine", {"damping": "0.3"}),
            (args.url, "lissajous", {"freq_x": "3", "freq_y": "2", "delta": "1.57"}),
        ]

    print(f"Opening {len(specs)} plot(s) on {args.url} ...")
    run_qt_app(specs, token=token)


if __name__ == "__main__":
    main()
