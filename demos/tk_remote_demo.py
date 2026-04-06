"""
Tk thin-client demo — view remote Matplotlib plots via WebSocket.

This script connects to a running mpl_fastapi server and displays
one or more plots in native Tk windows.  All rendering happens on the
server; the client only displays images and forwards user interactions.

Prerequisites
-------------
1. Start the demo server (from the repo root)::

       uvicorn demos.demo_server:app --reload

2. Run this script::

       python demos/tk_remote_demo.py

   This opens a **launcher window** that lists all available plots on
   the server.  Select a plot, fill in parameters, and click Launch.

   Or open a specific plot directly *and* show the launcher::

       python demos/tk_remote_demo.py --plot sine frequency=3.0

   Or with update parameters (applied after init)::

       python demos/tk_remote_demo.py --plot interactive_sine \\
           frequency=2.0 amplitude=1.5 --update phase=1.57

What you should see
-------------------
- A launcher window listing all available plots on the server.
- Selecting a plot shows its init- and update-parameter forms.
- Clicking Launch opens a Tk window with the server-rendered plot.
- Pan, zoom, home, back/forward buttons all work — they send commands
  to the server, which re-renders and streams back the updated image.
- Mouse coordinates appear in the toolbar status bar.
- The zoom rubberband rectangle is drawn locally from server messages.
- Closing a window disconnects that figure's WebSocket cleanly.
"""

from __future__ import annotations

import argparse
import os
import sys

from mpl_fastapi.remote.backend_tkremote import (
    open_launcher,
    open_remote_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tk thin-client viewer for remote mpl_fastapi plots",
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
            "Plot name to open directly.  The launcher window is "
            "always shown; this pre-opens the named plot as well."
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

    # Always show the launcher window
    print(f"Opening launcher for {args.url} ...")
    launcher = open_launcher(args.url, token=token)

    # If --plot was given, also pre-open that figure directly
    if args.plot:
        print(f"Pre-opening plot {args.plot!r} ...")
        try:
            mgr = open_remote_figure(
                url=args.url,
                plot_name=args.plot,
                init_params=init_params or None,
                update_params=update_params or None,
                token=token,
                master=launcher,
            )
            mgr.show()
            # Track in launcher so it cleans up on close
            launcher._managers.append(mgr)
        except RuntimeError as exc:
            print(f"Warning: could not open {args.plot!r}: {exc}")

    launcher.mainloop()


if __name__ == "__main__":
    main()
