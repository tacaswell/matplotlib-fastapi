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
import sys

from PySide6.QtWidgets import QApplication

from mpl_fastapi.remote.backend_qtremote import open_remote_figure


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
    args = parser.parse_args()

    # Parse key=value pairs into a dict
    init_params: dict[str, str] = {}
    for kv in args.params:
        if "=" not in kv:
            parser.error(f"Parameters must be KEY=VALUE, got: {kv!r}")
        key, value = kv.split("=", 1)
        init_params[key] = value

    # Ensure a QApplication exists
    app = QApplication.instance() or QApplication(sys.argv)

    managers = []

    if args.plot:
        # Open a single named plot
        print(f"Connecting to {args.url} / {args.plot} ...")
        mgr = open_remote_figure(
            url=args.url,
            plot_name=args.plot,
            init_params=init_params or None,
        )
        mgr.show()
        managers.append(mgr)
    else:
        # Open several demo plots to show multi-figure support
        demos = [
            ("sine", {"frequency": "2.0", "amplitude": "1.5"}),
            ("cosine", {"damping": "0.3"}),
            ("lissajous", {"freq_x": "3", "freq_y": "2", "delta": "1.57"}),
        ]
        for plot_name, params in demos:
            print(f"Connecting to {args.url} / {plot_name} ...")
            try:
                mgr = open_remote_figure(
                    url=args.url,
                    plot_name=plot_name,
                    init_params=params,
                )
                mgr.show()
                managers.append(mgr)
            except RuntimeError as exc:
                print(f"  ⚠ Could not open {plot_name!r}: {exc}")

    if not managers:
        print("No figures opened — is the demo server running?")
        print("  uvicorn demos.demo_server:app --reload")
        sys.exit(1)

    print(f"\n{len(managers)} figure(s) opened.  Close all windows to exit.")
    app.exec()


if __name__ == "__main__":
    main()
