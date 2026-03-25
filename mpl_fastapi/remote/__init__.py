"""Remote thin-client Matplotlib backend.

This package provides a Matplotlib backend where rendering happens on a
remote mpl_fastapi server and the local GUI toolkit only displays images
and forwards user interaction events.

This is object-oriented only — there is no pyplot integration.

Usage (Qt)::

    from mpl_fastapi.remote.backend_qtremote import open_remote_figure

    manager = open_remote_figure(
        url="ws://localhost:8000/plots",
        plot_name="sine",
        init_params={"frequency": 2.0},
    )
    manager.show()

Batch usage::

    from mpl_fastapi.remote.backend_qtremote import (
        open_remote_figures,
        run_qt_app,
    )

    managers = open_remote_figures([
        ("ws://localhost:8000/plots", "sine", {"frequency": 2.0}),
        ("ws://localhost:8000/plots", "cosine"),
    ])
    run_qt_app(managers)
"""

from mpl_fastapi.remote.transport import RemoteTransport, ServerConfig

__all__ = [
    "RemoteTransport",
    "ServerConfig",
]
