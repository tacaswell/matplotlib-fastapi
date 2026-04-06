"""
Qt thin-client demo — view remote Matplotlib plots via WebSocket.

This script is a thin shim around the ``mpl-fastapi-qt`` entry point that is
now built into the package.  See :mod:`mpl_fastapi.remote.backend_qtremote`
for full documentation.

Prerequisites
-------------
1. Start the demo server (from the repo root)::

       uvicorn demos.demo_server:app --reload

2. Run this script (or use the installed ``mpl-fastapi-qt`` command)::

       python demos/qt_remote_demo.py
       python -m mpl_fastapi.remote.backend_qtremote

   Both accept the same arguments; see ``--help`` for details.

Authentication
--------------
Supply a token with ``--token`` or by setting the
``MPL_FASTAPI_TOKEN`` environment variable.
"""

from mpl_fastapi.remote.backend_qtremote import _main

if __name__ == "__main__":
    _main()
