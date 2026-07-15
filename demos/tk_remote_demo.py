"""
Tk thin-client demo — view remote Matplotlib plots via WebSocket.

This script is a thin shim around the ``mpl-fastapi-tk`` entry point that is
now built into the package.  See :mod:`mpl_fastapi.remote.backend_tkremote`
for full documentation.

Prerequisites
-------------
1. Start the demo server (from the repo root)::

       uvicorn demos.demo_server:app --reload

2. Run this script (or use the installed ``mpl-fastapi-tk`` command)::

       python demos/tk_remote_demo.py
       python -m mpl_fastapi.remote.backend_tkremote

   Both accept the same arguments; see ``--help`` for details.

Authentication
--------------
Supply a token with ``--token`` or by setting the
``MPL_FASTAPI_TOKEN`` environment variable.
"""

from mpl_fastapi.remote.backend_tkremote import _main

if __name__ == "__main__":
    _main()
