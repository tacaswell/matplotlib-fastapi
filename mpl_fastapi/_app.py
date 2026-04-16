"""Importable app factory used by uvicorn's reload/worker mode.

When ``python -m mpl_fastapi`` is invoked with ``--reload`` or ``--workers``,
uvicorn requires an *import string* (``"module:attr"``) rather than an app
object.  This module provides the ``app`` attribute at a stable, importable
path so uvicorn can re-import it in each worker.

The plots target is communicated via the ``_MPL_FASTAPI_TARGET`` environment
variable, which ``__main__.py`` sets before handing off to uvicorn.

Authentication is controlled by:
- ``_MPL_FASTAPI_NO_AUTH`` env var (set by ``--no-auth`` flag)
- ``MPL_FASTAPI_TOKEN`` env var (if empty string, disables auth)
"""

from __future__ import annotations

import os
import sys

_target = os.environ.get("_MPL_FASTAPI_TARGET")
if not _target:
    print(
        "error: _MPL_FASTAPI_TARGET is not set — "
        "do not import mpl_fastapi._app directly; use 'python -m mpl_fastapi'",
        file=sys.stderr,
    )
    sys.exit(1)

import logging  # noqa: E402

from uvicorn.logging import DefaultFormatter  # noqa: E402

from mpl_fastapi.__main__ import _load_plots  # noqa: E402
from mpl_fastapi._server import build_app  # noqa: E402
from mpl_fastapi.auth import NoAuth  # noqa: E402

# Configure logging for mpl_fastapi to print startup messages (URLs, tokens) by default
# Use uvicorn's DefaultFormatter for visual consistency with uvicorn's own logs
_logger = logging.getLogger("mpl_fastapi")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s"))
_logger.addHandler(_handler)

# Determine which auth policy to use
_auth = None
if os.environ.get("_MPL_FASTAPI_NO_AUTH") == "1":
    # --no-auth flag was passed
    _auth = NoAuth()
elif os.environ.get("MPL_FASTAPI_TOKEN") == "":
    # MPL_FASTAPI_TOKEN explicitly set to empty string
    _auth = NoAuth()
# else: _auth remains None, build_app will create SingleUserToken

app = build_app(_load_plots(_target), auth=_auth)
