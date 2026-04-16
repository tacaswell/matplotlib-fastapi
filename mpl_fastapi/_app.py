"""Importable app factory used by uvicorn's reload/worker mode.

When ``python -m mpl_fastapi`` is invoked with ``--reload`` or ``--workers``,
uvicorn requires an *import string* (``"module:attr"``) rather than an app
object.  This module provides the ``app`` attribute at a stable, importable
path so uvicorn can re-import it in each worker.

The plots target is communicated via the ``_MPL_FASTAPI_TARGET`` environment
variable, which ``__main__.py`` sets before handing off to uvicorn.
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

# Configure logging for mpl_fastapi to print startup messages (URLs, tokens) by default
# Use uvicorn's DefaultFormatter for visual consistency with uvicorn's own logs
_logger = logging.getLogger("mpl_fastapi")
_logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(DefaultFormatter("%(levelprefix)s %(message)s"))
_logger.addHandler(_handler)

app = build_app(_load_plots(_target))
