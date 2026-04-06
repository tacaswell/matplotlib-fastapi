"""Hidden-complexity server factory for mpl-fastapi.

This module builds a fully-configured FastAPI application from a plain dict
of :class:`~mpl_fastapi.PlotConfig` objects.  Auth, middleware, static files,
and lifespan management are all handled internally so users never have to
touch them.

Typical usage (run via ``python -m mpl_fastapi``):

    # my_plots.py
    from mpl_fastapi import PlotConfig, InitConfig
    plots = {"sine": PlotConfig(...)}

    # shell
    python -m mpl_fastapi my_plots.py
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping
from typing import Any

from fastapi import FastAPI
from starlette.middleware.base import BaseHTTPMiddleware

from mpl_fastapi.auth import COOKIE_NAME, SingleUserToken
from mpl_fastapi.router import PlotConfig, create_mpl_router, install_mpl_router

logger = logging.getLogger(__name__)


def build_app(
    plots: Mapping[str, PlotConfig],
    *,
    title: str = "Matplotlib FastAPI",
    description: str = "Interactive matplotlib plots served via FastAPI and WebSockets",
    version: str = "1.0.0",
    prefix: str = "/plots",
    auth: SingleUserToken | None = None,
) -> FastAPI:
    """Build a fully-configured FastAPI application from a plots mapping.

    Parameters
    ----------
    plots:
        Mapping of plot name → :class:`~mpl_fastapi.PlotConfig`.
    title:
        FastAPI application title shown in the OpenAPI docs.
    description:
        FastAPI application description shown in the OpenAPI docs.
    version:
        Application version string.
    prefix:
        URL prefix under which all plot routes are mounted.
    auth:
        Authentication policy.  Defaults to a :class:`SingleUserToken` whose
        token comes from the ``MPL_FASTAPI_TOKEN`` environment variable (or a
        freshly-generated random token when the variable is absent).

    Returns
    -------
    FastAPI
        A ready-to-serve ASGI application.
    """
    if auth is None:
        auth = SingleUserToken()

    mpl = create_mpl_router(plots, auth=auth)

    app = FastAPI(title=title, description=description, version=version)

    # Set an auth cookie whenever ?token= appears on any request, not just
    # the protected plot routes.  This lets unprotected pages (index, etc.)
    # propagate the token for subsequent navigations.
    class _TokenCookieMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            token = request.query_params.get("token")
            if token and secrets.compare_digest(token, auth.token):
                response.set_cookie(
                    key=COOKIE_NAME,
                    value=token,
                    httponly=True,
                    samesite="lax",
                    path="/",
                )
            return response

    app.add_middleware(_TokenCookieMiddleware)

    install_mpl_router(app, mpl, prefix=prefix)

    # Log convenience URLs at startup.
    _host = os.environ.get("HOST", "127.0.0.1")
    _port = int(os.environ.get("PORT", "8000"))

    @app.on_event("startup")
    async def _log_urls() -> None:  # noqa: RUF029
        base = f"http://{_host}:{_port}"
        logger.info("Plots list:  %s%s/?token=%s", base, prefix, auth.token)
        for name in plots:
            logger.info("  %-20s %s%s/plot/%s?token=%s", name, base, prefix, name, auth.token)
        logger.info("Auth token:  %s", auth.token)

    return app
