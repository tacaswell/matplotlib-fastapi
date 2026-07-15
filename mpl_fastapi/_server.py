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
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from mpl_fastapi import __version__
from mpl_fastapi.auth import COOKIE_NAME, AuthPolicy, SingleUserToken
from mpl_fastapi.router import (
    ConnectionInfo,
    Lifespan,
    PlotConfig,
    compose_lifespans,
    create_mpl_router,
    install_mpl_router,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from contextlib import AbstractAsyncContextManager
    from typing import Any

    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.requests import Request
    from starlette.responses import Response

logger = logging.getLogger(__name__)


def build_app(
    plots: Mapping[str, PlotConfig],
    *,
    title: str = "Matplotlib FastAPI",
    description: str = "Interactive matplotlib plots served via FastAPI and WebSockets",
    version: str | None = None,
    prefix: str = "/plots",
    auth: AuthPolicy | None = None,
    allowed_origins: list[str] | None = None,
    lifespan: Lifespan | None = None,
    context_factory: (
        Callable[[ConnectionInfo], AbstractAsyncContextManager[Any]] | None
    ) = None,
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
        Application version string shown in the OpenAPI docs.  When ``None``
        (the default) the installed ``mpl_fastapi`` package version is used.
    prefix:
        URL prefix under which all plot routes are mounted.
    auth:
        Authentication policy. Options:

        - ``None`` (default): Creates a :class:`SingleUserToken` whose token
          comes from the ``MPL_FASTAPI_TOKEN`` environment variable (or a
          freshly-generated random token when the variable is absent).
        - :class:`NoAuth`: Accepts all requests without authentication.
        - :class:`SingleUserToken`: Single-user bearer token authentication.
        - Any custom object satisfying the :class:`~mpl_fastapi.auth.AuthPolicy`
          protocol.

    allowed_origins:
        Explicit list of origins permitted to make cross-origin requests, e.g.
        ``["https://example.com", "http://localhost:3000"]``.  When ``None``
        (the default) the ``BACKEND_CORS_ORIGINS`` environment variable is
        read and split on commas.  If neither is provided no cross-origin
        access is allowed.  Never pass ``["*"]`` — the cookie-based auth
        token cannot be sent with credentialled wildcard requests and a
        wildcard defeats the purpose of the token entirely.

    lifespan:
        Optional additional lifespan context manager for custom startup/shutdown
        logic.  Will be composed with the internal lifespan handlers (figure
        executor cleanup and URL logging).  Use this to mount additional routes,
        initialize resources, or perform other startup tasks.

    context_factory:
        Optional per-connection context factory forwarded verbatim to
        :func:`~mpl_fastapi.create_mpl_router`.  See its documentation for the
        expected async-context-manager shape.

    Returns
    -------
    FastAPI
        A ready-to-serve ASGI application.
    """
    if auth is None:
        auth = SingleUserToken()

    if version is None:
        version = __version__

    # Resolve origins from env var when not supplied programmatically.
    # Wildcards are forbidden: credentialled requests (cookies) require an
    # exact origin match and a wildcard defeats the token auth entirely.
    if allowed_origins is None:
        raw = os.environ.get("BACKEND_CORS_ORIGINS", "")
        allowed_origins = [o.strip() for o in raw.split(",") if o.strip()]
    if "*" in allowed_origins:
        raise ValueError(
            "allowed_origins must not contain '*': wildcard CORS is incompatible "
            "with cookie-based authentication.  List exact origins instead."
        )

    mpl = create_mpl_router(
        plots,
        auth=auth,
        allowed_origins=allowed_origins,
        context_factory=context_factory,
    )

    app = FastAPI(title=title, description=description, version=version)

    # HTTP CORS — same origin list as the WebSocket check above.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization"],
    )

    # Set an auth cookie whenever ?token= appears on any request, not just
    # the protected plot routes.  This lets unprotected pages (index, etc.)
    # propagate the token for subsequent navigations.
    # Only add this middleware if auth is SingleUserToken (which has a token attribute).
    if isinstance(auth, SingleUserToken):
        # Bind the token to a local so the closure below sees a plain ``str``
        # rather than ``AuthPolicy | None`` (mypy drops the isinstance
        # narrowing across the nested-class method boundary).
        expected_token = auth.token

        class _TokenCookieMiddleware(BaseHTTPMiddleware):
            async def dispatch(
                self, request: Request, call_next: RequestResponseEndpoint
            ) -> Response:
                response = await call_next(request)
                token = request.query_params.get("token")
                if token and secrets.compare_digest(token, expected_token):
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

    # Log convenience URLs at startup via a lifespan handler (the modern
    # replacement for the deprecated ``@app.on_event("startup")``).
    _host = os.environ.get("HOST", "127.0.0.1")
    _port = int(os.environ.get("PORT", "8000"))

    def _build_default_params(plot_config: PlotConfig) -> tuple[str, bool]:
        """Build query string with default/placeholder parameter values.

        Returns
        -------
        params_str : str
            URL-encoded query string with parameter values (defaults or placeholders).
        has_required : bool
            True if any parameters are required (have no defaults).
        """
        from urllib.parse import urlencode

        from pydantic_core import PydanticUndefined

        # Get the Pydantic model's default values
        model = plot_config.init.params_model
        params = {}
        has_required = False

        # Use model_fields (Pydantic v2) to extract defaults or create placeholders
        for field_name, field_info in model.model_fields.items():
            if field_info.default is not PydanticUndefined:
                # Has a default value - use it
                params[field_name] = str(field_info.default)
            elif field_info.default_factory is not None:
                # Has a default_factory - try to use it
                try:
                    default_val = field_info.default_factory()
                    params[field_name] = str(default_val)
                except Exception:
                    # Factory failed - use placeholder
                    params[field_name] = f"<{field_name}>"
                    has_required = True
            else:
                # No default - use placeholder based on type
                has_required = True
                # Try to infer a sensible placeholder from the type annotation
                annotation = field_info.annotation
                if annotation == str or (
                    hasattr(annotation, "__origin__") and annotation.__origin__ is str
                ):
                    params[field_name] = f"<{field_name}>"
                elif annotation in (int, float) or (
                    hasattr(annotation, "__origin__")
                    and annotation.__origin__ in (int, float)
                ):
                    params[field_name] = f"<{field_name}>"
                else:
                    params[field_name] = f"<{field_name}>"

        params_str = urlencode(params) if params else ""
        return params_str, has_required

    @asynccontextmanager
    async def _log_urls_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        base = f"http://{_host}:{_port}"
        if isinstance(auth, SingleUserToken):
            logger.info("Plots list:  %s%s/?token=%s", base, prefix, auth.token)
            for name, plot_config in plots.items():
                params, has_required = _build_default_params(plot_config)
                param_sep = "&" if params else ""
                url = (
                    f"{base}{prefix}/plot/{name}?token={auth.token}{param_sep}{params}"
                )
                if has_required:
                    logger.info(
                        "  %-20s %s  [edit placeholders: %s]", name, url, params
                    )
                else:
                    logger.info("  %-20s %s", name, url)
            logger.info("Auth token:  %s", auth.token)
        else:
            logger.info("Plots list:  %s%s/", base, prefix)
            for name, plot_config in plots.items():
                params, has_required = _build_default_params(plot_config)
                param_sep = "?" if params else ""
                url = f"{base}{prefix}/plot/{name}{param_sep}{params}"
                if has_required:
                    logger.info(
                        "  %-20s %s  [edit placeholders: %s]", name, url, params
                    )
                else:
                    logger.info("  %-20s %s", name, url)
        yield

    # Compose the logging lifespan with whatever install_mpl_router already
    # set (the figure-executor cleanup lifespan) so all hooks run correctly.
    # If a user-provided lifespan was passed, compose it in as well.
    existing_lifespan = app.router.lifespan_context
    lifespans_to_compose = [existing_lifespan, _log_urls_lifespan]
    if lifespan is not None:
        lifespans_to_compose.append(lifespan)
    app.router.lifespan_context = compose_lifespans(*lifespans_to_compose)

    return app
