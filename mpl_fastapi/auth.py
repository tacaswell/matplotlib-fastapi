"""Pluggable authentication and authorization for mpl_fastapi routers.

This module provides an :class:`AuthPolicy` protocol and two built-in
implementations:

* :class:`NoAuth` - accepts every request (the default).
* :class:`SingleUserToken` - validates a bearer token supplied via the
  ``Authorization`` header **or** a ``token`` query parameter.

Custom policies
---------------
Any object that satisfies the :class:`AuthPolicy` protocol can be passed
to :func:`~mpl_fastapi.create_mpl_router` via the ``auth`` keyword.
The object must expose two methods that return FastAPI dependencies:

* ``http_dependency`` - for regular HTTP endpoints.
* ``ws_dependency`` - for WebSocket endpoints (must work with
  ``WebSocket`` instead of ``Request``).

Both dependencies should raise ``HTTPException(status_code=401)`` (HTTP)
or call ``websocket.close(code=1008)`` and raise ``WebSocketException``
(WebSocket) on authentication failure.  On success a dependency may
**return a principal** (see :data:`Principal`); the router captures the
WebSocket dependency's return value and threads it into user plot
functions via the connection context.  ``NoAuth`` returns ``None``.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fastapi import Depends, HTTPException, Request, Response, WebSocket, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Principal
# ---------------------------------------------------------------------------

# The value an :class:`AuthPolicy` dependency returns on success.  It is
# captured by the router and threaded (via the connection context) into user
# plot functions.  Kept as ``Any`` at the framework boundary: real AuthN/AuthZ
# policies return whatever principal type they like (e.g. a struct carrying the
# user's token and claims for on-behalf-of flows), and user code annotates its
# own ``context.principal`` for type checking.
Principal = Any


@dataclass(frozen=True)
class TokenPrincipal:
    """Minimal principal produced by :class:`SingleUserToken`.

    ``SingleUserToken`` authenticates against a single shared secret, so there
    is no distinct per-user identity behind it.  This placeholder principal
    simply records that a valid token was presented; real policies return a
    richer object.
    """

    kind: str = "single-user"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AuthPolicy(Protocol):
    """Protocol that auth policies must satisfy.

    Implementations provide two FastAPI-compatible dependency callables:
    one for regular HTTP routes and one for WebSocket routes.

    On success, a dependency may **return a principal** (see
    :data:`Principal`).  The router captures the WebSocket dependency's return
    value and makes it available to user plot functions via the connection
    context, so a policy can surface the authenticated user's identity/token
    (e.g. for on-behalf-of flows).  Returning ``None`` means "no identity".
    """

    def http_dependency(self) -> Callable[..., Any]:
        """Return a FastAPI dependency for HTTP endpoints.

        The dependency receives a :class:`~fastapi.Request` and should
        raise ``HTTPException(401)`` if the request is not authorised.
        On success it may return a principal (or ``None``).
        """
        ...

    def ws_dependency(self) -> Callable[..., Any]:
        """Return a FastAPI dependency for WebSocket endpoints.

        The dependency receives a :class:`~fastapi.WebSocket` and should
        close the connection with code 1008 and raise if the request is
        not authorised.  This runs **before** the WebSocket is accepted.
        On success it may return a principal (or ``None``); the router
        threads that value into user plot functions via the connection
        context.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in: no authentication
# ---------------------------------------------------------------------------


class NoAuth:
    """Accept every request without authentication (the default)."""

    def http_dependency(self) -> Callable[..., Any]:
        async def _noop() -> None:
            pass

        return _noop

    def ws_dependency(self) -> Callable[..., Any]:
        async def _noop() -> None:
            pass

        return _noop


# ---------------------------------------------------------------------------
# Built-in: single-user bearer token
# ---------------------------------------------------------------------------

# Environment variable to set the shared secret.
_TOKEN_ENV_VAR = "MPL_FASTAPI_TOKEN"

# Cookie name used by SingleUserToken to persist the auth token in the browser.
COOKIE_NAME = "mpl_fastapi_token"


class SingleUserToken:
    """Validate a single shared bearer token.

    The token is resolved in the following order:

    1. Explicit *token* argument passed to the constructor.
    2. The ``MPL_FASTAPI_TOKEN`` environment variable.
    3. A cryptographically random 32-character URL-safe token generated
       at instantiation time (logged at ``WARNING`` level so the
       operator can retrieve it).

    Clients may supply the token in **either** of two ways:

    * ``Authorization: Bearer <token>`` header (preferred).
    * ``?token=<token>`` query parameter (necessary for browser
      WebSocket connections that cannot set custom headers).

    Parameters
    ----------
    token : str or None
        Explicit token value.  When *None*, the environment variable or
        an auto-generated token is used.
    """

    def __init__(self, token: str | None = None) -> None:
        if token is not None:
            self._token = token
        else:
            env_token = os.environ.get(_TOKEN_ENV_VAR, "").strip()
            if env_token:
                self._token = env_token
            else:
                self._token = secrets.token_urlsafe(24)
                logger.warning(
                    "No token configured - generated ephemeral token: %s",
                    self._token,
                )

    @property
    def token(self) -> str:
        """The active token value (read-only)."""
        return self._token

    # -- internal helpers ---------------------------------------------------

    def _extract_token(
        self,
        *,
        authorization: str | None = None,
        query_token: str | None = None,
    ) -> str | None:
        """Extract a bearer token from header or query parameter."""
        if authorization:
            parts = authorization.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1]
        return query_token

    # -- dependency factories -----------------------------------------------

    def http_dependency(self) -> Callable[..., Any]:
        expected = self._token
        _bearer = HTTPBearer(auto_error=False)

        async def _verify(
            request: Request,
            response: Response,
            credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
        ) -> TokenPrincipal:
            # Prefer the bearer token from the Authorization header
            # (parsed by HTTPBearer), fall back to ?token= query param,
            # then to the browser cookie.
            token: str | None = None
            from_query = False
            if credentials is not None:
                token = credentials.credentials
            else:
                token = request.query_params.get("token")
                if token:
                    from_query = True
                else:
                    token = request.cookies.get(COOKIE_NAME)
            if not token or not secrets.compare_digest(token, expected):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing authentication token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Persist as cookie so subsequent navigations just work.
            if from_query:
                response.set_cookie(
                    key=COOKIE_NAME,
                    value=token,
                    httponly=True,
                    samesite="lax",
                    path="/",
                )
            return TokenPrincipal()

        return _verify

    def ws_dependency(self) -> Callable[..., Any]:
        expected = self._token

        async def _verify(websocket: WebSocket) -> TokenPrincipal:
            token = self._extract_token(
                authorization=websocket.headers.get("authorization"),
                query_token=websocket.query_params.get("token"),
            )
            if token is None:
                token = websocket.cookies.get(COOKIE_NAME)
            if not token or not secrets.compare_digest(token, expected):
                # Raise HTTPException — Starlette's dependency resolver
                # converts this into an HTTP 403 denial before the
                # WebSocket upgrade, so the connection is never accepted.
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or missing authentication token",
                )
            return TokenPrincipal()

        return _verify
