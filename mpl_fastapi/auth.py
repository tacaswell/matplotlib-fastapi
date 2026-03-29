"""Pluggable authentication and authorization for mpl_fastapi routers.

This module provides an :class:`AuthPolicy` protocol and two built-in
implementations:

* :class:`NoAuth` – accepts every request (the default).
* :class:`SingleUserToken` – validates a bearer token supplied via the
  ``Authorization`` header **or** a ``token`` query parameter.

Custom policies
---------------
Any object that satisfies the :class:`AuthPolicy` protocol can be passed
to :func:`~mpl_fastapi.create_mpl_router` via the ``auth`` keyword.
The object must expose two methods that return FastAPI dependencies:

* ``http_dependency`` – for regular HTTP endpoints.
* ``ws_dependency`` – for WebSocket endpoints (must work with
  ``WebSocket`` instead of ``Request``).

Both dependencies should raise ``HTTPException(status_code=401)`` (HTTP)
or call ``websocket.close(code=1008)`` and raise ``WebSocketException``
(WebSocket) on authentication failure.
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from fastapi import HTTPException, Request, WebSocket, status
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AuthPolicy(Protocol):
    """Protocol that auth policies must satisfy.

    Implementations provide two FastAPI-compatible dependency callables:
    one for regular HTTP routes and one for WebSocket routes.
    """

    def http_dependency(self) -> Callable[..., Any]:
        """Return a FastAPI dependency for HTTP endpoints.

        The dependency receives a :class:`~fastapi.Request` and should
        raise ``HTTPException(401)`` if the request is not authorised.
        """
        ...

    def ws_dependency(self) -> Callable[..., Any]:
        """Return a FastAPI dependency for WebSocket endpoints.

        The dependency receives a :class:`~fastapi.WebSocket` and should
        close the connection with code 1008 and raise if the request is
        not authorised.  This runs **before** the WebSocket is accepted.
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
                    "No token configured – generated ephemeral token: %s",
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

        async def _verify(request: Request) -> None:
            token = self._extract_token(
                authorization=request.headers.get("authorization"),
                query_token=request.query_params.get("token"),
            )
            if not token or not secrets.compare_digest(token, expected):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing authentication token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        return _verify

    def ws_dependency(self) -> Callable[..., Any]:
        expected = self._token

        async def _verify(websocket: WebSocket) -> None:
            token = self._extract_token(
                authorization=websocket.headers.get("authorization"),
                query_token=websocket.query_params.get("token"),
            )
            if not token or not secrets.compare_digest(token, expected):
                # Reject before accepting — Starlette sends a 403 HTTP
                # response, which prevents the upgrade.
                await websocket.close(code=1008)
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Invalid or missing authentication token",
                )

        return _verify
