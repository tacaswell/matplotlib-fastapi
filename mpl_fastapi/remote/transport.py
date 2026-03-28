"""Async WebSocket transport for the mpl_fastapi v0 protocol.

This module provides :class:`RemoteTransport`, an async WebSocket client
that handles the v0 handshake (``init`` → ``config``), dispatches incoming
messages via callbacks, and offers a thread-safe ``send_json`` for use
from GUI main threads.

One transport instance is created per remote figure.

Example (standalone asyncio)::

    transport = RemoteTransport(
        url="ws://localhost:8000/plots/ws/v0/sine?frequency=2",
        on_binary=handle_image,
        on_json=handle_message,
        on_disconnect=handle_disconnect,
    )
    config = await transport.connect()
    # ... transport._receive_loop() runs until disconnect ...
    await transport.disconnect()
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode

import websockets
from websockets.asyncio.client import ClientConnection

from mpl_fastapi.ws_client import (
    PROTOCOL_VERSION,
    BinaryImageHeader,
    ImageFormat,
    ImageTypeMode,
    parse_binary_image,
)

__all__ = [
    "BinaryImageHeader",
    "ImageFormat",
    "ImageTypeMode",
    "RemoteTransport",
    "ServerConfig",
    "parse_binary_image",
]

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Parsed ``config`` message from the server handshake.

    Attributes
    ----------
    connection_id : str
        Unique identifier for this WebSocket session.
    protocol_version : int
        Protocol version negotiated with the server.
    figure_size : tuple[int, int]
        Figure size in CSS pixels ``(width, height)``.
    figure_dpi : float
        Figure DPI as reported by the server.
    figure_label : str
        Figure label / title.
    toolbar_items : list[list[str]]
        Toolbar button configuration ``[[name, tooltip, image, method], ...]``.
    save_formats : list[str]
        Available save formats (e.g. ``["png", "pdf", "svg"]``).
    default_save_format : str
        Default save format.
    image_format : str
        Image format for binary messages (e.g. ``"png"``).
    update_schema : dict[str, Any] | None
        JSON Schema for update parameters, or ``None`` if updates not supported.
    """

    connection_id: str
    protocol_version: int
    figure_size: tuple[int, int]
    figure_dpi: float
    figure_label: str
    toolbar_items: list[list[str]] = field(default_factory=list)
    save_formats: list[str] = field(default_factory=lambda: ["png"])
    default_save_format: str = "png"
    image_format: str = "png"
    update_schema: dict[str, Any] | None = None


def build_ws_url(
    base_url: str,
    plot_name: str,
    init_params: dict[str, Any] | None = None,
) -> str:
    """Build a WebSocket URL for the v0 endpoint.

    Parameters
    ----------
    base_url : str
        Server base URL (e.g. ``"ws://localhost:8000/plots"``).
    plot_name : str
        Name of the plot to connect to.
    init_params : dict, optional
        Query-string parameters for plot initialisation.

    Returns
    -------
    str
        Full WebSocket URL.
    """
    base = base_url.rstrip("/")
    url = f"{base}/ws/v0/{plot_name}"
    if init_params:
        url = f"{url}?{urlencode(init_params)}"
    return url


def _parse_config_message(msg: dict[str, Any]) -> ServerConfig:
    """Parse a server ``config`` message into a :class:`ServerConfig`.

    Parameters
    ----------
    msg : dict
        The raw JSON config message from the server.

    Returns
    -------
    ServerConfig
    """
    figure_cfg = msg.get("figure", {})
    size = figure_cfg.get("size", [640, 480])
    toolbar_cfg = msg.get("toolbar", {})
    save_cfg = msg.get("save", {})
    image_cfg = msg.get("image", {})

    return ServerConfig(
        connection_id=msg["connection_id"],
        protocol_version=msg.get("protocol_version", 0),
        figure_size=(size[0], size[1]),
        figure_dpi=figure_cfg.get("dpi", 100.0),
        figure_label=figure_cfg.get("label", ""),
        toolbar_items=toolbar_cfg.get("items", []),
        save_formats=save_cfg.get("formats", ["png"]),
        default_save_format=save_cfg.get("default_format", "png"),
        image_format=image_cfg.get("format", "png"),
        update_schema=msg.get("update_schema"),
    )


class RemoteTransport:
    """Async WebSocket transport for the mpl_fastapi v0 protocol.

    One instance per figure.  The transport manages its own asyncio event
    loop which may run on a dedicated thread (see ``_run``).

    Parameters
    ----------
    url : str
        Full WebSocket URL including ``/ws/v0/{plot_name}`` and query params.
    on_binary : callable
        ``on_binary(data: bytes)`` — called when a binary image arrives.
    on_json : callable
        ``on_json(msg: dict)`` — called when a JSON message arrives.
    on_disconnect : callable
        ``on_disconnect()`` — called when the WebSocket closes unexpectedly
        and no reconnect will be attempted (either reconnect is disabled or
        all attempts have been exhausted).
    on_reconnect : callable, optional
        ``on_reconnect(config: ServerConfig)`` — called after a successful
        reconnection.  The new :class:`ServerConfig` is passed so the
        canvas can reset its state.
    on_reconnecting : callable, optional
        ``on_reconnecting(attempt: int, max_attempts: int)`` — called at
        the start of each reconnection attempt, before the backoff sleep.
        Useful for updating a UI overlay.
    device_pixel_ratio : float
        Device pixel ratio sent in the ``init`` handshake message.
    reconnect_max_attempts : int
        Maximum number of reconnection attempts before giving up.
        Set to ``0`` to disable automatic reconnection.
    reconnect_initial_delay : float
        Backoff delay (seconds) before the first reconnection attempt.
    reconnect_max_delay : float
        Cap on the exponential backoff delay.
    reconnect_backoff_base : float
        Exponential base for backoff growth.
    """

    def __init__(
        self,
        url: str,
        *,
        on_binary: Callable[[bytes], None] | None = None,
        on_json: Callable[[dict[str, Any]], None] | None = None,
        on_disconnect: Callable[[], None] | None = None,
        on_reconnect: Callable[[ServerConfig], None] | None = None,
        on_reconnecting: Callable[[int, int], None] | None = None,
        device_pixel_ratio: float = 1.0,
        reconnect_max_attempts: int = 10,
        reconnect_initial_delay: float = 0.5,
        reconnect_max_delay: float = 15.0,
        reconnect_backoff_base: float = 2.0,
    ) -> None:
        self._url = url
        self._on_binary = on_binary or (lambda data: None)
        self._on_json = on_json or (lambda msg: None)
        self._on_disconnect = on_disconnect or (lambda: None)
        self._on_reconnect = on_reconnect
        self._on_reconnecting = on_reconnecting
        self._device_pixel_ratio = device_pixel_ratio

        self._ws: ClientConnection | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._server_config: ServerConfig | None = None
        self._explicitly_disconnecting = False

        # Reconnection parameters
        self._reconnect_max_attempts = reconnect_max_attempts
        self._reconnect_initial_delay = reconnect_initial_delay
        self._reconnect_max_delay = reconnect_max_delay
        self._reconnect_backoff_base = reconnect_backoff_base
        self._reconnect_task: asyncio.Task[None] | None = None
        self._reconnect_attempt = 0

    # -- properties ----------------------------------------------------------

    @property
    def server_config(self) -> ServerConfig | None:
        """Server configuration received during handshake, or ``None``."""
        return self._server_config

    @property
    def is_connected(self) -> bool:
        """Whether the WebSocket is currently open."""
        return self._ws is not None

    # -- lifecycle -----------------------------------------------------------

    async def connect(self) -> ServerConfig:
        """Open the WebSocket and perform the v0 handshake.

        Returns
        -------
        ServerConfig
            Parsed server configuration.

        Raises
        ------
        RuntimeError
            If the server returns an error or unexpected message.
        ValueError
            If the protocol version does not match.
        """
        self._loop = asyncio.get_running_loop()

        logger.info("Connecting to %s", self._url)
        self._ws = await websockets.connect(self._url)

        # --- v0 handshake: send init, receive config ---
        init_msg = {
            "type": "init",
            "protocol_version": PROTOCOL_VERSION,
            "device_pixel_ratio": self._device_pixel_ratio,
            "supports_binary": True,
        }
        await self._ws.send(json.dumps(init_msg))
        logger.debug("Sent init message")

        raw_config = await self._ws.recv()
        if isinstance(raw_config, bytes):
            raise RuntimeError("Expected JSON config message, got binary")
        config_msg: dict[str, Any] = json.loads(raw_config)

        if config_msg.get("type") == "error":
            raise RuntimeError(
                f"Server error during handshake: {config_msg.get('message')}"
            )
        if config_msg.get("type") != "config":
            raise RuntimeError(
                f"Expected 'config' message, got '{config_msg.get('type')}'"
            )

        server_version = config_msg.get("protocol_version")
        if server_version != PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol version mismatch: client={PROTOCOL_VERSION}, "
                f"server={server_version}"
            )

        self._server_config = _parse_config_message(config_msg)
        logger.info(
            "Connected: connection_id=%s, figure_size=%s",
            self._server_config.connection_id,
            self._server_config.figure_size,
        )
        return self._server_config

    async def disconnect(self) -> None:
        """Close the WebSocket connection cleanly."""
        self._explicitly_disconnecting = True

        # Cancel any pending reconnection attempt
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        if self._receive_task is not None:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
            self._receive_task = None

        if self._ws is not None:
            await self._ws.close()
            self._ws = None
            logger.info("Disconnected")

        self._explicitly_disconnecting = False
        self._on_disconnect()

    # -- outgoing (thread-safe) ---------------------------------------------

    def send_json(self, msg: dict[str, Any]) -> None:
        """Send a JSON message to the server (thread-safe).

        This may be called from any thread.  The actual send is posted to
        the transport's asyncio event loop.

        If the transport is disconnected or the event loop is closed, the
        message is silently dropped (with a debug log).

        Parameters
        ----------
        msg : dict
            JSON-serialisable message.
        """
        if self._loop is None or self._ws is None:
            logger.debug(
                "send_json: transport not connected, dropping %s", msg.get("type")
            )
            return
        try:
            asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(msg)), self._loop)
        except RuntimeError:
            # Event loop may have been closed between our check and the call
            logger.debug("send_json: event loop closed, dropping %s", msg.get("type"))

    def send_json_async(self, msg: dict[str, Any]) -> concurrent.futures.Future[None]:
        """Send a JSON message and return a Future for completion.

        Parameters
        ----------
        msg : dict
            JSON-serialisable message.

        Returns
        -------
        concurrent.futures.Future
            Resolves when the send completes.
        """
        if self._loop is None or self._ws is None:
            raise RuntimeError("Transport not connected")
        return asyncio.run_coroutine_threadsafe(
            self._ws.send(json.dumps(msg)), self._loop
        )

    # -- receive loop -------------------------------------------------------

    async def start_receive_loop(self) -> None:
        """Start the background receive loop as an asyncio task."""
        if self._receive_task is not None:
            return
        self._receive_task = asyncio.ensure_future(self._receive_loop())

    async def _receive_loop(self) -> None:
        """Read messages from the WebSocket and dispatch via callbacks.

        Runs until the connection closes or is cancelled.
        """
        assert self._ws is not None
        try:
            async for message in self._ws:
                if isinstance(message, bytes):
                    self._on_binary(message)
                else:
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON text message")
                        continue
                    self._on_json(msg)
        except websockets.ConnectionClosed as exc:
            logger.info("WebSocket closed: %s", exc)
        except asyncio.CancelledError:
            logger.debug("Receive loop cancelled")
            raise
        except Exception:
            logger.exception("Unexpected error in receive loop")
        finally:
            self._ws = None
            self._receive_task = None
            if not self._explicitly_disconnecting:
                if self._reconnect_max_attempts > 0:
                    self._reconnect_task = asyncio.ensure_future(
                        self._reconnect_loop()
                    )
                else:
                    self._on_disconnect()

    # -- reconnection -------------------------------------------------------

    async def _reconnect_loop(self) -> None:
        """Attempt to reconnect with exponential backoff.

        On success, restarts the receive loop and invokes
        ``on_reconnect(config)``.  On failure (all attempts exhausted
        or cancelled), invokes ``on_disconnect()``.
        """
        for attempt in range(1, self._reconnect_max_attempts + 1):
            self._reconnect_attempt = attempt

            if self._on_reconnecting is not None:
                self._on_reconnecting(attempt, self._reconnect_max_attempts)

            delay = min(
                self._reconnect_initial_delay
                * self._reconnect_backoff_base ** (attempt - 1),
                self._reconnect_max_delay,
            )
            # Add ±25 % jitter
            delay *= 1.0 + 0.25 * (2.0 * random.random() - 1.0)

            logger.info(
                "Reconnect attempt %d/%d in %.1f s",
                attempt,
                self._reconnect_max_attempts,
                delay,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.debug("Reconnect cancelled during backoff")
                return

            try:
                config = await self.connect()
            except asyncio.CancelledError:
                logger.debug("Reconnect cancelled during connect")
                return
            except Exception:
                logger.debug(
                    "Reconnect attempt %d/%d failed",
                    attempt,
                    self._reconnect_max_attempts,
                    exc_info=True,
                )
                continue

            # Success — restart the receive loop
            logger.info("Reconnected on attempt %d", attempt)
            self._reconnect_attempt = 0
            self._reconnect_task = None
            await self.start_receive_loop()

            if self._on_reconnect is not None:
                self._on_reconnect(config)
            return

        # All attempts exhausted
        logger.warning(
            "Reconnect failed after %d attempts", self._reconnect_max_attempts
        )
        self._reconnect_attempt = 0
        self._reconnect_task = None
        self._on_disconnect()

    # -- combined run (for use on a dedicated thread) -----------------------

    async def _run(self) -> None:
        """Connect, start receive loop, and wait until it finishes.

        Intended for ``asyncio.run(transport._run())`` on a background
        thread.  Returns when the connection closes or ``disconnect()``
        is called.  If reconnection is enabled, this will keep running
        through reconnect cycles until the transport is explicitly
        disconnected or reconnection fails permanently.
        """
        await self.connect()
        await self.start_receive_loop()

        # Loop through receive/reconnect cycles until done.
        while True:
            if self._receive_task is not None:
                try:
                    await self._receive_task
                except asyncio.CancelledError:
                    return

            # If no reconnect was started, we're done
            if self._reconnect_task is None:
                return

            # Wait for the reconnect loop to finish
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                return

            # After reconnect, a new _receive_task should exist.
            # If it doesn't, reconnection failed permanently — exit.
            if self._receive_task is None:
                return
