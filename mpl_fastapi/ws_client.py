"""WebSocket client for matplotlib-fastapi protocol.

This module provides a Python client for connecting to matplotlib-fastapi
WebSocket endpoints. It supports both real HTTP connections (via httpx) and
testing via FastAPI's TestClient.

Example usage with httpx:
    >>> import httpx
    >>> from mpl_fastapi.ws_client import MatplotlibWebSocketClient, create_httpx_adapter
    >>>
    >>> client = httpx.Client()
    >>> ws_client = MatplotlibWebSocketClient(
    ...     create_httpx_adapter(client),
    ...     base_url="http://localhost:8000/plots",
    ...     plot_name="sine",
    ...     init_params={"frequency": 2.0}
    ... )
    >>> with ws_client.connect():
    ...     ws_client.send_refresh()
    ...     image_data = ws_client.wait_for_image()

Example usage with TestClient:
    >>> from fastapi.testclient import TestClient
    >>> from mpl_fastapi.ws_client import MatplotlibWebSocketClient, create_fastapi_test_client_adapter
    >>>
    >>> test_client = TestClient(app)
    >>> ws_client = MatplotlibWebSocketClient(
    ...     create_fastapi_test_client_adapter(test_client),
    ...     base_url="/plots",
    ...     plot_name="sine",
    ... )
    >>> with ws_client.connect():
    ...     ws_client.send_refresh()
    ...     image_data = ws_client.wait_for_image()
"""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Protocol
from urllib.parse import urlencode

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class WebSocketAdapter(Protocol):
    """Protocol for WebSocket connection adapters.

    This allows the client to work with different WebSocket implementations
    (httpx, TestClient, etc.) through a common interface.
    """

    def connect(self, url: str) -> None:
        """Establish WebSocket connection."""
        ...

    def disconnect(self) -> None:
        """Close WebSocket connection."""
        ...

    def send_json(self, data: dict[str, Any]) -> None:
        """Send JSON message to server."""
        ...

    def send_bytes(self, data: bytes) -> None:
        """Send binary message to server."""
        ...

    def receive_json(self) -> dict[str, Any]:
        """Receive JSON message from server."""
        ...

    def receive_bytes(self) -> bytes:
        """Receive binary message from server."""
        ...

    def is_connected(self) -> bool:
        """Check if connection is active."""
        ...


class ContextManagerWebSocketAdapter:
    """Generic adapter for WebSocket connections using context managers.

    This adapter works with any client that provides a WebSocket connection
    through a context manager (TestClient, httpx, etc.).

    Parameters
    ----------
    client : Any
        Client instance with a method to create WebSocket connections
    connect_method : str, optional
        Name of the method to call on client (default: "websocket_connect")
    adapter_name : str, optional
        Name for logging purposes (default: "WebSocket")

    Examples
    --------
    For FastAPI TestClient:
        >>> adapter = ContextManagerWebSocketAdapter(test_client)

    For httpx:
        >>> adapter = ContextManagerWebSocketAdapter(
        ...     httpx_client,
        ...     connect_method="ws_connect",
        ...     adapter_name="httpx"
        ... )
    """

    def __init__(
        self,
        client: Any,
        connect_method: str = "websocket_connect",
        adapter_name: str = "WebSocket",
    ) -> None:
        """Initialize adapter."""
        self.client = client
        self.connect_method = connect_method
        self.adapter_name = adapter_name
        self._websocket: Any | None = None

    def connect(self, url: str) -> None:
        """Establish WebSocket connection."""
        # Get the connection method and call it
        connect_fn = getattr(self.client, self.connect_method)
        self._websocket = connect_fn(url)
        self._websocket.__enter__()
        logger.debug(f"{self.adapter_name} connected to {url}")

    def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._websocket is not None:
            self._websocket.__exit__(None, None, None)
            self._websocket = None
            logger.debug(f"{self.adapter_name} disconnected")

    def send_json(self, data: dict[str, Any]) -> None:
        """Send JSON message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        self._websocket.send_json(data)
        logger.debug(f"{self.adapter_name} sent JSON: {data}")

    def send_bytes(self, data: bytes) -> None:
        """Send binary message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        self._websocket.send_bytes(data)
        logger.debug(f"{self.adapter_name} sent {len(data)} bytes")

    def receive_json(self) -> dict[str, Any]:
        """Receive JSON message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        msg: dict[str, Any] = self._websocket.receive_json()
        logger.debug(f"{self.adapter_name} received JSON: {msg}")
        return msg

    def receive_bytes(self) -> bytes:
        """Receive binary message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        data: bytes = self._websocket.receive_bytes()
        logger.debug(f"{self.adapter_name} received {len(data)} bytes")
        return data

    def is_connected(self) -> bool:
        """Check if connection is active."""
        return self._websocket is not None


# Convenience factory functions for common adapters


def create_fastapi_test_client_adapter(
    test_client: Any,
) -> ContextManagerWebSocketAdapter:
    """Create adapter for FastAPI TestClient.

    Parameters
    ----------
    test_client : TestClient
        FastAPI test client instance

    Returns
    -------
    ContextManagerWebSocketAdapter
        Configured adapter for TestClient
    """
    return ContextManagerWebSocketAdapter(
        test_client,
        connect_method="websocket_connect",
        adapter_name="TestClient",
    )


def create_httpx_adapter(client: Any) -> ContextManagerWebSocketAdapter:
    """Create adapter for httpx WebSocket connections.

    Parameters
    ----------
    client : httpx.Client
        httpx client instance (must support WebSocket)

    Returns
    -------
    ContextManagerWebSocketAdapter
        Configured adapter for httpx
    """
    return ContextManagerWebSocketAdapter(
        client,
        connect_method="ws_connect",
        adapter_name="httpx",
    )


class MatplotlibWebSocketClient:
    """Client for matplotlib-fastapi WebSocket protocol.

    This client handles the full protocol including:
    - Protocol version negotiation
    - Initial server messages (config, formats, etc.)
    - Device pixel ratio setup
    - Interactive events (mouse, keyboard, toolbar)
    - Parameter updates
    - Drawing and image retrieval
    - Save/download operations

    Parameters
    ----------
    adapter : WebSocketAdapter
        WebSocket connection adapter (TestClientAdapter or HttpxAdapter)
    base_url : str
        Base URL for the plot server (e.g., "/plots" or "http://localhost:8000/plots")
    plot_name : str
        Name of the plot to connect to
    init_params : dict[str, Any], optional
        Initial parameters for plot generation
    device_pixel_ratio : float, optional
        Device pixel ratio for high-DPI displays (default: 1.0)
    supports_binary : bool, optional
        Whether client supports binary WebSocket messages (default: True)

    Attributes
    ----------
    connection_id : str | None
        Unique connection ID from server (for downloads)
    toolbar_items : list[dict[str, Any]]
        Toolbar button configuration from server
    save_formats : list[str]
        Available save formats from server
    default_save_format : str | None
        Default save format from server
    image_mode : str | None
        Current image mode (full or diff)
    """

    def __init__(
        self,
        adapter: WebSocketAdapter,
        base_url: str,
        plot_name: str,
        init_params: dict[str, Any] | None = None,
        device_pixel_ratio: float = 1.0,
        supports_binary: bool = True,
    ) -> None:
        """Initialize WebSocket client."""
        self.adapter = adapter
        self.base_url = base_url.rstrip("/")
        self.plot_name = plot_name
        self.init_params = init_params or {}
        self.device_pixel_ratio = device_pixel_ratio
        self.supports_binary = supports_binary

        # Server-provided configuration (populated on connection)
        self.connection_id: str | None = None
        self.toolbar_items: list[dict[str, Any]] = []
        self.save_formats: list[str] = []
        self.default_save_format: str | None = None
        self.image_mode: str | None = None

        # Image state for diff compositing
        self._current_image: Image.Image | None = None

        # Protocol state
        self._server_protocol_version: int | None = None
        self._initialized = False

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters."""
        url = f"{self.base_url}/ws/{self.plot_name}"
        if self.init_params:
            query = urlencode(self.init_params)
            url = f"{url}?{query}"
        return url

    @contextmanager
    def connect(self) -> Iterator[MatplotlibWebSocketClient]:
        """Context manager for WebSocket connection.

        Yields
        ------
        MatplotlibWebSocketClient
            This client instance after initialization

        Example
        -------
        >>> with client.connect():
        ...     client.send_refresh()
        ...     image = client.wait_for_image()
        """
        try:
            # Establish connection
            url = self._build_ws_url()
            self.adapter.connect(url)
            logger.info(f"Connected to {self.plot_name}")

            # Receive and process initial server messages
            self._receive_initial_messages()

            # Send client initialization messages
            self._send_client_init()

            # Mark as initialized
            self._initialized = True
            logger.info(f"Client initialized for {self.plot_name}")

            yield self

        finally:
            # Cleanup on exit
            self._initialized = False
            self._current_image = None
            self.adapter.disconnect()
            logger.info(f"Disconnected from {self.plot_name}")

    def _receive_initial_messages(self) -> None:
        """Receive initial message from server and complete handshake.

        Protocol flow:
        1. Server sends: protocol_version (first message)
        2. Client sends: protocol_version (MUST be first client message)
        3. Server validates and sends: image_mode, connection_id, toolbar_config,
           save_formats, default_save_format, history_buttons (6 messages)

        This method receives protocol_version, validates it, sends client protocol_version,
        then receives the remaining 6 configuration messages.
        """
        # 1. Receive server's protocol_version (first message)
        msg = self.adapter.receive_json()
        if msg["type"] != "protocol_version":
            raise RuntimeError(
                f"Expected protocol_version as first message, got {msg['type']}"
            )

        # Protocol version is REQUIRED
        if "version" not in msg:
            raise ValueError("Protocol version missing from server message")
        self._server_protocol_version = msg["version"]
        if self._server_protocol_version != 0:
            raise ValueError(
                f"Incompatible protocol version: {self._server_protocol_version}"
            )
        logger.debug(f"Server protocol version: {self._server_protocol_version}")

        # 2. Send client protocol_version (MUST be first client message)
        self.adapter.send_json({"type": "protocol_version", "version": 0})
        logger.debug("Sent client protocol version: 0")

        # 3. Receive 6 configuration messages from server
        expected_types = {
            "image_mode",
            "connection_id",
            "toolbar_config",
            "save_formats",
            "default_save_format",
            "history_buttons",
        }
        seen_types: set[str] = set()

        for _ in range(10):  # Safety limit (should only need 6)
            msg = self.adapter.receive_json()
            msg_type = msg["type"]
            seen_types.add(msg_type)

            # Process message through centralized handler
            self._process_message(msg)

            # Log history_buttons since it's not tracked in persistent state
            if msg_type == "history_buttons":
                logger.debug("Received initial history_buttons")

            # Check if we've received all expected messages
            if expected_types.issubset(seen_types):
                break

        # Verify we got all expected messages
        if not expected_types.issubset(seen_types):
            missing = expected_types - seen_types
            raise RuntimeError(
                f"Did not receive all initial messages. Missing: {missing}"
            )

    def _send_client_init(self) -> None:
        """Send client initialization messages (after protocol handshake).

        Client sends:
        1. supports_binary
        2. send_image_mode
        3. set_device_pixel_ratio (if not 1.0)
        4. refresh (triggers initial draw)

        Note: protocol_version is now sent in _receive_initial_messages as first message.
        """
        # 1. Send supports_binary
        self.adapter.send_json(
            {"type": "supports_binary", "value": self.supports_binary}
        )

        # 2. Request image mode
        self.adapter.send_json({"type": "send_image_mode"})
        mode_msg = self.adapter.receive_json()
        self._process_message(mode_msg)
        if mode_msg["type"] != "image_mode":
            logger.warning(f"Expected image_mode, got {mode_msg['type']}")

        # 3. Set device pixel ratio (if not default)
        if self.device_pixel_ratio != 1.0:
            self.adapter.send_json(
                {
                    "type": "set_device_pixel_ratio",
                    "device_pixel_ratio": self.device_pixel_ratio,
                }
            )
            # May receive draw message if DPI changed
            # Peek at next message to see if it's a draw
            next_msg = self.adapter.receive_json()
            self._process_message(next_msg)
            if next_msg["type"] == "draw":
                logger.debug("DPI change triggered draw request")
                # Draw message received, the refresh will also trigger a draw
            else:
                # Not a draw message, must be something else
                # This shouldn't happen in normal flow, but handle it
                logger.warning(
                    f"Unexpected message after set_device_pixel_ratio: {next_msg['type']}"
                )

        # 5. Send refresh to trigger initial draw
        self.adapter.send_json({"type": "refresh"})

        # Receive figure_label and draw messages
        label_msg = self.adapter.receive_json()
        self._process_message(label_msg)
        if label_msg["type"] == "figure_label":
            logger.debug(f"Figure label: {label_msg.get('label', '')}")

        draw_msg = self.adapter.receive_json()
        self._process_message(draw_msg)
        if draw_msg["type"] == "draw":
            logger.debug("Received initial draw request from refresh")

        # If DPI draw was received, we already got one draw message
        # The refresh also triggers a draw, but both point to the same image
        # We only need to request the draw once

        # Now send draw request to get the actual image and drain it
        # This completes the initialization handshake
        self.adapter.send_json({"type": "draw"})

        # Receive image_mode and image data
        mode_msg = self.adapter.receive_json()
        self._process_message(mode_msg)
        if mode_msg["type"] != "image_mode":
            logger.warning(f"Expected image_mode during init, got {mode_msg['type']}")

        # Drain the initial image data and process it
        initial_image = self.adapter.receive_bytes()
        logger.debug(f"Received initial image: {len(initial_image)} bytes")
        # Process through _process_image to initialize state
        self._process_image(initial_image, composite_diffs=True)
        logger.debug("Initialized image state")

    # Message sending methods

    def send_draw(self) -> None:
        """Request figure redraw from server."""
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "draw"})
        logger.debug("Sent draw request")

    def send_refresh(self) -> None:
        """Request figure refresh (full redraw)."""
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "refresh"})
        logger.debug("Sent refresh request")

    def send_resize(self, width: int, height: int) -> None:
        """Request figure resize.

        Parameters
        ----------
        width : int
            New width in pixels
        height : int
            New height in pixels
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "resize", "width": width, "height": height})
        logger.debug(f"Sent resize request: {width}x{height}")

    def send_toolbar_button(self, button_name: str) -> None:
        """Trigger toolbar button action.

        Parameters
        ----------
        button_name : str
            Button name (home, back, forward, pan, zoom)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "toolbar_button", "name": button_name})
        logger.debug(f"Sent toolbar button: {button_name}")

    def send_update_params(self, params: dict[str, Any]) -> None:
        """Update plot parameters.

        Parameters
        ----------
        params : dict[str, Any]
            Update parameters matching the plot's update schema
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "update_params", "params": params})
        logger.debug(f"Sent update params: {params}")

    def send_mouse_event(
        self,
        event_type: str,
        x: float,
        y: float,
        button: int | None = None,
        step: int | None = None,
    ) -> None:
        """Send mouse event to server.

        Parameters
        ----------
        event_type : str
            Event type (button_press, button_release, motion_notify, scroll, etc.)
        x : float
            X coordinate
        y : float
            Y coordinate
        button : int, optional
            Mouse button (1=left, 2=middle, 3=right)
        step : int, optional
            Scroll step (for scroll events)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        msg: dict[str, Any] = {"type": event_type, "x": x, "y": y}
        if button is not None:
            msg["button"] = button
        if step is not None:
            msg["step"] = step

        self.adapter.send_json(msg)
        # Don't log motion events to reduce noise
        if event_type not in ("motion_notify", "figure_enter", "figure_leave"):
            logger.debug(f"Sent mouse event: {event_type}")

    def send_keyboard_event(self, event_type: str, key: str) -> None:
        """Send keyboard event to server.

        Parameters
        ----------
        event_type : str
            Event type (key_press, key_release)
        key : str
            Key name
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": event_type, "key": key})
        logger.debug(f"Sent keyboard event: {event_type} - {key}")

    def send_save_figure(
        self,
        format: str = "png",  # noqa: A002
        dpi: int = 100,
        transparent: bool = False,
    ) -> dict[str, Any]:
        """Request figure save and return save info.

        Parameters
        ----------
        format : str, optional
            Save format (png, pdf, svg, etc.)
        dpi : int, optional
            DPI for raster formats
        transparent : bool, optional
            Transparency for PNG format

        Returns
        -------
        dict[str, Any]
            Save completion info with file_id and download_url
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        self.adapter.send_json(
            {
                "type": "save_figure",
                "format": format,
                "dpi": dpi,
                "transparent": transparent,
            }
        )
        logger.debug(f"Sent save request: {format} @ {dpi} DPI")

        # Wait for save_complete response
        response = self.adapter.receive_json()
        if response["type"] == "save_complete":
            logger.info(f"Save complete: {response['filename']}")
            return response
        if response["type"] == "save_error":
            raise RuntimeError(f"Save failed: {response['message']}")
        raise RuntimeError(f"Unexpected response to save_figure: {response['type']}")

    # Message receiving methods

    def _process_message(self, msg: dict[str, Any]) -> None:
        """Process a JSON message and update internal state.

        This method should be called for every JSON message received to ensure
        state is kept consistent. It updates image_mode and other client state.

        Parameters
        ----------
        msg : dict[str, Any]
            JSON message from server
        """
        msg_type = msg["type"]

        # Update state based on message type
        if msg_type == "image_mode":
            self.image_mode = msg["mode"]
            logger.debug(f"Updated image mode: {self.image_mode}")
        elif msg_type == "connection_id":
            self.connection_id = msg["id"]
            logger.debug(f"Updated connection ID: {self.connection_id}")
        elif msg_type == "toolbar_config":
            self.toolbar_items = msg["items"]
            logger.debug(f"Updated toolbar items: {len(self.toolbar_items)}")
        elif msg_type == "save_formats":
            self.save_formats = msg["formats"]
            logger.debug(f"Updated save formats: {self.save_formats}")
        elif msg_type == "default_save_format":
            self.default_save_format = msg["format"]
            logger.debug(f"Updated default save format: {self.default_save_format}")
        # Other message types don't update persistent state

    def _process_image(self, image_data: bytes, *, composite_diffs: bool = True) -> bytes:
        """Process image data and update internal state.

        Handles diff compositing if enabled. Should be called for every image
        received to maintain correct state.

        Parameters
        ----------
        image_data : bytes
            PNG image data from server
        composite_diffs : bool, optional
            If True, composite diff images on top of the last full image

        Returns
        -------
        bytes
            PNG image data (composited if diff mode and composite_diffs=True)
        """
        if not composite_diffs or self.image_mode == "full":
            # Store full image for future diff compositing
            self._current_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            logger.debug(f"Stored full image: {self._current_image.size}")
            return image_data
        elif self.image_mode == "diff":
            if self._current_image is None:
                # No base image, treat diff as full
                logger.warning("Received diff image without base image, treating as full")
                self._current_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
                return image_data

            # Composite diff on top of current image
            diff_image = Image.open(io.BytesIO(image_data)).convert("RGBA")

            # Ensure same size
            if diff_image.size != self._current_image.size:
                raise RuntimeError(
                    f"Diff image size {diff_image.size} doesn't match "
                    f"base image size {self._current_image.size}"
                )

            # Composite diff onto current image (diff has transparency for unchanged pixels)
            self._current_image.paste(diff_image, (0, 0), diff_image)

            # Convert back to PNG bytes
            output = io.BytesIO()
            self._current_image.save(output, format="PNG")
            composited_data = output.getvalue()
            logger.debug(f"Composited diff: {len(composited_data)} bytes")
            return composited_data
        else:
            raise RuntimeError(f"Unknown image mode: {self.image_mode}")

    def receive_message(self) -> dict[str, Any] | bytes:
        """Receive next message from server and update state.

        This method processes JSON messages through _process_message() to
        ensure state is always updated correctly.

        Returns
        -------
        dict[str, Any] | bytes
            JSON message or binary image data
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        # Try to receive JSON first
        try:
            msg = self.adapter.receive_json()
            self._process_message(msg)
            return msg
        except Exception:
            # If that fails, try binary
            return self.adapter.receive_bytes()

    def wait_for_image(self, *, composite_diffs: bool = True) -> bytes:
        """Wait for and return image data from server.

        This method expects the server to send:
        1. image_mode message ("full" or "diff")
        2. binary PNG data

        When `composite_diffs=True` (default), diff images are composited
        on top of the previous full image to produce a complete image.
        This matches the browser behavior.

        Parameters
        ----------
        composite_diffs : bool, optional
            If True (default), composite diff images on top of the last full image.
            If False, return the raw image bytes from server.

        Returns
        -------
        bytes
            PNG image data (composited if diff mode and composite_diffs=True)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        # Receive image_mode message - this updates self.image_mode via _process_message
        mode_msg = self.adapter.receive_json()
        self._process_message(mode_msg)

        if mode_msg["type"] != "image_mode":
            raise RuntimeError(f"Expected image_mode, got {mode_msg['type']}")

        # Receive binary image data
        image_data = self.adapter.receive_bytes()
        logger.debug(f"Received image: {len(image_data)} bytes ({self.image_mode} mode)")

        # Process image (handles compositing)
        return self._process_image(image_data, composite_diffs=composite_diffs)

    def drain_messages(self, max_messages: int = 20) -> list[dict[str, Any]]:
        """Drain all queued JSON messages from server.

        This is useful after toolbar actions or other events that may
        queue multiple messages.

        All messages are processed through _process_message() to ensure
        state is kept up to date.

        **Warning:** With TestClient adapter, this may block indefinitely if
        there are no messages to receive. Use with caution in tests, or use
        `wait_for_message_type()` instead to receive specific messages.

        Parameters
        ----------
        max_messages : int, optional
            Maximum messages to drain (safety limit)

        Returns
        -------
        list[dict[str, Any]]
            List of received messages
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        messages = []
        for _ in range(max_messages):
            try:
                msg = self.adapter.receive_json()
                self._process_message(msg)
                messages.append(msg)
                logger.debug(f"Drained message: {msg['type']}")
            except Exception:
                # No more messages available
                break
        return messages

    def wait_for_message_type(
        self,
        target_type: str,
        max_messages: int = 20,
        skip_types: set[str] | None = None,
    ) -> dict[str, Any]:
        """Wait for specific message type, processing all messages.

        Unlike the old implementation that dropped messages, this processes
        all messages through _process_message() to keep state updated.

        Parameters
        ----------
        target_type : str
            Message type to wait for
        max_messages : int, optional
            Maximum messages to check (safety limit)
        skip_types : set[str], optional
            Message types to not return (but still process for state updates)

        Returns
        -------
        dict[str, Any]
            The target message

        Raises
        ------
        RuntimeError
            If target message not found within max_messages
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        skip_types = skip_types or set()

        for _ in range(max_messages):
            msg = self.adapter.receive_json()
            self._process_message(msg)

            if msg["type"] in skip_types:
                logger.debug(f"Skipping message (but state updated): {msg['type']}")
                continue
            if msg["type"] == target_type:
                return msg

        raise RuntimeError(
            f"Did not receive {target_type} message within {max_messages} messages"
        )
