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
    ...     image_data = ws_client.send_refresh()  # Get first image

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
    ...     image_data = ws_client.send_refresh()  # Get first image
"""

from __future__ import annotations

import io
import logging
import struct
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Protocol
from urllib.parse import urlencode

from PIL import Image

logger = logging.getLogger(__name__)

# Protocol constants
PROTOCOL_VERSION = 0
BINARY_HEADER_SIZE = 8


class ImageTypeMode(IntEnum):
    """Binary image type/mode - first byte of header."""

    FULL = 0x00
    DIFF = 0x01


class ImageFormat(IntEnum):
    """Binary image format - second byte of header."""

    PNG = 0x01
    JPEG = 0x02
    WEBP = 0x03


@dataclass
class BinaryImageHeader:
    """Parsed binary image header (8 bytes total).

    Structure: [type_mode: 1, format: 1, seq_num: 2, base_seq: 2, flags: 2]
    """

    type_mode: ImageTypeMode
    format: ImageFormat
    seq_num: int
    base_seq: int
    flags: int


def parse_binary_image(data: bytes) -> tuple[BinaryImageHeader, bytes]:
    """Parse binary image with 8-byte header.

    Parameters
    ----------
    data : bytes
        Binary data with header + image

    Returns
    -------
    tuple[BinaryImageHeader, bytes]
        Parsed header and image data
    """
    if len(data) < BINARY_HEADER_SIZE:
        raise ValueError(f"Binary data too short: {len(data)} < {BINARY_HEADER_SIZE}")

    type_mode = ImageTypeMode(data[0])
    img_format = ImageFormat(data[1])
    seq_num, base_seq, flags = struct.unpack(">HHH", data[2:8])

    header = BinaryImageHeader(
        type_mode=type_mode,
        format=img_format,
        seq_num=seq_num,
        base_seq=base_seq,
        flags=flags,
    )
    image_data = data[BINARY_HEADER_SIZE:]
    return header, image_data


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
        logger.debug("%s connected to %s", self.adapter_name, url)

    def disconnect(self) -> None:
        """Close WebSocket connection."""
        if self._websocket is not None:
            self._websocket.__exit__(None, None, None)
            self._websocket = None
            logger.debug("%s disconnected", self.adapter_name)

    def send_json(self, data: dict[str, Any]) -> None:
        """Send JSON message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        self._websocket.send_json(data)
        logger.debug("%s sent JSON: %s", self.adapter_name, data)

    def send_bytes(self, data: bytes) -> None:
        """Send binary message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        self._websocket.send_bytes(data)
        logger.debug("%s sent %d bytes", self.adapter_name, len(data))

    def receive_json(self) -> dict[str, Any]:
        """Receive JSON message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        msg: dict[str, Any] = self._websocket.receive_json()
        logger.debug("%s received JSON: %s", self.adapter_name, msg)
        return msg

    def receive_bytes(self) -> bytes:
        """Receive binary message."""
        if self._websocket is None:
            raise RuntimeError("Not connected")
        data: bytes = self._websocket.receive_bytes()
        logger.debug("%s received %d bytes", self.adapter_name, len(data))
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
    """Client for matplotlib-fastapi WebSocket protocol (v0).

    This client handles the full v0 protocol including:
    - Consolidated init/config handshake
    - Binary images with 8-byte headers
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
    figure_size : tuple[int, int] | None
        Current figure size in CSS pixels
    figure_dpi : float | None
        Figure DPI
    figure_label : str | None
        Figure label/title
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
        self.image_format: str | None = None
        self.image_mode: str | None = None
        self.figure_size: tuple[int, int] | None = None
        self.figure_dpi: float | None = None
        self.figure_label: str | None = None
        self.update_schema: dict[str, Any] | None = None

        # Image state for diff compositing
        self._current_image: Image.Image | None = None
        self._last_seq_num: int = 0

        # Protocol state
        self._server_protocol_version: int | None = None
        self._initialized = False

    def _build_ws_url(self) -> str:
        """Build WebSocket URL with query parameters (v0 endpoint)."""
        url = f"{self.base_url}/ws/v0/{self.plot_name}"
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
        ...     image = client.send_refresh()
        """
        try:
            # Establish connection
            url = self._build_ws_url()
            self.adapter.connect(url)
            logger.info("Connected to %s", self.plot_name)

            # Send init message and receive config (v0 protocol)
            self._perform_handshake()

            logger.info("Client initialized for %s", self.plot_name)

            yield self

        finally:
            # Cleanup on exit
            self._initialized = False
            self._current_image = None
            self._last_seq_num = 0
            self.adapter.disconnect()
            logger.info("Disconnected from %s", self.plot_name)

    def _perform_handshake(self) -> None:
        """Perform v0 protocol handshake.

        Protocol v0 flow:
        1. Client sends: init message (protocol_version, device_pixel_ratio, supports_binary)
        2. Server sends: config message (consolidated configuration)
        3. Client sends: refresh to get first image
        """
        # 1. Send init message (MUST be first client message)
        init_msg = {
            "type": "init",
            "protocol_version": PROTOCOL_VERSION,
            "device_pixel_ratio": self.device_pixel_ratio,
            "supports_binary": self.supports_binary,
        }
        self.adapter.send_json(init_msg)
        logger.debug("Sent init message: protocol_version=%s", PROTOCOL_VERSION)

        # 2. Receive config message
        config_msg = self.adapter.receive_json()
        if config_msg["type"] == "error":
            raise RuntimeError(f"Server error: {config_msg.get('message', 'Unknown')}")
        if config_msg["type"] != "config":
            raise RuntimeError(f"Expected config message, got {config_msg['type']}")

        # Validate protocol version
        server_version = config_msg.get("protocol_version")
        if server_version != PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol version mismatch: client={PROTOCOL_VERSION}, server={server_version}"
            )
        self._server_protocol_version = server_version
        logger.debug("Server protocol version validated: %s", server_version)

        # Extract configuration from consolidated config message
        self.connection_id = config_msg["connection_id"]

        # Figure config
        figure_config = config_msg.get("figure", {})
        size = figure_config.get("size", [640, 480])
        self.figure_size = (size[0], size[1])
        self.figure_dpi = figure_config.get("dpi", 100)
        self.figure_label = figure_config.get("label", "")

        # Toolbar config
        toolbar_config = config_msg.get("toolbar", {})
        self.toolbar_items = toolbar_config.get("items", [])

        # Save config
        save_config = config_msg.get("save", {})
        self.save_formats = save_config.get("formats", ["png"])
        self.default_save_format = save_config.get("default_format", "png")

        # Image config
        image_config = config_msg.get("image", {})
        self.image_format = image_config["format"]
        self.image_mode = "full"  # Start with full mode

        # Update schema
        self.update_schema = config_msg.get("update_schema")

        logger.debug(
            "Config received: connection_id=%s, figure_size=%s, toolbar_items=%d",
            self.connection_id, self.figure_size, len(self.toolbar_items),
        )

        # Mark as initialized
        self._initialized = True

    # Core message receive/send methods

    def _receive_json(self) -> dict[str, Any]:
        """Receive JSON message and update state.

        Returns
        -------
        dict[str, Any]
            JSON message with state automatically updated
        """
        msg = self.adapter.receive_json()
        self._process_message(msg)  # Always update state
        return msg

    def _receive_bytes(self) -> bytes:
        """Receive binary message.

        Returns
        -------
        bytes
            Binary message data
        """
        return self.adapter.receive_bytes()

    # Message sending methods

    def _receive_render_response(self) -> bytes:
        """Receive binary image with 8-byte header (v0 protocol).

        Returns
        -------
        bytes
            Image data (composited if diff mode)
        """
        # Receive binary data with header
        raw_data = self._receive_bytes()
        header, image_data = parse_binary_image(raw_data)

        # Update image mode based on header
        self.image_mode = "full" if header.type_mode == ImageTypeMode.FULL else "diff"

        # Track sequence numbers
        self._last_seq_num = header.seq_num

        logger.debug(
            "Received image: %d bytes (mode=%s, seq=%s, base=%s)",
            len(image_data), self.image_mode, header.seq_num, header.base_seq,
        )

        return self._process_image(image_data, composite_diffs=True)

    def send_render(self) -> bytes:
        """Request figure render and return image data.

        Automatically waits for binary image with header.

        Returns
        -------
        bytes
            Image data (composited if diff mode)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "render"})
        logger.debug("Sent render request")

        return self._receive_render_response()

    def send_refresh(self) -> bytes:
        """Request figure refresh with full render.

        Forces a full (non-diff) render and returns the image data.

        Returns
        -------
        bytes
            Full image data
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "refresh"})
        logger.debug("Sent refresh request")

        # Receive binary image with header
        return self._receive_render_response()

    def send_resize(self, width: int, height: int) -> dict[str, Any]:
        """Request figure resize and wait for acknowledgment.

        Parameters
        ----------
        width : int
            New width in pixels
        height : int
            New height in pixels

        Returns
        -------
        dict[str, Any]
            Resize response with actual size and forward flag
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "resize", "width": width, "height": height})
        logger.debug("Sent resize request: %dx%d", width, height)

        # Wait for resize acknowledgment
        resize_msg = self._receive_json()
        if resize_msg["type"] != "resize":
            raise RuntimeError(f"Expected resize response, got {resize_msg['type']}")
        return resize_msg

    def send_toolbar_button(self, button_name: str) -> None:
        """Trigger toolbar button action.

        Toolbar actions may queue multiple messages (navigate_mode, message,
        history_buttons, invalidate, etc.). These are sent after the command completes.
        Use receive_message() to get messages as needed.

        Parameters
        ----------
        button_name : str
            Button name (home, back, forward, pan, zoom)
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "toolbar_button", "name": button_name})
        logger.debug("Sent toolbar button: %s", button_name)

    def send_update_params(self, params: dict[str, Any]) -> None:
        """Update plot parameters and wait for invalidate message.

        Parameters
        ----------
        params : dict[str, Any]
            Update parameters matching the plot's update schema
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")
        self.adapter.send_json({"type": "update_params", "params": params})
        logger.debug("Sent update params: %s", params)

        # Wait for invalidate message (update always triggers redraw)
        invalidate_msg = self._receive_json()
        if invalidate_msg["type"] == "error":
            raise RuntimeError(
                f"Update failed: {invalidate_msg.get('message', 'Unknown error')}"
            )
        if invalidate_msg["type"] != "invalidate":
            logger.warning(
                "Expected invalidate after update_params, got %s", invalidate_msg['type']
            )

    def send_mouse_event(
        self,
        event_type: str,
        x: float,
        y: float,
        button: int | None = None,
        step: int | None = None,
    ) -> None:
        """Send mouse event to server.

        Mouse events may trigger callbacks that queue messages. Use
        receive_message() to get messages as needed.

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
            logger.debug("Sent mouse event: %s", event_type)

    def send_keyboard_event(self, event_type: str, key: str) -> None:
        """Send keyboard event to server.

        Keyboard events may trigger callbacks that queue messages. Use
        receive_message() to get messages as needed.

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
        logger.debug("Sent keyboard event: %s - %s", event_type, key)

    def send_save_figure(
        self,
        format: str = "png",
        dpi: int = 100,
        transparent: bool = False,
    ) -> dict[str, Any]:
        """Request figure save and return save info.

        Automatically waits for save_complete or save_error response.

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

        Raises
        ------
        RuntimeError
            If save fails or unexpected response received
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
        logger.debug("Sent save request: %s @ %d DPI", format, dpi)

        # Wait for save_complete or save_error response
        response = self._receive_json()
        if response["type"] == "save_complete":
            logger.info("Save complete: %s", response['filename'])
            return response
        if response["type"] == "error":
            raise RuntimeError(
                f"Save failed: {response.get('message', 'Unknown error')}"
            )
        raise RuntimeError(f"Unexpected response to save_figure: {response['type']}")

    # Image processing

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
            logger.debug("Updated image mode: %s", self.image_mode)
        elif msg_type == "connection_id":
            self.connection_id = msg["id"]
            logger.debug("Updated connection ID: %s", self.connection_id)
        elif msg_type == "toolbar_config":
            self.toolbar_items = msg["items"]
            logger.debug("Updated toolbar items: %d", len(self.toolbar_items))
        elif msg_type == "save_formats":
            self.save_formats = msg["formats"]
            logger.debug("Updated save formats: %s", self.save_formats)
        elif msg_type == "default_save_format":
            self.default_save_format = msg["format"]
            logger.debug("Updated default save format: %s", self.default_save_format)
        elif msg_type == "figure_size":
            # Store figure size information (not strictly necessary for client state,
            # but useful for debugging and validation)
            figure_size = msg.get("size")
            figure_dpi = msg.get("dpi")
            logger.debug("Received figure size: %s @ %s DPI", figure_size, figure_dpi)
        # Other message types don't update persistent state

    def _process_image(
        self, image_data: bytes, *, composite_diffs: bool = True
    ) -> bytes:
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
            logger.debug("Stored full image: %s", self._current_image.size)
            return image_data
        if self.image_mode == "diff":
            if self._current_image is None:
                # No base image, treat diff as full
                logger.warning(
                    "Received diff image without base image, treating as full"
                )
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
            logger.debug("Composited diff: %d bytes", len(composited_data))
            return composited_data
        raise RuntimeError(f"Unknown image mode: {self.image_mode}")

    # Utility methods for advanced use cases

    def receive_message(self, timeout: float | None = None) -> dict[str, Any]:
        """Receive the next message from the WebSocket.

        This is a blocking call that waits for and returns the next JSON message
        from the server. The message is automatically processed through
        _process_message() to keep state updated.

        Parameters
        ----------
        timeout : float | None, optional
            Timeout in seconds. If None, blocks indefinitely.
            Note: timeout support depends on the underlying WebSocket adapter.

        Returns
        -------
        dict[str, Any]
            The next JSON message from the server

        Raises
        ------
        RuntimeError
            If client is not initialized
        TimeoutError
            If timeout is reached (adapter-dependent)

        Example
        -------
        >>> with client.connect():
        ...     client.send_toolbar_button("pan")
        ...     msg = client.receive_message()
        ...     while msg["type"] != "draw":
        ...         msg = client.receive_message()
        """
        if not self._initialized:
            raise RuntimeError("Client not initialized. Use connect() context manager.")

        # Note: timeout parameter is accepted but may not be supported by all adapters
        # The underlying receive_json() implementation is synchronous/blocking
        if timeout is not None:
            logger.warning(
                "Timeout parameter provided but may not be supported by all WebSocket adapters"
            )

        return self._receive_json()
