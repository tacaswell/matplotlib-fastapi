"""Toolkit-agnostic remote Matplotlib canvas and toolbar.

This module provides :class:`FigureCanvasRemote` and
:class:`RemoteNavigationToolbar2`, which handle all protocol logic
(image compositing, event forwarding, server-message dispatch) without
depending on any GUI toolkit.  A toolkit layer (e.g.
:mod:`~mpl_fastapi.remote.backend_qtremote`) subclasses these to
provide actual widget rendering and event loop integration.
"""

from __future__ import annotations

import io
import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, ClassVar, cast

import matplotlib as mpl
from matplotlib import cbook
from matplotlib.backend_bases import Event as _MplEvent
from matplotlib.backend_bases import (
    FigureCanvasBase,
    FigureManagerBase,
    KeyEvent,
    MouseEvent,
    NavigationToolbar2,
)
from matplotlib.figure import Figure
from PIL import Image

from mpl_fastapi.remote.transport import RemoteTransport, ServerConfig
from mpl_fastapi.ws_client import ImageTypeMode, parse_binary_image

__all__ = [
    "FigureCanvasRemote",
    "FigureManagerRemote",
    "RemoteNavigationToolbar2",
    "RemotePlotInfo",
    "list_remote_figures",
]

logger = logging.getLogger(__name__)


class FigureCanvasRemote(FigureCanvasBase):
    """Canvas that delegates all rendering to a remote mpl_fastapi server.

    This canvas does **not** inherit from ``FigureCanvasAgg`` — there is no
    local renderer.  All images come from the server via the WebSocket
    protocol.

    Threading contract
    ------------------
    * ``draw()`` and ``schedule_repaint()`` run on the **main** (GUI) thread
      and only blit the last received image.
    * Network IO runs on the transport's background thread/task.
    * ``_on_binary_message`` / ``_on_json_message`` are expected to be called
      on the main thread (the toolkit layer must marshal via signals or
      similar).

    Subclass responsibilities (toolkit layer)
    ------------------------------------------
    * :meth:`schedule_repaint` — post a repaint/update to the toolkit event
      loop so that ``draw()`` / ``paintEvent()`` will be called.
    """

    _remote_image: Image.Image | None
    _transport: RemoteTransport
    _server_config: ServerConfig
    toolbar: RemoteNavigationToolbar2 | None  # type: ignore[assignment]

    # Rubberband state (set by server "rubberband" messages)
    _rubberband_rect: tuple[float, float, float, float] | None

    # Motion-notify throttle state
    _mouse_move_interval: float = 1.0 / 30  # 30 Hz max
    _last_mouse_move_time: float = 0.0
    _pending_mouse_move: tuple[float, float] | None = None
    _mouse_move_timer: threading.Timer | None = None

    def __init__(
        self,
        figure: Figure,
        transport: RemoteTransport,
        server_config: ServerConfig,
    ) -> None:
        self._is_drawing = False
        super().__init__(figure)
        self._transport = transport
        self._server_config = server_config
        self._remote_image = None
        self._rubberband_rect = None
        self._last_seq_num = 0

        # The server config carries *scaled* DPI (original_dpi * dpr) and
        # CSS-pixel dimensions.  Recover original_dpi so that
        # Figure._original_dpi is correct — this lets subsequent calls to
        # _set_device_pixel_ratio() (e.g. from Wayland DPR updates) work.
        w_css, h_css = server_config.figure_size
        dpr = transport._device_pixel_ratio
        original_dpi = server_config.figure_dpi / dpr

        figure.set_dpi(original_dpi)
        figure.set_size_inches(
            w_css / original_dpi, h_css / original_dpi, forward=False
        )
        self._set_device_pixel_ratio(dpr)  # type: ignore[attr-defined]
        if server_config.figure_label:
            figure.set_label(server_config.figure_label)

        # Subscribe to mpl events so that *any* toolkit backend that fires
        # them (Qt, GTK, Tk, …) automatically gets wire-protocol forwarding
        # without needing per-toolkit overrides.
        _mouse = cast("Callable[[_MplEvent], Any]", self._on_mpl_mouse_event)
        self.mpl_connect("button_press_event", _mouse)
        self.mpl_connect("button_release_event", _mouse)
        self.mpl_connect("motion_notify_event", _mouse)
        self.mpl_connect("scroll_event", _mouse)
        _key = cast("Callable[[_MplEvent], Any]", self._on_mpl_key_event)
        self.mpl_connect("key_press_event", _key)
        self.mpl_connect("key_release_event", _key)

    # -- properties ---------------------------------------------------------

    @property
    def transport(self) -> RemoteTransport:
        """The WebSocket transport for this canvas."""
        return self._transport

    @property
    def remote_image(self) -> Image.Image | None:
        """The most recent composited image from the server, or ``None``."""
        return self._remote_image

    @property
    def has_update_params(self) -> bool:
        """Whether the server supports update parameters for this plot."""
        return self._server_config.update_schema is not None

    # -- rendering (main thread, blit only) ---------------------------------


    def draw(self) -> None:
        """Request a server re-render and schedule a Qt repaint."""
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            self._transport.send_json({"type": "render"})
        self.schedule_repaint()

    def draw_idle(self) -> None:
        """Schedule a ``draw()`` on the toolkit event loop.

        Subclasses should override this to e.g.
        ``QTimer.singleShot(0, self._draw_idle)``.
        """
        # Default: call draw immediately (sufficient for non-GUI testing)
        self.draw()

    def get_width_height(self, *, physical: bool = False) -> tuple[int, int]:
        # docstring inherited
        w, h = self._server_config.figure_size
        if physical:
            dpr = self.device_pixel_ratio
            return int(w * dpr), int(h * dpr)
        return w, h

    # -- reconnection (called on main thread) -------------------------------

    def _on_reconnected(self, config: ServerConfig) -> None:
        """Reset canvas state after a successful reconnection.

        Called by the transport (via the toolkit layer's signal/callback
        mechanism) when a WebSocket reconnection succeeds.  Resets the
        composited image and server config, then tells the new
        server-side figure to match the client's current geometry.

        The server always creates a fresh default-sized figure on
        reconnect.  Rather than adopting that default, we keep the
        client's existing figure geometry and send a ``resize`` so the
        server figure matches the client window.  This avoids the
        window shrinking back to the default on every reconnect.

        Parameters
        ----------
        config : ServerConfig
            The new server config from the reconnection handshake.
        """
        # Preserve the client's current CSS-pixel size *before*
        # overwriting _server_config (which carries figure_size).
        old_w, old_h = self._server_config.figure_size

        self._server_config = config
        self._remote_image = None
        self._rubberband_rect = None
        self._last_seq_num = 0

        if config.figure_label:
            self.figure.set_label(config.figure_label)

        # Tell the new server-side figure to use the client's current
        # size, not the default it was created with.  The resize message
        # is in CSS pixels; the server multiplies by DPR internally.
        self._transport.send_json(
            {"type": "resize", "width": old_w, "height": old_h}
        )
        # Update _server_config to reflect the size we just requested
        self._server_config = ServerConfig(
            **{**self._server_config.__dict__, "figure_size": (old_w, old_h)}
        )

        # Request a full render at the correct size
        self._transport.send_json({"type": "refresh"})
        self.schedule_repaint()

    # -- incoming message handlers (called on main thread) ------------------

    def _on_binary_message(self, data: bytes) -> None:
        """Process an incoming binary image message.

        Parses the 8-byte header, composites diff images, stores the
        result in ``_remote_image``, and calls ``schedule_repaint()``.

        Parameters
        ----------
        data : bytes
            Raw binary message (header + image data).
        """
        header, image_data = parse_binary_image(data)
        self._last_seq_num = header.seq_num

        if header.type_mode == ImageTypeMode.FULL or self._remote_image is None:
            self._remote_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
            logger.debug(
                "Full image received: seq=%d, size=%s",
                header.seq_num,
                self._remote_image.size,
            )
        elif header.type_mode == ImageTypeMode.DIFF:
            diff = Image.open(io.BytesIO(image_data)).convert("RGBA")
            if diff.size != self._remote_image.size:
                logger.warning(
                    "Diff size %s != base size %s — treating as full",
                    diff.size,
                    self._remote_image.size,
                )
                self._remote_image = diff
            else:
                self._remote_image.paste(diff, (0, 0), diff)
                logger.debug(
                    "Diff composited: seq=%d, base_seq=%d",
                    header.seq_num,
                    header.base_seq,
                )
        else:
            logger.warning("Unknown image type_mode: %s", header.type_mode)
            self._remote_image = Image.open(io.BytesIO(image_data)).convert("RGBA")

        self.schedule_repaint()

    def _on_json_message(self, msg: dict[str, Any]) -> None:
        """Dispatch an incoming JSON message from the server.

        Parameters
        ----------
        msg : dict
            Parsed JSON message.
        """
        msg_type = msg.get("type")
        logger.debug("JSON message: type=%s", msg_type)

        if msg_type == "invalidate":
            # Server says figure changed — request a new render
            self._transport.send_json({"type": "render"})

        elif msg_type == "resize":
            size = msg.get("size", [])
            if len(size) == 2:
                w_css, h_css = int(size[0]), int(size[1])
                self._server_config = ServerConfig(
                    **{
                        **self._server_config.__dict__,
                        "figure_size": (w_css, h_css),
                    }
                )
                # Recover original DPI and set size in inches.
                original_dpi = self._server_config.figure_dpi / self.device_pixel_ratio
                self.figure.set_size_inches(
                    w_css / original_dpi, h_css / original_dpi, forward=False
                )
                # Request a re-render at the new size (mirrors JS client's
                # handle_resize → send_render_request flow).
                self._transport.send_json({"type": "render"})
            self.schedule_repaint()

        elif msg_type == "navigate_mode":
            mode = msg.get("mode", "")
            if self.toolbar is not None:
                self.toolbar._on_navigate_mode(mode)

        elif msg_type == "message":
            text = msg.get("message", "")
            if self.toolbar is not None:
                self.toolbar.set_message(text)

        elif msg_type == "rubberband":
            x0 = msg.get("x0", -1)
            y0 = msg.get("y0", -1)
            x1 = msg.get("x1", -1)
            y1 = msg.get("y1", -1)
            if x0 < 0 and y0 < 0:
                self._rubberband_rect = None
                if self.toolbar is not None:
                    self.toolbar.remove_rubberband()
                else:
                    self.schedule_repaint()
            else:
                self._rubberband_rect = (x0, y0, x1, y1)
                if self.toolbar is not None:
                    self.toolbar.draw_rubberband(None, x0, y0, x1, y1)
                else:
                    self.schedule_repaint()

        elif msg_type == "history_buttons":
            if self.toolbar is not None:
                self.toolbar._on_history_buttons(
                    back=msg.get("Back", False),
                    forward=msg.get("Forward", False),
                )

        elif msg_type == "error":
            logger.error("Server error: %s", msg.get("message", "unknown"))

        elif msg_type == "image_mode":
            # Informational — we handle mode per-image via the binary header
            pass

        elif msg_type == "save_complete":
            pending = self._pending_print_figure_callback
            if pending is not None:
                self._pending_print_figure_callback = None
                filepath, callback = pending
                callback(msg)
                self._handle_print_figure_complete(filepath, msg)
            elif self.toolbar is not None:
                self.toolbar._on_save_complete(msg)

        elif msg_type == "save_error":
            logger.error("Save error: %s", msg.get("message", "unknown"))
            pending = self._pending_print_figure_callback
            if pending is not None:
                self._pending_print_figure_callback = None
                _, callback = pending
                callback(msg)
            elif self.toolbar is not None:
                self.toolbar._on_save_error(msg)

        else:
            logger.debug("Unhandled server message type: %s", msg_type)

    # -- mpl event callbacks (subscribed to via mpl_connect) -----------------
    #
    # These translate toolkit-agnostic matplotlib events into wire-protocol
    # messages.  Because every GUI backend (Qt, GTK, Tk, …) already creates
    # these events from its native UI handling, no per-toolkit forwarding
    # code is needed.

    # Map from mpl callback event names to wire-protocol type strings.
    _MPL_TO_WIRE_EVENT: ClassVar[dict[str, str]] = {
        "button_press_event": "button_press",
        "button_release_event": "button_release",
        "motion_notify_event": "motion_notify",
        "scroll_event": "scroll",
        "key_press_event": "key_press",
        "key_release_event": "key_release",
    }

    def _on_mpl_mouse_event(self, event: MouseEvent) -> None:
        """Translate an mpl MouseEvent → wire-protocol mouse message.

        Converts from matplotlib figure coordinates (y from bottom,
        physical pixels) to wire-protocol coordinates (y from top,
        physical pixels) and forwards to the server.

        Subclasses may override this to add throttling (e.g. for
        ``motion_notify`` events).
        """
        wire_type = self._MPL_TO_WIRE_EVENT.get(event.name)
        if wire_type is None:
            return
        # mpl coords: physical pixels, y=0 at bottom.
        # wire coords: physical pixels, y=0 at top.
        wire_x = event.x
        wire_y = self.figure.bbox.height - event.y

        if event.name == "motion_notify_event":
            self._forward_motion_notify(wire_x, wire_y)
        elif event.name == "scroll_event":
            self._forward_mouse_event(
                wire_type, wire_x, wire_y, step=event.step
            )
        else:
            # button_press / button_release
            button = int(event.button) - 1 if event.button is not None else 0
            self._forward_mouse_event(
                wire_type, wire_x, wire_y, button=button
            )

    def _forward_motion_notify(self, x: float, y: float) -> None:
        """Rate-limited motion_notify forwarding.

        Discrete events (press, release, scroll) are forwarded
        immediately.  Motion events are throttled to
        ``_mouse_move_interval`` Hz with a trailing-edge send so the
        server always sees the final pointer position.
        """
        now = time.monotonic()
        elapsed = now - self._last_mouse_move_time
        if elapsed >= self._mouse_move_interval:
            # Enough time has passed — send immediately.
            self._pending_mouse_move = None
            if self._mouse_move_timer is not None:
                self._mouse_move_timer.cancel()
                self._mouse_move_timer = None
            self._last_mouse_move_time = now
            self._forward_mouse_event("motion_notify", x, y)
        else:
            # Too soon — stash and schedule a trailing send so the
            # server always sees the final position.
            self._pending_mouse_move = (x, y)
            if self._mouse_move_timer is None:
                remaining = self._mouse_move_interval - elapsed
                self._mouse_move_timer = threading.Timer(
                    remaining, self._flush_mouse_move
                )
                self._mouse_move_timer.daemon = True
                self._mouse_move_timer.start()

    def _flush_mouse_move(self) -> None:
        """Send the most recent throttled motion_notify."""
        self._mouse_move_timer = None
        if self._pending_mouse_move is not None:
            wx, wy = self._pending_mouse_move
            self._pending_mouse_move = None
            self._last_mouse_move_time = time.monotonic()
            self._forward_mouse_event("motion_notify", wx, wy)

    def _on_mpl_key_event(self, event: KeyEvent) -> None:
        """Translate an mpl KeyEvent → wire-protocol key message."""
        wire_type = self._MPL_TO_WIRE_EVENT.get(event.name)
        if wire_type is None or event.key is None:
            return
        self._forward_key_event(wire_type, event.key)

    # -- event forwarding (main thread → transport) -------------------------

    def _forward_mouse_event(
        self,
        event_type: str,
        x: float,
        y: float,
        button: int = 0,
        step: float = 0,
    ) -> None:
        """Forward a mouse event to the server.

        Parameters
        ----------
        event_type : str
            E.g. ``"button_press"``, ``"button_release"``, ``"motion_notify"``.
        x, y : float
            Physical (device) pixel coordinates in wire-protocol
            convention (x from left, y from **top**).  The server
            flips y internally to obtain matplotlib figure
            coordinates.
        button : int
            Mouse button, **0-indexed** (wire protocol / JS convention:
            0 = left, 1 = middle, 2 = right).  The server adds 1 to
            convert to matplotlib's 1-indexed ``MouseButton`` values.
        step : float
            Scroll step (for scroll events).
        """
        msg: dict[str, Any] = {"type": event_type, "x": x, "y": y, "button": button}
        if event_type == "scroll":
            msg["step"] = step
        self._transport.send_json(msg)

    def _forward_key_event(self, event_type: str, key: str) -> None:
        """Forward a keyboard event to the server.

        Parameters
        ----------
        event_type : str
            ``"key_press"`` or ``"key_release"``.
        key : str
            Key name (matplotlib format).
        """
        self._transport.send_json({"type": event_type, "key": key})

    def _forward_resize(self, width: int, height: int) -> None:
        """Forward a resize event to the server.

        Parameters
        ----------
        width, height : int
            New size in CSS (logical) pixels.  The server will multiply
            by the device pixel ratio to obtain physical pixels.
        """
        self._transport.send_json({"type": "resize", "width": width, "height": height})

    def _forward_set_device_pixel_ratio(self, dpr: float) -> None:
        """Forward a device-pixel-ratio change to the server.

        Parameters
        ----------
        dpr : float
            The new device pixel ratio.
        """
        self._transport.send_json(
            {"type": "set_device_pixel_ratio", "device_pixel_ratio": dpr}
        )
        # Update the transport's record so future reconnections use it
        self._transport._device_pixel_ratio = dpr

    def _forward_toolbar_button(self, name: str) -> None:
        """Forward a toolbar button press to the server.

        Parameters
        ----------
        name : str
            Button name (``"pan"``, ``"zoom"``, ``"home"``, etc.).
        """
        self._transport.send_json({"type": "toolbar_button", "name": name})

    def _forward_save_figure(
        self,
        format: str = "png",
        dpi: float = 100.0,
        transparent: bool = False,
    ) -> None:
        """Send a ``save_figure`` request to the server.

        The response (``save_complete`` or ``save_error``) arrives via
        the normal receive loop and is dispatched to
        ``toolbar._on_save_complete()`` / ``toolbar._on_save_error()``.
        """
        self._transport.send_json(
            {
                "type": "save_figure",
                "format": format,
                "dpi": dpi,
                "transparent": transparent,
            }
        )

    def _forward_update_params(self, params: dict[str, Any]) -> None:
        """Send an ``update_params`` message to the server.

        The server validates the parameters against the ``update_schema``,
        calls the update function, and triggers an ``invalidate`` → re-render
        cycle.  If the parameters are invalid, the server sends an ``error``
        message (dispatched by :meth:`_on_json_message`).

        Parameters
        ----------
        params : dict
            Parameter names and values matching the server's ``update_schema``.
        """
        self._transport.send_json({"type": "update_params", "params": params})

    # -- save (programmatic fig.savefig) ------------------------------------

    # Pending save callback: set by print_figure, consumed by _on_json_message
    # when a save_complete / save_error arrives.
    _pending_print_figure_callback: (
        tuple[str, Callable[[dict[str, Any]], None]] | None
    ) = None

    def print_figure(
        self,
        filename: str | os.PathLike[Any] | IO[Any],
        dpi: float | str | None = None,
        facecolor: Any = None,  # noqa: ARG002
        edgecolor: Any = None,  # noqa: ARG002
        orientation: str = "portrait",  # noqa: ARG002
        format: str | None = None,
        *,
        bbox_inches: Any = None,  # noqa: ARG002
        pad_inches: Any = None,  # noqa: ARG002
        bbox_extra_artists: Any = None,  # noqa: ARG002
        backend: str | None = None,  # noqa: ARG002
        transparent: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Save the figure via the remote server.

        This overrides the base-class method so that
        ``fig.savefig("plot.pdf", dpi=150)`` works.  The server renders
        the file (supporting vector formats, correct DPI, etc.) and
        returns a ``download_url``.  The file is then fetched via HTTP
        and written to *filename*.

        The save request is sent through the normal message flow.  The
        ``save_complete`` response is intercepted by
        :meth:`_on_json_message` and routed to either the pending
        callback registered here, or to ``toolbar._on_save_complete``
        if no programmatic save is pending.

        Parameters
        ----------
        filename : str, path-like, or file-like
            Destination file path.  File-like objects are not supported
            for remote saves; a ``ValueError`` is raised.
        dpi : float or 'figure', optional
            Resolution.  ``'figure'`` uses the figure's own DPI.
        format : str, optional
            File format (e.g. ``'png'``, ``'pdf'``).  Inferred from
            *filename* extension if not given.
        transparent : bool
            Passed to the server.
        **kwargs
            Absorbed for compatibility with ``Figure.savefig``.
        """


        if hasattr(filename, "write"):
            raise ValueError(
                "Remote print_figure does not support file-like objects; "
                "pass a file path instead."
            )
        path = Path(filename)  # type: ignore[arg-type]

        # Resolve format
        if format is None:
            fmt = path.suffix.lstrip(".").lower()
            if not fmt:
                fmt = mpl.rcParams.get("savefig.format", "png")
        else:
            fmt = format.lower()

        # Resolve DPI
        resolved_dpi: float
        if dpi is None:
            dpi = mpl.rcParams.get("savefig.dpi", "figure")
        if dpi == "figure":
            resolved_dpi = float(getattr(self.figure, "_original_dpi", self.figure.dpi))
        else:
            resolved_dpi = float(dpi)  # type: ignore[arg-type]

        # Register a callback so _on_json_message routes save_complete here
        def on_complete(msg: dict[str, Any]) -> None:
            if msg.get("type") == "save_error":
                raise RuntimeError(
                    f"Server save failed: {msg.get('message', 'unknown')}"
                )

        self._pending_print_figure_callback = (str(path), on_complete)

        # Send the request
        self._forward_save_figure(format=fmt, dpi=resolved_dpi, transparent=transparent)

        # The actual download happens asynchronously when _on_json_message
        # receives save_complete and calls _handle_print_figure_complete.
        # For synchronous callers the toolkit event loop must be running.

    def _handle_print_figure_complete(self, filepath: str, msg: dict[str, Any]) -> None:
        """Download the saved file after a ``save_complete`` response.

        Called by :meth:`_on_json_message` when a pending
        ``print_figure`` save completes.
        """
        download_url = msg.get("download_url", "")
        if not download_url:
            logger.error("save_complete without download_url")
            return
        try:
            self._download_url_to_file(download_url, filepath)
            logger.info("Saved figure to %s", filepath)
        except Exception:
            logger.exception("Failed to download saved figure")

    def _download_url_to_file(self, download_url: str, filepath: str) -> None:
        """HTTP GET *download_url* and write the response to *filepath*.

        Builds a full URL from the transport's WebSocket URL and the
        relative *download_url* path returned by the server.  Auth
        credentials are forwarded via the ``Authorization: Bearer``
        header (if a token is present in the WS URL query string).
        """
        import urllib.parse
        import urllib.request

        ws_url = self._transport._url
        parsed = urllib.parse.urlparse(ws_url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        base = f"{scheme}://{parsed.netloc}"
        full_url = urllib.parse.urljoin(base, download_url)

        # Guard against a malicious download_url redirecting to another host.
        resolved = urllib.parse.urlparse(full_url)
        if resolved.netloc != parsed.netloc:
            raise ValueError(
                f"download_url resolved to a different host: {full_url!r}"
            )

        # Forward auth token from WS query string via Authorization header.
        ws_qs = urllib.parse.parse_qs(parsed.query)
        token = (ws_qs.get("token", [None]) or [None])[0]

        req = urllib.request.Request(full_url)
        if token is not None:
            req.add_header("Authorization", f"Bearer {token}")
        with urllib.request.urlopen(req, timeout=30) as resp:
            Path(filepath).write_bytes(resp.read())

    # -- abstract (toolkit must implement) ----------------------------------

    def schedule_repaint(self) -> None:
        """Ask the toolkit to schedule a repaint.

        Must be overridden by the toolkit subclass (e.g. ``self.update()``
        in Qt).
        """
        # Default: no-op (headless / testing mode)


class RemoteNavigationToolbar2(NavigationToolbar2):
    """Navigation toolbar that delegates all actions to the remote server.

    Toolbar button presses are forwarded over the WebSocket.  The toolbar
    UI state (message, history buttons, navigate mode) is driven entirely
    by server push messages.

    This toolbar does **not** call ``super().pan()`` / ``super().zoom()``
    because there are no local axes to manipulate.
    """

    canvas: FigureCanvasRemote  # type: ignore[assignment]

    # Keep the standard toolitems but drop Subplots / Customize
    # (those require local axes which we don't have).
    toolitems = (
        *tuple(
            item
            for item in NavigationToolbar2.toolitems
            if item[3] in {"home", "back", "forward", "pan", "zoom", None}
        ),
        ("Download", "Download plot", "filesave", "download"),
    )

    def __init__(self, canvas: FigureCanvasRemote, *args: Any, **kwargs: Any) -> None:
        self.message = ""
        # Suppress set_history_buttons during __init__ (no nav stack yet)
        self._initializing = True
        super().__init__(canvas, *args, **kwargs)
        self._initializing = False

    # -- toolbar actions → send to server -----------------------------------

    def home(self, *args: Any) -> None:  # noqa: ARG002
        """Reset view to home."""
        self.canvas._forward_toolbar_button("home")

    def back(self, *args: Any) -> None:  # noqa: ARG002
        """Navigate back in view history."""
        self.canvas._forward_toolbar_button("back")

    def forward(self, *args: Any) -> None:  # noqa: ARG002
        """Navigate forward in view history."""
        self.canvas._forward_toolbar_button("forward")

    def pan(self, *args: Any) -> None:  # noqa: ARG002
        """Toggle pan/zoom mode."""
        self.canvas._forward_toolbar_button("pan")

    def zoom(self, *args: Any) -> None:  # noqa: ARG002
        """Toggle zoom mode."""
        self.canvas._forward_toolbar_button("zoom")

    def download(self, *args: Any) -> None:  # noqa: ARG002
        """Trigger download (save) via server."""
        self.canvas._forward_toolbar_button("download")

    def save_figure(self, *args: Any) -> None:  # noqa: ARG002
        """Save figure via the server."""
        self.canvas._forward_toolbar_button("download")

    # -- state updates from server push messages ----------------------------

    def set_message(self, s: str) -> None:
        """Update the toolbar status message.

        Called by the canvas's ``_on_json_message`` when a ``"message"``
        arrives from the server.
        """
        self.message = s

    def set_history_buttons(self) -> None:
        """No-op — history buttons are driven by the server."""
        if self._initializing:
            return

    def _on_history_buttons(self, *, back: bool, forward: bool) -> None:
        """Handle a ``history_buttons`` message from the server.

        Subclasses (Qt, GTK, …) override this to enable/disable buttons.
        """
        logger.debug("history_buttons: back=%s, forward=%s", back, forward)

    def _on_navigate_mode(self, mode: str) -> None:
        """Handle a ``navigate_mode`` message from the server.

        Subclasses override this to update toggle-button states.
        """
        logger.debug("navigate_mode: %s", mode)

    def _on_save_complete(self, msg: dict[str, Any]) -> None:
        """Handle a ``save_complete`` message from the server.

        Subclasses override this to download the saved file.
        """
        logger.debug("save_complete: %s", msg)

    def _on_save_error(self, msg: dict[str, Any]) -> None:
        """Handle a ``save_error`` message from the server."""
        logger.error("save_error: %s", msg.get("message", "unknown"))

    def _on_update_params_submitted(self, params: dict[str, Any]) -> None:
        """Handle an update-parameters submission from the UI.

        The default implementation forwards the params to the server
        via the canvas.  Toolkit subclasses may override this to add
        feedback (e.g. disable the submit button until the invalidate
        response arrives).

        Parameters
        ----------
        params : dict
            Parameter names and values from the update form.
        """
        self.canvas._forward_update_params(params)

    def draw_rubberband(
        self,
        event: Any,  # noqa: ARG002
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Draw rubberband — delegated to canvas for toolkit rendering."""
        self.canvas._rubberband_rect = (x0, y0, x1, y1)
        self.canvas.schedule_repaint()

    def remove_rubberband(self) -> None:
        """Remove rubberband."""
        self.canvas._rubberband_rect = None
        self.canvas.schedule_repaint()


# ---------------------------------------------------------------------------
# Remote plot discovery
# ---------------------------------------------------------------------------


@dataclass
class RemotePlotInfo:
    """Metadata about a single remote plot, returned by :func:`list_remote_figures`.

    Attributes
    ----------
    name : str
        Plot name (key in the server's plot registry).
    description : str
        Human-readable plot description.
    ws_url : str
        Full WebSocket URL for this plot's v0 endpoint.
    view_url : str
        Full HTTP URL for the browser-based viewer.
    init_schema : dict
        JSON Schema for initialisation parameters.
    update_schema : dict or None
        JSON Schema for update parameters, or *None* if the plot
        does not support dynamic updates.
    """

    name: str
    description: str
    ws_url: str
    view_url: str
    init_schema: dict[str, Any] = field(default_factory=dict)
    update_schema: dict[str, Any] | None = None


def list_remote_figures(
    base_url: str,
    *,
    token: str | None = None,
    timeout: float = 10.0,
) -> list[RemotePlotInfo]:
    """Query a remote mpl_fastapi server for available plots.

    Fetches the ``/plots`` JSON endpoint and returns a list of
    :class:`RemotePlotInfo` objects describing each available figure.

    Parameters
    ----------
    base_url : str
        Server base URL.  Both ``ws://`` / ``wss://`` and
        ``http://`` / ``https://`` schemes are accepted — WebSocket
        schemes are converted to HTTP automatically.
    token : str, optional
        Authentication token.  Sent via ``Authorization: Bearer``
        header when provided.
    timeout : float
        HTTP request timeout in seconds.  Default 10.

    Returns
    -------
    list of RemotePlotInfo

    Raises
    ------
    urllib.error.URLError
        If the server is unreachable.
    RuntimeError
        If the server returns an unexpected response.
    """
    parsed = urllib.parse.urlparse(base_url)
    scheme = parsed.scheme
    if scheme in ("ws", "wss"):
        scheme = "https" if scheme == "wss" else "http"
    http_base = f"{scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    url = f"{http_base}/plots"

    req = urllib.request.Request(url)
    if token is not None:
        req.add_header("Authorization", f"Bearer {token}")

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())

    plots_dict = data.get("plots", {})
    results: list[RemotePlotInfo] = []
    for name, info in plots_dict.items():
        results.append(
            RemotePlotInfo(
                name=name,
                description=info.get("description", ""),
                ws_url=info.get("ws_url", ""),
                view_url=info.get("view_url", ""),
                init_schema=info.get("parameters", {}),
                update_schema=info.get("update_schema"),
            )
        )
    return results


class FigureManagerRemote(FigureManagerBase):
    """Minimal figure manager for remote canvases.

    Toolkit subclasses (e.g. ``FigureManagerQTRemote``) add window
    management, toolbar wiring, and lifecycle handling.
    """

    canvas: FigureCanvasRemote  # type: ignore[assignment]
    toolbar: RemoteNavigationToolbar2 # type: ignore[assignment]

    def __init__(self, canvas: FigureCanvasRemote, num: int) -> None:
        super().__init__(canvas, num)
        self.toolbar = RemoteNavigationToolbar2(canvas)  # type: ignore[assignment]

    def destroy(self) -> None:
        """Disconnect the transport and clean up."""
        # Transport cleanup is handled by the toolkit layer which owns
        # the thread / event loop.
