"""Matplotlib backend implementation for FastAPI integration.

This module provides the matplotlib backend classes that handle:
- Figure rendering with differential updates (FastAPICanvas)
- Navigation toolbar with WebSocket communication (NavigationToolbar2FastAPI)
- Figure manager and backend registration (FastAPIManger, FastAPIBackend)

The backend uses WebSocket for bidirectional communication and supports
efficient differential image updates to minimize data transfer.
"""

import logging
from collections import deque
from io import BytesIO
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi import WebSocket
from matplotlib.backend_bases import (
    FigureManagerBase,
    KeyEvent,
    LocationEvent,
    MouseEvent,
    NavigationToolbar2,
    _Backend,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from PIL import Image

# Set up logging
logger = logging.getLogger(__name__)


def _register_backend(backend_class: type) -> None:
    """
    Register and activate a matplotlib backend.

    Parameters
    ----------
    backend_class : type
        The backend class with FigureCanvas and FigureManager attributes.
    """
    import matplotlib
    import matplotlib.backend_bases

    # Register our backend module
    matplotlib.backend_bases.register_backend(
        "module://mpl_fastapi.backend", backend_class
    )

    # Use our backend
    matplotlib.use("module://mpl_fastapi.backend", force=True)


class FastAPICanvas(FigureCanvasAgg):
    """Matplotlib canvas with WebSocket communication for FastAPI.

    This canvas extends the Agg backend with:
    - Differential image updates to reduce bandwidth
    - Event queue for async WebSocket messages
    - Mouse and keyboard event handling from browser
    - Server-side blitting support: ``blit()`` immediately encodes the
      current renderer buffer as a diff PNG and enqueues it in
      ``_binary_queue`` for the router to push to the client, bypassing
      the usual invalidate → render round-trip.
    """

    # Attributes from this class
    _force_full: bool
    _png_is_old: bool
    _current_image_mode: str
    _msg_queue: deque[dict[str, Any]]
    _binary_queue: deque[tuple[bytes, bool]]
    _last_buff: npt.NDArray[np.uint32]
    _renderer: RendererAgg
    supports_binary: bool = True
    supports_blit: bool = True

    # Declare attributes from parent FigureCanvasBase that we use
    call_info: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._force_full = False
        self._png_is_old = False
        self._current_image_mode = "full"
        self._msg_queue = deque()
        # Queued (png_bytes, is_diff) tuples pushed by blit(); drained by the
        # router's _drain_canvas() helper and sent as unsolicited binary frames.
        self._binary_queue: deque[tuple[bytes, bool]] = deque()

    def start_event_loop(self, timeout: float = 0) -> None:
        self.call_info["start_event_loop"] = {"timeout": timeout}

    async def handle_unknown_event(
        self,
        ev: dict[str, Any],
        websocket: WebSocket,  # noqa: ARG002
    ) -> None:
        logger.debug("Unknown event type: %s, data: %s", ev["type"], ev)
        return

    async def handle_ack(self, ev: dict[str, Any], websocket: WebSocket) -> None: ...

    async def handle_resize(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        w = int(ev["width"] * self.device_pixel_ratio)
        h = int(ev["height"] * self.device_pixel_ratio)
        fig = self.figure
        fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
        px_w, px_h = fig.bbox.size
        await websocket.send_json(
            {
                "type": "resize",
                "size": (
                    round(px_w / self.device_pixel_ratio),
                    round(px_h / self.device_pixel_ratio),
                ),
                "forward": True,
            }
        )

    async def handle_set_device_pixel_ratio(
        self, ev: dict[str, Any], websocket: WebSocket
    ) -> None:
        device_pixel_ratio = ev["device_pixel_ratio"]
        if self._set_device_pixel_ratio(device_pixel_ratio):  # type: ignore[attr-defined]
            self._force_full = True
            await websocket.send_json({"type": "invalidate"})

    async def handle_send_image_mode(
        self,
        ev: dict[str, Any],  # noqa: ARG002
        websocket: WebSocket,
    ) -> None:
        await websocket.send_json(
            {"type": "image_mode", "mode": self._current_image_mode}
        )

    def get_renderer(self, cleared: bool | None = None) -> RendererAgg:
        """Get renderer with caching for differential updates."""
        # Mirrors super.get_renderer, but caches the old one so that we can do
        # things such as produce a diff image in get_diff_image.
        w, h = np.asarray(self.figure.bbox.size).astype(int)
        key = w, h, self.figure.dpi
        try:
            _, _ = self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self._renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key  # type: ignore[assignment]
            self._last_buff = np.copy(
                np.frombuffer(self._renderer.buffer_rgba(), dtype=np.uint32).reshape(
                    (self._renderer.height, self._renderer.width)
                )
            )

        elif cleared:
            self._renderer.clear()

        return self._renderer

    def draw(self) -> None:
        """Render the figure and mark the PNG buffer as stale.

        Sets ``_png_is_old = True`` so that the next call to
        :meth:`get_diff_image` (from either the render/refresh path or
        :meth:`blit`) will re-encode the buffer.
        """
        self._png_is_old = True
        super().draw()

    def get_diff_image(self) -> tuple[bytes, bool] | None:
        """Encode the current renderer buffer as a diff or full PNG.

        Matches ``FigureCanvasWebAggCore.get_diff_image()`` semantics from
        upstream matplotlib's WebAgg backend.

        Returns ``None`` when the buffer has not changed since the last call
        (i.e. ``_png_is_old`` is False).  Otherwise returns a
        ``(png_bytes, is_diff)`` tuple where *is_diff* is ``True`` when only
        changed pixels are encoded (transparent elsewhere).

        Notes
        -----
        This method is intentionally synchronous — it runs in a thread-pool
        worker via :meth:`draw_and_get_diff` for the render/refresh path, and
        directly on the caller's thread for :meth:`blit`.
        """
        if not self._png_is_old:
            return None

        renderer = self.get_renderer()

        # Buffer as uint32 for whole-pixel comparison (one call vs. per-plane).
        buff = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint32).reshape(
            (int(renderer.height), int(renderer.width))
        )
        pixels = buff.view(dtype=np.uint8).reshape((*buff.shape, 4))

        if (
            self._force_full
            or buff.shape != self._last_buff.shape
            or np.any(pixels[:, :, 3] != 255)
        ):
            is_diff = False
            output = buff
        else:
            is_diff = True
            diff = buff != self._last_buff
            output = np.where(diff, buff, 0)

        # Store current buffer for the next diff.
        np.copyto(self._last_buff, buff)
        self._force_full = False
        self._png_is_old = False

        # Update tracked image mode for send_image_mode responses.
        self._current_image_mode = "diff" if is_diff else "full"

        data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
        png_buf = BytesIO()
        Image.fromarray(data).save(png_buf, format="png")
        return png_buf.getvalue(), is_diff

    def draw_and_get_diff(self) -> tuple[bytes, bool]:
        """Draw the figure and return a diff or full PNG.

        Combines :meth:`draw` and :meth:`get_diff_image` in a single call
        suitable for running in a thread-pool worker via
        ``loop.run_in_executor``.

        This is the canonical entry-point for the render/refresh path in the
        router — all rendering logic stays on the canvas rather than leaking
        into the router module.

        Returns
        -------
        tuple[bytes, bool]
            ``(png_bytes, is_diff)`` — same contract as :meth:`get_diff_image`
            but always returns a result (forces a full render if needed).
        """
        self.draw()
        result = self.get_diff_image()
        if result is None:
            # Defensive fallback: draw() should always set _png_is_old, but
            # guard here so callers never receive None.
            self._force_full = True
            self._png_is_old = True
            result = self.get_diff_image()
            assert result is not None  # guaranteed after force_full + png_is_old
        return result

    def blit(self, bbox: Any = None) -> None:  # noqa: ARG002
        """Push the current renderer state to connected clients immediately.

        Encodes the renderer buffer as a diff PNG (via :meth:`get_diff_image`)
        and places it in ``_binary_queue`` (at most one entry at a time).
        The router's ``_drain_canvas()`` helper will pick this up after
        processing each client message and send it as an unsolicited binary
        frame, bypassing the usual ``invalidate`` → client ``render``
        round-trip.

        **Coalescing rule:** if a previous blit has not yet been drained from
        the queue, that frame is discarded and ``_force_full`` is set so that
        the replacement frame is a full image.  This guarantees the queue
        never holds more than one entry and the client never receives a diff
        chain whose base has not arrived yet.

        Parameters
        ----------
        bbox : `~matplotlib.transforms.BboxBase` or None
            Region to blit.  Ignored — the full renderer buffer is always
            encoded, matching upstream WebAgg behaviour (partial blits cannot
            be expressed in the PNG diff protocol).
        """
        self._png_is_old = True

        if self._binary_queue:
            # A previous blit is still pending — discard it and force a full
            # frame so the client receives a self-contained image rather than
            # a diff whose base may not have been painted yet.
            logger.debug("blit(): coalescing pending blit into full frame")
            self._binary_queue.clear()
            self._force_full = True

        result = self.get_diff_image()
        if result is not None:
            logger.debug(
                "blit(): queuing %s image (%d bytes)",
                "diff" if result[1] else "full",
                len(result[0]),
            )
            self._binary_queue.append(result)

    async def _handle_mouse(self, event: dict[str, Any], _websocket: WebSocket) -> None:
        """Handle mouse events from the browser."""
        x = event["x"]
        y = event["y"]
        renderer_height = self.get_renderer().height
        y = renderer_height - y

        # Javascript button numbers and matplotlib button numbers are
        # off by 1
        button = event["button"] + 1

        e_type = event["type"]
        gui_event = event.get("guiEvent")
        if e_type == "button_press":
            MouseEvent(
                "button_press_event", self, x, y, button, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "dblclick":
            MouseEvent(
                "button_press_event",
                self,
                x,
                y,
                button,
                dblclick=True,
                guiEvent=gui_event,
            )._process()  # type: ignore[attr-defined]
        elif e_type == "button_release":
            MouseEvent(
                "button_release_event", self, x, y, button, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "motion_notify":
            MouseEvent("motion_notify_event", self, x, y, guiEvent=gui_event)._process()  # type: ignore[attr-defined]
        elif e_type == "figure_enter":
            LocationEvent(
                "figure_enter_event", self, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "figure_leave":
            LocationEvent(
                "figure_leave_event", self, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "scroll":
            MouseEvent(
                "scroll_event", self, x, y, step=event["step"], guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]

    handle_button_press = handle_button_release = handle_dblclick = (
        handle_figure_enter
    ) = handle_figure_leave = handle_motion_notify = handle_scroll = _handle_mouse

    async def _handle_key(self, event: dict[str, Any], _websocket: WebSocket) -> None:
        """Handle key press/release events from the browser."""
        key = event.get("key", "")
        gui_event = event.get("guiEvent")
        x = event.get("x", 0)
        y = event.get("y", 0)
        renderer_height = self.get_renderer().height
        y = renderer_height - y
        e_type = event["type"]
        if e_type == "key_press":
            KeyEvent(
                "key_press_event", self, key, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "key_release":
            KeyEvent(
                "key_release_event", self, key, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]

    handle_key_press = handle_key_release = _handle_key

    async def handle_toolbar_button(
        self, event: dict[str, Any], _websocket: WebSocket
    ) -> None:
        """Handle toolbar button clicks from the browser."""
        name = event.get("name", "")
        if name not in _ALLOWED_TOOL_ITEMS or name is None:
            logger.warning("Blocked unknown toolbar action: %r", name)
            return
        logger.info("Toolbar button pressed: %s", name)
        getattr(self.toolbar, name)()
        # Queue an invalidate event for the client to request render
        self.queue_event("invalidate")

    def queue_event(self, event_type: str, **kwargs: Any) -> None:
        """Queue a message to be sent to the client."""
        logger.debug("Queueing event: type=%s, kwargs=%s", event_type, kwargs)
        self._msg_queue.append({"type": event_type, **kwargs})

    def draw_idle(self) -> None:
        """Queue an invalidate event to notify client figure has changed."""
        self.queue_event("invalidate")

    async def drain_queue(self, websocket: WebSocket) -> None:
        """Send all queued JSON messages to the client."""
        while len(self._msg_queue):
            payload = self._msg_queue.popleft()
            logger.debug("Sending message to client: type=%s", payload.get("type"))
            await websocket.send_json(payload)


_ALLOWED_TOOL_ITEMS: set[str | None] = {
    "home",
    "back",
    "forward",
    "pan",
    "zoom",
    "download",
    None,
}


class NavigationToolbar2FastAPI(NavigationToolbar2):
    """Navigation toolbar with WebSocket communication.

    Sends toolbar state changes and navigation events to the browser client.
    """

    # Use the standard toolbar items + download button
    toolitems = tuple(
        (text, tooltip_text, image_file, name_of_method)  # type: ignore[misc]
        for text, tooltip_text, image_file, name_of_method in (
            *NavigationToolbar2.toolitems,
            ("Download", "Download plot", "filesave", "download"),
        )
        if name_of_method in _ALLOWED_TOOL_ITEMS
    )

    message: str
    _cursor: None
    canvas: FastAPICanvas  # Override parent type to be more specific

    # Declare parent class attribute we use
    _nav_stack: Any  # NavigationToolbar2._NavStack

    def __init__(self, canvas: FastAPICanvas) -> None:
        self.message = ""
        self._cursor = None  # Remove with deprecation.

        # Call parent __init__ but suppress set_history_buttons
        # We'll call it explicitly later to make initialization deterministic
        self._defer_history_buttons = True
        super().__init__(canvas)
        self._defer_history_buttons = False

    def set_message(self, message: str) -> None:
        """Display a message in the browser toolbar."""
        # logger.debug(f"set_message called: '{message}'")
        if message != self.message:
            self.canvas.queue_event("message", message=message)
        self.message = message

    def draw_rubberband(
        self,
        event: Any,  # noqa: ARG002
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Draw a rubberband selection rectangle."""
        self.canvas.queue_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def remove_rubberband(self) -> None:
        """Remove the rubberband selection rectangle."""
        self.canvas.queue_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    def save_figure(self, *args: Any) -> None:  # noqa: ARG002
        """Save the current figure"""
        self.canvas.queue_event("save")

    def pan(self) -> None:
        """Activate pan/zoom mode."""
        super().pan()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def zoom(self) -> None:
        """Activate zoom mode."""
        super().zoom()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def set_history_buttons(self) -> None:
        """Update the enabled state of back/forward buttons."""
        # During __init__, defer this to make initialization deterministic
        if getattr(self, "_defer_history_buttons", False):
            return

        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        self.canvas.queue_event(
            "history_buttons", Back=can_backward, Forward=can_forward
        )


class FastAPIManger(FigureManagerBase):
    """Figure manager for FastAPI integration.

    Manages the canvas and toolbar, and provides the JavaScript bundle
    for the browser client.
    """

    ToolbarCls = NavigationToolbar2FastAPI
    web_sockets: set[WebSocket]
    toolbar: NavigationToolbar2FastAPI
    supports_binary: bool = True

    def __init__(self, canvas: FastAPICanvas, num: int) -> None:
        self.web_sockets = set()
        super().__init__(canvas, num)
        self.toolbar = self.ToolbarCls(canvas)

    @classmethod
    def get_toolbar_config(cls) -> dict[str, Any]:
        """Get toolbar configuration as JSON-serializable dict."""
        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(["", "", "", ""])
            else:
                toolitems.append([name, tooltip, image, method])  # type: ignore[list-item]

        save_formats = []
        for _filetype, ext in sorted(
            FastAPICanvas.get_supported_filetypes_grouped().items()
        ):
            save_formats.append(ext[0])

        return {
            "toolbar_items": toolitems,
            "save_formats": save_formats,
            "default_save_format": FastAPICanvas.get_default_filetype(),
        }


class FastAPIBackend(_Backend):
    """Matplotlib backend for FastAPI integration."""

    FigureCanvas = FastAPICanvas
    FigureManager = FastAPIManger


# Configure matplotlib to use our backend
_register_backend(FastAPIBackend)
