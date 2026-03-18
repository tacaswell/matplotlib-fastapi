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
from typing import Any

import numpy as np
import numpy.typing as npt
from fastapi import WebSocket
from matplotlib.backend_bases import (
    FigureManagerBase,
    LocationEvent,
    MouseEvent,
    NavigationToolbar2,
    _Backend,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg

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
    """

    # Attributes from this class
    _force_full: bool
    _current_image_mode: str
    _msg_queue: deque[dict[str, Any]]
    _last_buff: npt.NDArray[np.uint32]
    _renderer: RendererAgg
    supports_binary: bool = True

    # Declare attributes from parent FigureCanvasBase that we use
    call_info: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._force_full = False
        self._current_image_mode = "full"
        self._msg_queue = deque()

    def start_event_loop(self, timeout: float = 0) -> None:
        self.call_info["start_event_loop"] = {"timeout": timeout}

    async def handle_unknown_event(
        self,
        ev: dict[str, Any],
        websocket: WebSocket,  # noqa: ARG002
    ) -> None:
        logger.debug(f"Unknown event type: {ev['type']}, data: {ev}")
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

    async def handle_toolbar_button(
        self, event: dict[str, Any], _websocket: WebSocket
    ) -> None:
        """Handle toolbar button clicks from the browser."""
        logger.info(f"Toolbar button pressed: {event['name']}")
        # Call the toolbar method
        getattr(self.toolbar, event["name"])()
        # Queue an invalidate event for the client to request render
        self.queue_event("invalidate")

    def queue_event(self, event_type: str, **kwargs: Any) -> None:
        """Queue a message to be sent to the client."""
        logger.debug(f"Queueing event: type={event_type}, kwargs={kwargs}")
        self._msg_queue.append({"type": event_type, **kwargs})

    def draw_idle(self) -> None:
        """Queue an invalidate event to notify client figure has changed."""
        self.queue_event("invalidate")

    async def drain_queue(self, websocket: WebSocket) -> None:
        """Send all queued messages to the client."""
        while len(self._msg_queue):
            payload = self._msg_queue.popleft()
            logger.debug(f"Sending message to client: type={payload.get('type')}")
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
