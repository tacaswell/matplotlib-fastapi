"""Qt thin-client backend for remote Matplotlib rendering.

This module provides the Qt-specific widgets and integration layer for
displaying server-rendered Matplotlib figures.  It builds on the
toolkit-agnostic :mod:`~mpl_fastapi.remote.backend_remote` layer,
adding:

* **TransportThread** — a :class:`QThread` running the asyncio
  transport, with signals that marshal incoming messages to the main
  thread.
* **FigureCanvasQTRemote** — a :class:`QWidget` that paints the most
  recently received image and forwards mouse/key/resize events to the
  server.
* **NavigationToolbar2QTRemote** — a :class:`QToolBar` whose buttons
  send toolbar commands to the server.
* **FigureManagerQTRemote** — wires canvas + toolbar + window chrome.
* **open_remote_figure()** — factory function that creates all of the
  above from a server URL and plot name.

Usage::

    from PySide6.QtWidgets import QApplication
    from mpl_fastapi.remote.backend_qtremote import open_remote_figure

    app = QApplication.instance() or QApplication([])
    manager = open_remote_figure(
        url="ws://localhost:8000/plots",
        plot_name="sine",
        init_params={"frequency": 2.0},
    )
    manager.show()
    app.exec()
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from matplotlib import cbook
from matplotlib.backend_bases import (
    CloseEvent,
    KeyEvent,
    MouseEvent,
    ResizeEvent,
)
from matplotlib.backends.backend_qt import (
    FigureCanvasQT,
    FigureManagerQT,
    MainWindow,
    NavigationToolbar2QT,
    _create_qApp,
)
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets
from matplotlib.figure import Figure

from mpl_fastapi.remote.backend_remote import (
    FigureCanvasRemote,
    RemoteNavigationToolbar2,
)
from mpl_fastapi.remote.transport import (
    RemoteTransport,
    ServerConfig,
    build_ws_url,
)

__all__ = [
    "FigureCanvasQTRemote",
    "FigureManagerQTRemote",
    "NavigationToolbar2QTRemote",
    "TransportThread",
    "open_remote_figure",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transport thread — runs asyncio event loop on a QThread
# ---------------------------------------------------------------------------


class TransportThread(QtCore.QThread):
    """Background QThread that runs the asyncio transport loop.

    Incoming messages are emitted as Qt signals so that they are
    delivered on the main thread via the Qt event loop's signal/slot
    mechanism.

    Parameters
    ----------
    transport : RemoteTransport
        The transport instance to run.  Must **not** be connected yet.
    parent : QObject, optional
        Parent for QObject ownership.
    """

    # Signals (emitted from the transport thread, delivered on the main thread)
    binary_received = QtCore.Signal(bytes)
    json_received = QtCore.Signal(object)  # dict
    disconnected = QtCore.Signal()
    connected = QtCore.Signal(object)  # ServerConfig
    connection_error = QtCore.Signal(str)

    def __init__(
        self,
        transport: RemoteTransport,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._transport = transport

    @property
    def transport(self) -> RemoteTransport:
        return self._transport

    # -- QThread entry point ------------------------------------------------

    def run(self) -> None:
        """Thread entry — run the transport's asyncio loop."""
        try:
            asyncio.run(self._async_main())
        except Exception:
            logger.exception("TransportThread crashed")

    async def _async_main(self) -> None:
        """Connect, emit ``connected``, then run the receive loop."""
        try:
            config = await self._transport.connect()
        except Exception as exc:
            self.connection_error.emit(str(exc))
            return

        self.connected.emit(config)
        await self._transport.start_receive_loop()

        # Wait for the receive loop to complete (disconnect or error)
        if self._transport._receive_task is not None:
            try:
                await self._transport._receive_task
            except asyncio.CancelledError:
                pass

    # -- public helpers (called from the main thread) -----------------------

    def stop(self) -> None:
        """Request a clean shutdown: disconnect transport, then quit thread."""
        loop = self._transport._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._transport.disconnect(), loop)
        # Wait for the thread to finish (with timeout)
        if not self.wait(5000):
            logger.warning("TransportThread did not stop within 5 s")
            self.terminate()


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class FigureCanvasQTRemote(FigureCanvasRemote, FigureCanvasQT):
    """Qt widget that displays images received from a remote mpl_fastapi server.

    Inherits from both :class:`FigureCanvasRemote` (protocol logic) and
    :class:`FigureCanvasQT` (Qt widget integration).

    Threading
    ---------
    * ``paintEvent()`` / ``draw()`` run on the **main** thread (blit only).
    * Transport IO runs on :class:`TransportThread`.
    * Incoming messages arrive via Qt signals → main-thread slots.
    * Outgoing sends post to the transport's asyncio loop from the main
      thread via ``run_coroutine_threadsafe``.
    """

    manager_class = property(lambda self: FigureManagerQTRemote)  # noqa: ARG005

    def __init__(
        self,
        figure: Figure,
        transport: RemoteTransport,
        server_config: ServerConfig,
    ) -> None:
        _create_qApp()

        # Pre-set attributes that get_width_height() needs, because the MRO
        # chain (FigureCanvasRemote → FigureCanvasQT → QWidget) triggers
        # self.resize(*self.get_width_height()) during __init__.
        self._server_config = server_config
        self._transport = transport
        self._remote_image = None
        self._rubberband_rect = None
        self._last_seq_num = 0

        # Pre-set Qt attributes needed before QWidget.__init__
        self._draw_pending = False
        self._is_drawing = False
        self._draw_rect_callback = lambda painter: None  # noqa: ARG005
        self._in_resize_event = False

        # Pre-converted QPixmap for blitting
        self._remote_qpixmap: QtGui.QPixmap | None = None

        # Thread will be set up by the manager / factory
        self._transport_thread: TransportThread | None = None

        # Now initialise via the MRO.  FigureCanvasRemote.__init__ will call
        # super().__init__(figure), which resolves to FigureCanvasQT.__init__
        # (and thence QWidget.__init__).  Since we pre-set _server_config
        # above, get_width_height() will work.
        FigureCanvasRemote.__init__(self, figure, transport, server_config)

        # Ensure Qt widget attributes are configured
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)

        palette = QtGui.QPalette(QtGui.QColor("white"))
        self.setPalette(palette)

    # -- painting -----------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # noqa: ARG002
        """Blit the most recently received server image to the widget."""
        painter = QtGui.QPainter(self)
        try:
            if self._remote_qpixmap is not None:
                painter.drawPixmap(0, 0, self._remote_qpixmap)
            else:
                # No image yet — fill with the background colour
                painter.fillRect(self.rect(), self.palette().window())

            # Draw rubberband overlay (zoom rectangle from the server)
            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def _update_qpixmap(self) -> None:
        """Convert ``_remote_image`` (PIL) to a QPixmap for painting."""
        img = self._remote_image
        if img is None:
            self._remote_qpixmap = None
            return

        # Ensure RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        data = img.tobytes("raw", "RGBA")
        qimage = QtGui.QImage(
            data, img.width, img.height, QtGui.QImage.Format.Format_RGBA8888
        )
        # Keep a reference to *data* so it isn't GC'd while qimage exists
        qimage._mpl_data_ref = data  # type: ignore[attr-defined]
        qimage.setDevicePixelRatio(self.devicePixelRatioF() or 1)
        self._remote_qpixmap = QtGui.QPixmap.fromImage(qimage)

    # -- FigureCanvasRemote hooks -------------------------------------------

    def schedule_repaint(self) -> None:
        """Post a paint event to the Qt event loop."""
        self._update_qpixmap()
        self.update()

    def draw(self) -> None:
        """Blit-only draw — no server IO."""
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            # Fire the matplotlib draw_event but don't render anything
            self.draw_event(self)
        self.update()

    def draw_idle(self) -> None:
        """Queue a draw via QTimer (same pattern as upstream Qt backend)."""
        if not (
            getattr(self, "_draw_pending", False) or getattr(self, "_is_drawing", False)
        ):
            self._draw_pending = True
            QtCore.QTimer.singleShot(0, self._draw_idle)

    def _draw_idle(self) -> None:
        if not self._draw_pending:
            return
        self._draw_pending = False
        if self.height() <= 0 or self.width() <= 0:
            return
        try:
            self.draw()
        except Exception:
            import traceback

            traceback.print_exc()

    # -- DPR handling -------------------------------------------------------

    def _update_pixel_ratio(self) -> None:
        """Handle a device-pixel-ratio change.

        On Wayland the true (fractional) DPR is not available until the
        window is mapped to a screen.  This override forwards the new
        ratio to the server so it can re-render at the correct
        resolution, then triggers a resize so the canvas geometry is
        updated.
        """
        new_ratio = self.devicePixelRatioF() or 1
        if self._set_device_pixel_ratio(new_ratio):
            # Tell the server about the change
            self._forward_set_device_pixel_ratio(new_ratio)
            # Synthesise a resize so we recalculate figure geometry and
            # tell the server the (unchanged) CSS size at the new DPR.
            event = QtGui.QResizeEvent(self.size(), self.size())
            self.resizeEvent(event)

    # -- resize handling ----------------------------------------------------

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        """Forward resize to the server instead of resizing a local figure."""
        if self._in_resize_event:
            return
        if self.figure is None:
            return

        self._in_resize_event = True
        try:
            w = event.size().width()
            h = event.size().height()
            dpr = self.devicePixelRatioF() or 1

            # Update local figure geometry for coordinate transforms.
            dpival = self.figure.dpi
            self.figure.set_size_inches(
                (w * dpr) / dpival, (h * dpr) / dpival, forward=False
            )

            # Let QWidget handle the resize
            QtWidgets.QWidget.resizeEvent(self, event)

            # Fire local resize event (for mpl_connect callbacks)
            ResizeEvent("resize_event", self)._process()

            # Tell the server about the new size.
            # The server expects CSS (logical) pixels and will multiply
            # by the device_pixel_ratio itself to get physical pixels.
            self._forward_resize(w, h)
        finally:
            self._in_resize_event = False

    def sizeHint(self) -> QtCore.QSize:
        """Return size hint from the server config."""
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    # -- mouse / key event overrides ----------------------------------------
    # The base FigureCanvasQT handlers fire local Matplotlib events.
    # We override them to *also* forward events to the server.

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        x, y = self.mouseEventCoords(event)
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            MouseEvent(
                "button_press_event",
                self,
                x,
                y,
                button,
                modifiers=self._mpl_modifiers(),
                guiEvent=event,
            )._process()
            # Wire protocol uses 0-indexed buttons (JS convention);
            # the server adds +1 to get matplotlib MouseButton values.
            self._forward_mouse_event("button_press", x, y, button=int(button) - 1)

    def mouseDoubleClickEvent(self, event: QtGui.QMouseEvent) -> None:
        x, y = self.mouseEventCoords(event)
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            MouseEvent(
                "button_press_event",
                self,
                x,
                y,
                button,
                dblclick=True,
                modifiers=self._mpl_modifiers(),
                guiEvent=event,
            )._process()
            self._forward_mouse_event("button_press", x, y, button=int(button) - 1)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self.figure is None:
            return
        x, y = self.mouseEventCoords(event)
        MouseEvent(
            "motion_notify_event",
            self,
            x,
            y,
            buttons=self._mpl_buttons(event.buttons()),
            modifiers=self._mpl_modifiers(),
            guiEvent=event,
        )._process()
        self._forward_mouse_event("motion_notify", x, y)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        button = self.buttond.get(event.button())
        if button is not None and self.figure is not None:
            x, y = self.mouseEventCoords(event)
            MouseEvent(
                "button_release_event",
                self,
                x,
                y,
                button,
                modifiers=self._mpl_modifiers(),
                guiEvent=event,
            )._process()
            self._forward_mouse_event("button_release", x, y, button=int(button) - 1)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if (
            event.pixelDelta().isNull()
            or QtWidgets.QApplication.instance().platformName() == "xcb"
        ):
            steps = event.angleDelta().y() / 120
        else:
            steps = event.pixelDelta().y()
        if steps and self.figure is not None:
            x, y = self.mouseEventCoords(event)
            MouseEvent(
                "scroll_event",
                self,
                x,
                y,
                step=steps,
                modifiers=self._mpl_modifiers(),
                guiEvent=event,
            )._process()
            self._forward_mouse_event("scroll", x, y, step=steps)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = self._get_key(event)
        if key is not None and self.figure is not None:
            KeyEvent(
                "key_press_event",
                self,
                key,
                *self.mouseEventCoords(),
                guiEvent=event,
            )._process()
            self._forward_key_event("key_press", key)

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        key = self._get_key(event)
        if key is not None and self.figure is not None:
            KeyEvent(
                "key_release_event",
                self,
                key,
                *self.mouseEventCoords(),
                guiEvent=event,
            )._process()
            self._forward_key_event("key_release", key)

    # -- cleanup ------------------------------------------------------------

    def close_event(self) -> None:
        """Handle widget close."""
        if self._transport_thread is not None:
            self._transport_thread.stop()
            self._transport_thread = None


# ---------------------------------------------------------------------------
# Navigation toolbar
# ---------------------------------------------------------------------------


class NavigationToolbar2QTRemote(NavigationToolbar2QT):
    """Qt toolbar that sends navigation commands to the remote server.

    Button presses (pan, zoom, home, …) are forwarded to the server via
    the canvas's transport.  UI state (message, history buttons, toggle
    states) is driven by server push messages.
    """

    # Use the remote toolbar's items (no Subplots / Customize)
    toolitems = RemoteNavigationToolbar2.toolitems

    def __init__(
        self,
        canvas: FigureCanvasQTRemote,
        parent: QtWidgets.QWidget | None = None,
        coordinates: bool = True,
    ) -> None:
        # NavigationToolbar2QT.__init__ sets up the QToolBar and calls
        # NavigationToolbar2.__init__ which calls set_history_buttons
        self._initializing = True
        super().__init__(canvas, parent, coordinates)
        self._initializing = False

    # -- toolbar actions → server -------------------------------------------

    def home(self, *args: Any) -> None:  # noqa: ARG002
        self.canvas._forward_toolbar_button("home")

    def back(self, *args: Any) -> None:  # noqa: ARG002
        self.canvas._forward_toolbar_button("back")

    def forward(self, *args: Any) -> None:  # noqa: ARG002
        self.canvas._forward_toolbar_button("forward")

    def pan(self, *args: Any) -> None:  # noqa: ARG002
        self.canvas._forward_toolbar_button("pan")

    def zoom(self, *args: Any) -> None:  # noqa: ARG002
        self.canvas._forward_toolbar_button("zoom")

    def download(self, *args: Any) -> None:  # noqa: ARG002
        self._save_remote_figure()

    def save_figure(self, *args: Any) -> None:  # noqa: ARG002
        self._save_remote_figure()

    def _save_remote_figure(self) -> None:
        """Prompt for a filename, then ask the server to save.

        The actual file download happens asynchronously in
        :meth:`_on_save_complete` once the server responds.
        """
        from pathlib import Path

        import matplotlib as mpl

        config = self.canvas._server_config
        save_formats = config.save_formats
        default_format = config.default_save_format

        startpath = Path(mpl.rcParams["savefig.directory"]).expanduser()
        start = str(startpath / f"figure.{default_format}")

        # Build filter string
        filters = []
        selected_filter = None
        for fmt in save_formats:
            filt = f"{fmt.upper()} files (*.{fmt})"
            if fmt == default_format:
                selected_filter = filt
            filters.append(filt)
        filter_str = ";;".join(filters)

        fname, _chosen = QtWidgets.QFileDialog.getSaveFileName(
            self.canvas.parent(),
            "Choose a filename to save to",
            start,
            filter_str,
            selected_filter or "",
        )
        if not fname:
            return

        # Update save directory for next time
        if mpl.rcParams["savefig.directory"]:
            mpl.rcParams["savefig.directory"] = str(Path(fname).parent)

        fmt = Path(fname).suffix.lstrip(".").lower() or default_format

        # Stash the local path so _on_save_complete can finish the job.
        self._pending_save_path = fname

        # Send save request through the normal message flow.
        # The response arrives via canvas._on_json_message → toolbar._on_save_complete.
        self.canvas._forward_save_figure(format=fmt)

    def _on_save_complete(self, msg: dict[str, Any]) -> None:
        """Download the saved file from the server."""
        local_path = getattr(self, "_pending_save_path", None)
        self._pending_save_path = None
        if local_path is None:
            return

        download_url = msg.get("download_url", "")
        if not download_url:
            QtWidgets.QMessageBox.warning(
                self.canvas, "Save error", "No download URL returned."
            )
            return

        self._download_saved_file(download_url, local_path)

    def _on_save_error(self, msg: dict[str, Any]) -> None:
        """Show an error dialog for a failed server save."""
        self._pending_save_path = None
        QtWidgets.QMessageBox.critical(
            self.canvas,
            "Save error",
            msg.get("message", "Unknown server error"),
        )

    def _download_saved_file(self, download_url: str, local_path: str) -> None:
        """Download a saved file from the server via HTTP."""
        import urllib.parse
        import urllib.request

        # Build full URL from the WebSocket URL
        ws_url = self.canvas._transport._url
        parsed = urllib.parse.urlparse(ws_url)
        scheme = "https" if parsed.scheme == "wss" else "http"
        base = f"{scheme}://{parsed.netloc}"
        full_url = urllib.parse.urljoin(base, download_url)

        try:
            urllib.request.urlretrieve(full_url, local_path)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self.canvas, "Download error", str(exc))

    # -- server push state updates ------------------------------------------

    def set_message(self, s: str) -> None:
        """Update status bar text (inherited label from NavigationToolbar2QT)."""
        if hasattr(self, "locLabel"):
            self.locLabel.setText(s)

    def set_history_buttons(self) -> None:
        """No-op during init; later driven by server ``history_buttons``."""
        if getattr(self, "_initializing", True):
            return

    def _on_history_buttons(self, *, back: bool, forward: bool) -> None:
        """Enable/disable back/forward buttons from server message."""
        if "back" in self._actions:
            self._actions["back"].setEnabled(back)
        if "forward" in self._actions:
            self._actions["forward"].setEnabled(forward)

    def _on_navigate_mode(self, mode: str) -> None:
        """Update pan/zoom toggle buttons from server navigate_mode message."""
        if "pan" in self._actions:
            self._actions["pan"].setChecked(mode.lower() == "pan")
        if "zoom" in self._actions:
            self._actions["zoom"].setChecked(mode.lower() == "zoom")

    def draw_rubberband(
        self,
        event: Any,  # noqa: ARG002
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Draw rubberband overlay via the canvas."""
        height = self.canvas.figure.bbox.height
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self) -> None:
        self.canvas.drawRectangle(None)


# ---------------------------------------------------------------------------
# Figure manager
# ---------------------------------------------------------------------------


class FigureManagerQTRemote(FigureManagerQT):
    """Manager that wires a remote canvas to a Qt window with toolbar.

    Extends :class:`FigureManagerQT` by:

    * Using :class:`NavigationToolbar2QTRemote` instead of the standard toolbar.
    * Wiring server push messages (navigate_mode, history_buttons, message,
      rubberband) to the toolbar's Qt widgets.
    * Cleaning up the transport thread on window close.
    """

    canvas: FigureCanvasQTRemote  # type: ignore[assignment]
    toolbar: NavigationToolbar2QTRemote | None  # type: ignore[assignment]

    # Prevent FigureManagerBase.__init__ from creating a default toolbar;
    # we create our own NavigationToolbar2QTRemote below.
    _toolbar2_class = None

    def __init__(self, canvas: FigureCanvasQTRemote, num: int) -> None:
        # Create the main window
        self.window = MainWindow()
        # Call FigureManagerBase.__init__ (skip FigureManagerQT.__init__
        # because we need custom toolbar creation)
        from matplotlib.backend_bases import FigureManagerBase

        FigureManagerBase.__init__(self, canvas, num)
        self.window.closing.connect(self._widgetclosed)

        # Create our custom toolbar
        self.toolbar = NavigationToolbar2QTRemote(canvas, self.window)
        self.window.addToolBar(self.toolbar)
        tbs_height = self.toolbar.sizeHint().height()

        # Size the window to fit canvas + toolbar
        cs = canvas.sizeHint()
        self.window.resize(cs.width(), cs.height() + tbs_height)
        self.window.setCentralWidget(self.canvas)

        # Focus policy
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()

        # Set window title from server config
        if canvas._server_config.figure_label:
            self.window.setWindowTitle(canvas._server_config.figure_label)

    def _widgetclosed(self) -> None:
        """Handle the window close event."""
        CloseEvent("close_event", self.canvas)._process()
        if getattr(self.window, "_destroying", False):
            return
        self.window._destroying = True
        self.destroy()

    def show(self) -> None:
        """Show the window."""
        self.window._destroying = False
        self.window.show()
        self.window.activateWindow()
        self.window.raise_()

    def destroy(self, *args: Any) -> None:
        """Disconnect transport, stop thread, close window."""
        if QtWidgets.QApplication.instance() is None:
            return
        if getattr(self.window, "_destroying", False) and not args:
            # Already destroying — avoid re-entry.
            # (But allow explicit destroy calls to proceed.)
            pass

        self.window._destroying = True

        # Stop the transport thread
        if self.canvas._transport_thread is not None:
            self.canvas._transport_thread.stop()
            self.canvas._transport_thread = None

        if self.toolbar:
            self.toolbar.destroy()
        self.window.close()

    def resize(self, width: int, height: int) -> None:
        """Resize the window (physical pixels → logical pixels).

        Parameters
        ----------
        width, height : int
            Canvas size in physical pixels (matching upstream
            ``FigureManagerBase.resize`` contract).
        """
        dpr = self.canvas.devicePixelRatioF() or 1
        width = int(width / dpr)
        height = int(height / dpr)
        extra_width = self.window.width() - self.canvas.width()
        extra_height = self.window.height() - self.canvas.height()
        self.canvas.resize(width, height)
        self.window.resize(width + extra_width, height + extra_height)

    def get_window_title(self) -> str:
        return self.window.windowTitle()

    def set_window_title(self, title: str) -> None:
        self.window.setWindowTitle(title)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def open_remote_figure(
    url: str,
    plot_name: str,
    init_params: dict[str, Any] | None = None,
    *,
    device_pixel_ratio: float | None = None,
) -> FigureManagerQTRemote:
    """Connect to a remote plot and return a ready-to-show Qt figure manager.

    This is the primary entry point.  It creates a
    :class:`RemoteTransport`, starts a :class:`TransportThread`, waits
    for the handshake to complete, then builds the canvas and manager.

    No ``pyplot``, no ``Gcf``, no ``matplotlib.use()`` — pure OO.

    Parameters
    ----------
    url : str
        Server base URL, e.g. ``"ws://localhost:8000/plots"``.
    plot_name : str
        Name of the plot to connect to.
    init_params : dict, optional
        Query-string parameters for plot initialisation.
    device_pixel_ratio : float, optional
        Device pixel ratio for the init handshake.  If *None* (the
        default), the ratio is auto-detected from the primary screen.

    Returns
    -------
    FigureManagerQTRemote
        Manager with canvas, toolbar, and window ready to ``show()``.

    Raises
    ------
    RuntimeError
        If the connection fails or the handshake times out.
    """
    _create_qApp()

    # Auto-detect device pixel ratio from the primary screen if not given.
    if device_pixel_ratio is None:
        screen = QtWidgets.QApplication.primaryScreen()
        device_pixel_ratio = screen.devicePixelRatio() if screen is not None else 1.0

    ws_url = build_ws_url(url, plot_name, init_params)

    # We'll collect the server config from the connected signal
    result: dict[str, Any] = {}
    error: list[str] = []

    def on_binary(data: bytes) -> None:
        pass  # Will be wired to canvas later

    def on_json(msg: dict[str, Any]) -> None:
        pass  # Will be wired to canvas later

    def on_disconnect() -> None:
        pass  # Will be wired to canvas later

    transport = RemoteTransport(
        ws_url,
        on_binary=on_binary,
        on_json=on_json,
        on_disconnect=on_disconnect,
        device_pixel_ratio=device_pixel_ratio,
    )

    # Start the transport thread and wait for the handshake
    thread = TransportThread(transport)

    event_loop = QtCore.QEventLoop()

    def on_connected(config: ServerConfig) -> None:
        result["config"] = config
        event_loop.quit()

    def on_error(msg: str) -> None:
        error.append(msg)
        event_loop.quit()

    thread.connected.connect(on_connected)
    thread.connection_error.connect(on_error)
    thread.start()

    # Block until connected or error (with timeout)
    QtCore.QTimer.singleShot(10000, event_loop.quit)  # 10s timeout
    event_loop.exec()

    if error:
        thread.stop()
        raise RuntimeError(f"Failed to connect: {error[0]}")
    if "config" not in result:
        thread.stop()
        raise RuntimeError("Connection timed out")

    config = result["config"]

    # FigureCanvasRemote.__init__ overwrites the figure's DPI and size
    # from the server config, so a bare Figure() is fine here.
    figure = Figure()

    # Create the canvas
    canvas = FigureCanvasQTRemote(figure, transport, config)
    canvas._transport_thread = thread

    # Now rewire the transport callbacks to the canvas (on the main thread)
    # We use signals from the TransportThread for thread-safe dispatch.
    thread.binary_received.connect(canvas._on_binary_message)
    thread.json_received.connect(canvas._on_json_message)

    # Also rewire the transport's raw callbacks to emit signals
    transport._on_binary = thread.binary_received.emit
    transport._on_json = thread.json_received.emit
    transport._on_disconnect = thread.disconnected.emit

    # Create the manager (which creates the window + toolbar).
    # Toolbar messages (navigate_mode, history_buttons, message,
    # rubberband) are dispatched by canvas._on_json_message via
    # canvas.toolbar, so no extra wiring is needed.
    manager = FigureManagerQTRemote(canvas, num=-1)

    # Request initial render
    transport.send_json({"type": "refresh"})

    return manager
