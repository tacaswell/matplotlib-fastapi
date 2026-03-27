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
import sys
from collections.abc import Sequence
from typing import Any

from matplotlib.backend_bases import (
    CloseEvent,
    ResizeEvent,
)
from matplotlib.backends.backend_qt import (  # type: ignore[import-untyped]
    FigureCanvasQT,
    FigureManagerQT,
    MainWindow,
    NavigationToolbar2QT,
    _create_qApp,
)
from matplotlib.backends.qt_compat import QtCore, QtGui, QtWidgets  # type: ignore[import-untyped]
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
    "UpdateParametersWidget",
    "open_remote_figure",
    "open_remote_figures",
    "run_qt_app",
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
    reconnecting = QtCore.Signal(int, int)  # (attempt, max_attempts)
    reconnected = QtCore.Signal(object)  # ServerConfig
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
        """Connect, emit ``connected``, then run the receive loop.

        After the initial connection, this loops through reconnect
        cycles (receive-loop drop → reconnect → new receive-loop)
        until the transport is explicitly disconnected or reconnection
        is permanently exhausted.
        """
        try:
            config = await self._transport.connect()
        except Exception as exc:
            self.connection_error.emit(str(exc))
            return

        self.connected.emit(config)
        await self._transport.start_receive_loop()

        # Loop through receive/reconnect cycles until done.
        while True:
            # Wait for the current receive loop to finish
            if self._transport._receive_task is not None:
                try:
                    await self._transport._receive_task
                except asyncio.CancelledError:
                    return

            # If no reconnect was started, we're done
            if self._transport._reconnect_task is None:
                return

            # Wait for the reconnect loop to finish
            try:
                await self._transport._reconnect_task
            except asyncio.CancelledError:
                return

            # After reconnect, a new _receive_task should exist.
            # If it doesn't, reconnection failed permanently — exit.
            if self._transport._receive_task is None:
                return

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

    toolbar: NavigationToolbar2QTRemote  # type: ignore[assignment]
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
        self._draw_rect_callback = lambda painter: None  # noqa: ARG005
        self._in_resize_event = False

        # Pre-converted QPixmap for blitting
        self._remote_qpixmap: QtGui.QPixmap | None = None

        # Thread will be set up by the manager / factory
        self._transport_thread: TransportThread | None = None

        # Rate-limit resize events to the server.  Only the final size
        # matters, so we use a pure trailing-edge debounce: stash the
        # latest size and send it after a short quiet period.
        self._resize_interval_ms: int = 100  # ms
        self._pending_resize: tuple[int, int] | None = None
        self._resize_timer: QtCore.QTimer = QtCore.QTimer()
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._flush_resize)

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

        # "Reconnecting…" overlay — a child QLabel shown during reconnect
        self._reconnect_overlay = QtWidgets.QLabel(self)
        self._reconnect_overlay.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._reconnect_overlay.setStyleSheet(
            "background-color: rgba(0, 0, 0, 160);"
            "color: white;"
            "font-size: 16px;"
            "padding: 12px;"
            "border-radius: 8px;"
        )
        self._reconnect_overlay.hide()

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

    # -- reconnection overlay -----------------------------------------------

    def _show_reconnect_overlay(self, attempt: int, max_attempts: int) -> None:
        """Show the reconnecting overlay with attempt info."""
        self._reconnect_overlay.setText(
            f"Reconnecting\u2026  ({attempt}/{max_attempts})"
        )
        self._reconnect_overlay.adjustSize()
        # Centre on the canvas
        ow = self._reconnect_overlay.width()
        oh = self._reconnect_overlay.height()
        self._reconnect_overlay.move(
            (self.width() - ow) // 2,
            (self.height() - oh) // 2,
        )
        self._reconnect_overlay.show()
        self._reconnect_overlay.raise_()

    def _hide_reconnect_overlay(self) -> None:
        """Hide the reconnecting overlay."""
        self._reconnect_overlay.hide()

    def _show_disconnected_overlay(self) -> None:
        """Show a permanent 'Disconnected' overlay."""
        self._reconnect_overlay.setText("Disconnected")
        self._reconnect_overlay.adjustSize()
        ow = self._reconnect_overlay.width()
        oh = self._reconnect_overlay.height()
        self._reconnect_overlay.move(
            (self.width() - ow) // 2,
            (self.height() - oh) // 2,
        )
        self._reconnect_overlay.show()
        self._reconnect_overlay.raise_()

    def _on_reconnected(self, config: ServerConfig) -> None:
        """Reset Qt canvas state after a successful reconnection.

        The server creates a fresh default-sized figure, but the Qt
        window may be a different size.  We stamp the *actual widget
        size* into ``_server_config`` before the base-class handler
        runs, so the resize message it sends carries the real client
        dimensions — not the old ``_server_config.figure_size`` which
        may be stale if the user resized the window while disconnected.
        """
        self._remote_qpixmap = None
        self._hide_reconnect_overlay()

        # Use the actual Qt widget size as the authoritative size.
        # This is in CSS (logical) pixels — the base class sends it
        # as a resize message and the server applies DPR internally.
        w = self.width()
        h = self.height()
        self._server_config = ServerConfig(
            **{**self._server_config.__dict__, "figure_size": (w, h)}
        )

        # Delegate to the toolkit-agnostic reset (clears _remote_image,
        # _rubberband_rect, sends resize + refresh, etc.)
        super()._on_reconnected(config)

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
            ResizeEvent("resize_event", self)._process()  # type: ignore[attr-defined]

            # Debounce the resize to the server — only the final size
            # matters, so we restart a short timer on every resize and
            # send once the user stops dragging.
            self._pending_resize = (w, h)
            self._resize_timer.start(self._resize_interval_ms)
        finally:
            self._in_resize_event = False

    def _flush_resize(self) -> None:
        """Send the most recently stashed resize to the server."""
        if self._pending_resize is not None:
            w, h = self._pending_resize
            self._pending_resize = None
            self._forward_resize(w, h)

    def sizeHint(self) -> QtCore.QSize:
        """Return size hint from the server config."""
        w, h = self.get_width_height()
        return QtCore.QSize(w, h)

    # -- cleanup ------------------------------------------------------------

    def close_event(self) -> None:
        """Handle widget close."""
        self._resize_timer.stop()
        if self._transport_thread is not None:
            self._transport_thread.stop()
            self._transport_thread = None


# ---------------------------------------------------------------------------
# Navigation toolbar
# ---------------------------------------------------------------------------


class NavigationToolbar2QTRemote(RemoteNavigationToolbar2, NavigationToolbar2QT):
    """Qt toolbar that sends navigation commands to the remote server.

    Button presses (pan, zoom, home, …) are forwarded to the server via
    the canvas's transport.  UI state (message, history buttons, toggle
    states) is driven by server push messages.

    Inherits from both :class:`RemoteNavigationToolbar2` (protocol logic,
    toolbar-button forwarding) and :class:`NavigationToolbar2QT` (Qt widget
    integration).  The forwarding methods (``home``, ``back``, ``forward``,
    ``pan``, ``zoom``) come from the remote base; Qt-specific overrides
    (rubberband drawing, status label, save dialogs) are defined here.
    """

    canvas: FigureCanvasQTRemote  # type: ignore[assignment]

    def __init__(
        self,
        canvas: FigureCanvasQTRemote,
        parent: QtWidgets.QWidget | None = None,
        coordinates: bool = True,
    ) -> None:
        super().__init__(canvas, parent=parent, coordinates=coordinates)

    # -- toolbar actions → server -------------------------------------------
    # home(), back(), forward(), pan(), zoom() are inherited from
    # RemoteNavigationToolbar2 — they forward to the server via
    # canvas._forward_toolbar_button().

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
        """Draw rubberband overlay via the canvas.

        The server sends coordinates in matplotlib figure space
        (physical pixels, y from bottom).  Flip y using the widget's
        actual physical-pixel height (matching the JS client's use of
        ``canvas.height``) and pass to ``drawRectangle``.
        """
        dpr = self.canvas.devicePixelRatioF() or 1
        height = self.canvas.height() * dpr
        y1 = height - y1
        y0 = height - y0
        rect = [int(val) for val in (x0, y0, x1 - x0, y1 - y0)]
        self.canvas.drawRectangle(rect)

    def remove_rubberband(self) -> None:
        self.canvas.drawRectangle(None)


# ---------------------------------------------------------------------------
# Update parameters widget
# ---------------------------------------------------------------------------


class UpdateParametersWidget(QtWidgets.QDockWidget):
    """Dockable parameter editor driven by a JSON Schema.

    When the server's ``update_schema`` is non-null, this widget provides
    a form with one input per parameter.  Submitting the form sends an
    ``update_params`` message to the server, which re-renders the figure.

    The widget is auto-generated from the JSON Schema included in the
    server's ``config`` handshake message.

    Parameters
    ----------
    schema : dict
        JSON Schema for the update parameters (from ``ServerConfig.update_schema``).
    canvas : FigureCanvasQTRemote
        Canvas to send update messages through.
    parent : QWidget, optional
        Parent widget.
    """

    # Emitted when the user submits new parameters.
    params_submitted = QtCore.Signal(dict)

    def __init__(
        self,
        schema: dict[str, Any],
        canvas: FigureCanvasQTRemote,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__("Update Parameters", parent)
        self._schema = schema
        self._canvas = canvas
        self._inputs: dict[str, QtWidgets.QWidget] = {}

        self.setAllowedAreas(
            QtCore.Qt.DockWidgetArea.LeftDockWidgetArea
            | QtCore.Qt.DockWidgetArea.RightDockWidgetArea
            | QtCore.Qt.DockWidgetArea.BottomDockWidgetArea
        )
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._build_form()

    # -- form construction --------------------------------------------------

    def _build_form(self) -> None:
        """Build the form widgets from the JSON Schema."""
        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        properties = self._schema["properties"]
        required_fields = set(self._schema.get("required", []))

        for name, prop in properties.items():
            widget = self._create_input_widget(name, prop, name in required_fields)
            if widget is not None:
                label_text = prop.get("title", name)
                description = prop.get("description", "")
                if description:
                    label_text += f"  ({description})"
                layout.addRow(label_text, widget)
                self._inputs[name] = widget

        # Submit button
        button_row = QtWidgets.QHBoxLayout()
        self._submit_button = QtWidgets.QPushButton("Update")
        self._submit_button.clicked.connect(self._on_submit)
        self._reset_button = QtWidgets.QPushButton("Reset")
        self._reset_button.clicked.connect(self._on_reset)
        button_row.addWidget(self._submit_button)
        button_row.addWidget(self._reset_button)
        button_row.addStretch()
        layout.addRow(button_row)

        # Wrap in a scroll area for schemas with many parameters
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        self.setWidget(scroll)

    def _create_input_widget(
        self,
        name: str,
        prop: dict[str, Any],
        required: bool,  # noqa: ARG002
    ) -> QtWidgets.QWidget | None:
        """Create an appropriate input widget for a JSON Schema property.

        Parameters
        ----------
        name : str
            Property name.
        prop : dict
            JSON Schema property descriptor.
        required : bool
            Whether the field is required by the schema.

        Returns
        -------
        QWidget or None
            The input widget, or ``None`` if the type is unsupported.
        """
        prop_type = prop["type"]
        default = prop.get("default")

        if prop_type in ("number", "integer"):
            widget = QtWidgets.QDoubleSpinBox()
            widget.setObjectName(name)
            widget.setDecimals(4 if prop_type == "number" else 0)

            # Range
            minimum = prop.get("minimum", prop.get("exclusiveMinimum"))
            maximum = prop.get("maximum", prop.get("exclusiveMaximum"))
            if minimum is not None:
                widget.setMinimum(float(minimum))
            else:
                widget.setMinimum(-1e9)
            if maximum is not None:
                widget.setMaximum(float(maximum))
            else:
                widget.setMaximum(1e9)

            # Step — pick a sensible default based on range
            if minimum is not None and maximum is not None:
                span = float(maximum) - float(minimum)
                widget.setSingleStep(span / 100)
            else:
                widget.setSingleStep(0.1 if prop_type == "number" else 1)

            if default is not None:
                widget.setValue(float(default))

            return widget

        if prop_type == "boolean":
            widget = QtWidgets.QCheckBox()
            widget.setObjectName(name)
            if default is not None:
                widget.setChecked(bool(default))
            return widget

        if prop_type == "string":
            # Handle enums as combo boxes
            enum_values = prop.get("enum")
            if enum_values is not None:
                widget = QtWidgets.QComboBox()
                widget.setObjectName(name)
                for val in enum_values:
                    widget.addItem(str(val))
                if default is not None:
                    idx = widget.findText(str(default))
                    if idx >= 0:
                        widget.setCurrentIndex(idx)
                return widget
            # Plain string → line edit
            widget = QtWidgets.QLineEdit()
            widget.setObjectName(name)
            if default is not None:
                widget.setText(str(default))
            return widget

        logger.warning("Unsupported schema type %r for parameter %r", prop_type, name)
        return None

    # -- value extraction ---------------------------------------------------

    def get_values(self) -> dict[str, Any]:
        """Read current form values and return as a dict.

        Returns
        -------
        dict
            Parameter name → value, with types matching the schema.
        """
        values: dict[str, Any] = {}
        properties = self._schema["properties"]
        for name, widget in self._inputs.items():
            prop_type = properties[name]["type"]
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                val = widget.value()
                if prop_type == "integer":
                    val = int(val)
                values[name] = val
            elif isinstance(widget, QtWidgets.QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QtWidgets.QComboBox):
                values[name] = widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                values[name] = widget.text()
        return values

    def set_values(self, params: dict[str, Any]) -> None:
        """Programmatically set form values.

        Parameters
        ----------
        params : dict
            Parameter name → value.  Unknown names are silently ignored.
        """
        for name, value in params.items():
            widget = self._inputs.get(name)
            if widget is None:
                continue
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QtWidgets.QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QtWidgets.QComboBox):
                idx = widget.findText(str(value))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.setText(str(value))

    # -- slots --------------------------------------------------------------

    def _on_submit(self) -> None:
        """Read form values and send an ``update_params`` message."""
        params = self.get_values()
        logger.debug("Submitting update params: %s", params)
        self._canvas._forward_update_params(params)
        self.params_submitted.emit(params)

    def _on_reset(self) -> None:
        """Reset all inputs to their schema default values."""
        properties = self._schema["properties"]
        defaults = {
            name: prop["default"]
            for name, prop in properties.items()
            if "default" in prop
        }
        self.set_values(defaults)


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
    toolbar: NavigationToolbar2QTRemote  # type: ignore[assignment]

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
        self.toolbar = NavigationToolbar2QTRemote(canvas, self.window)  # type: ignore[assignment]
        self.window.addToolBar(self.toolbar)
        if self.toolbar is not None:
            tbs_height = self.toolbar.sizeHint().height()
        else:
            tbs_height = 0

        # Size the window to fit canvas + toolbar
        cs = canvas.sizeHint()
        self.window.resize(cs.width(), cs.height() + tbs_height)
        self.window.setCentralWidget(self.canvas)

        # Create update parameters dock widget (if the server supports it)
        self.update_widget: UpdateParametersWidget | None = None
        if canvas._server_config.update_schema is not None:
            self.update_widget = UpdateParametersWidget(
                canvas._server_config.update_schema,
                canvas,
                self.window,
            )
            self.window.addDockWidget(
                QtCore.Qt.DockWidgetArea.BottomDockWidgetArea,
                self.update_widget,
            )

        # Focus policy
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.canvas.setFocus()

        # Set window title from server config
        if canvas._server_config.figure_label:
            self.window.setWindowTitle(canvas._server_config.figure_label)

    def _widgetclosed(self) -> None:
        """Handle the window close event."""
        CloseEvent("close_event", self.canvas)._process()  # type: ignore[attr-defined]
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

        if self.update_widget is not None:
            self.update_widget.close()
            self.update_widget = None
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
        on_reconnect=lambda config: None,   # noqa: ARG005
        on_reconnecting=lambda a, m: None,  # noqa: ARG005
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

    # Reconnection signals → canvas overlay and state reset
    thread.reconnecting.connect(canvas._show_reconnect_overlay)
    thread.reconnected.connect(canvas._on_reconnected)
    thread.disconnected.connect(canvas._show_disconnected_overlay)

    # Also rewire the transport's raw callbacks to emit signals
    transport._on_binary = thread.binary_received.emit
    transport._on_json = thread.json_received.emit
    transport._on_disconnect = thread.disconnected.emit
    transport._on_reconnect = thread.reconnected.emit
    transport._on_reconnecting = thread.reconnecting.emit

    # Create the manager (which creates the window + toolbar).
    # Toolbar messages (navigate_mode, history_buttons, message,
    # rubberband) are dispatched by canvas._on_json_message via
    # canvas.toolbar, so no extra wiring is needed.
    manager = FigureManagerQTRemote(canvas, num=-1)
    manager.set_window_title(plot_name)

    # Request initial render
    transport.send_json({"type": "refresh"})

    return manager


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def open_remote_figures(
    specs: Sequence[tuple[str, str] | tuple[str, str, dict[str, Any] | None]],
    *,
    device_pixel_ratio: float | None = None,
) -> list[FigureManagerQTRemote]:
    """Batch-open multiple remote figures.

    This is a thin wrapper around :func:`open_remote_figure` that opens
    several plots in one call.  Figures may be on different servers.

    Parameters
    ----------
    specs : list of tuples
        Each element is either ``(url, plot_name)`` or
        ``(url, plot_name, init_params)``.  *url* is the server base
        WebSocket URL, *plot_name* is the name of the plot, and
        *init_params* is an optional dict of initialisation parameters.
    device_pixel_ratio : float, optional
        Passed to :func:`open_remote_figure`.  If *None*, auto-detected.

    Returns
    -------
    list of FigureManagerQTRemote
        One manager per successfully opened figure.  Figures that fail
        to connect are skipped with a warning.

    Examples
    --------
    ::

        managers = open_remote_figures([
            ("ws://localhost:8000/plots", "sine", {"frequency": 2.0}),
            ("ws://localhost:8000/plots", "cosine"),
            ("ws://other-host:9000/plots", "heatmap"),
        ])
        run_qt_app(managers)
    """
    managers: list[FigureManagerQTRemote] = []
    for spec in specs:
        if len(spec) == 2:
            url, plot_name = spec  # type: ignore[misc]
            init_params: dict[str, Any] | None = None
        else:
            url, plot_name, init_params = spec  # type: ignore[misc]

        try:
            mgr = open_remote_figure(
                url=url,
                plot_name=plot_name,
                init_params=init_params,
                device_pixel_ratio=device_pixel_ratio,
            )
            managers.append(mgr)
        except RuntimeError:
            logger.warning("Could not open %r on %s — skipping", plot_name, url)
    return managers


def run_qt_app(
    managers: list[FigureManagerQTRemote] | None = None,
) -> None:
    """Show figures and enter the Qt event loop.

    A convenience helper for scripts that only need to display remote
    plots.  Creates a :class:`QApplication` if one does not already
    exist, calls ``manager.show()`` on every manager, and enters
    ``app.exec()``.

    Parameters
    ----------
    managers : list of FigureManagerQTRemote, optional
        Managers to show.  If *None* or empty, exits immediately with
        a message.

    Examples
    --------
    ::

        from mpl_fastapi.remote.backend_qtremote import (
            open_remote_figure,
            run_qt_app,
        )

        mgr = open_remote_figure(
            "ws://localhost:8000/plots", "sine",
        )
        run_qt_app([mgr])
    """
    if not managers:
        logger.warning("No figures to show.")
        return

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    for mgr in managers:
        mgr.show()

    app.exec()
