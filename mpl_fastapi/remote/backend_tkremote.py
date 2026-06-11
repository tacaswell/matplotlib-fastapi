"""Tkinter thin-client backend for remote Matplotlib rendering.

This module provides the Tk-specific widgets and integration layer for
displaying server-rendered Matplotlib figures.  It builds on the
toolkit-agnostic :mod:`~mpl_fastapi.remote.backend_remote` layer, adding:

* **TransportWorker** — a :class:`threading.Thread` running the asyncio
  transport, with callbacks that marshal incoming messages to the main
  thread via ``widget.after()``.
* **FigureCanvasTkRemote** — a Tk widget that displays the most recently
  received server image and forwards mouse/key/resize events to the server.
* **NavigationToolbar2TkRemote** — a Tk toolbar whose buttons send commands
  to the server.
* **FigureManagerTkRemote** — wires canvas + toolbar + window chrome.
* **open_remote_figure()** — factory function that creates all of the above
  from a server URL and plot name.
* **FigureLauncherWindow** — a discovery + launch UI that queries the server
  for available plots and lets the user open them with custom parameters.
* **open_launcher()** — factory that shows a :class:`FigureLauncherWindow`.

Usage::

    from mpl_fastapi.remote.backend_tkremote import open_remote_figure

    manager = open_remote_figure(
        url="ws://localhost:8000/plots",
        plot_name="sine",
        init_params={"frequency": 2.0},
    )
    manager.show()
    manager.mainloop()
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import tkinter as tk
import tkinter.filedialog
import tkinter.messagebox
import tkinter.ttk as ttk
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
from matplotlib.backend_bases import (
    CloseEvent,
    ResizeEvent,
)
from matplotlib.backends._backend_tk import (  # type: ignore[import-untyped]
    FigureCanvasTk,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from mpl_fastapi.remote.backend_remote import (
    FigureCanvasRemote,
    RemoteNavigationToolbar2,
    RemotePlotInfo,
    _build_watermark_text,
    fetch_watermark,
    list_remote_figures,
    run_transport_lifecycle,
)
from mpl_fastapi.remote.transport import (
    RemoteTransport,
    ServerConfig,
    build_ws_url,
)

__all__ = [
    "FigureCanvasTkRemote",
    "FigureLauncherWindow",
    "FigureManagerTkRemote",
    "NavigationToolbar2TkRemote",
    "TransportWorker",
    "UpdateParametersFrame",
    "open_launcher",
    "open_remote_figure",
    "open_remote_figures",
    "run_tk_app",
]

logger = logging.getLogger(__name__)


def _detect_screen_dpi(root: Any) -> float:
    """Detect the true screen DPI, working around X11/XWayland limitations.

    On X11 and XWayland ``winfo_fpixels('1i')`` is hardcoded to 96 regardless
    of the actual physical display DPI, so it cannot be used to detect HiDPI
    scaling.  This function queries more reliable sources in preference order:

    1. ``GDK_DPI_SCALE`` x ``GDK_SCALE`` environment variables (GNOME/GTK).
    2. ``Xft.dpi`` from ``xrdb`` (set by most desktop environments on login).
    3. ``QT_FONT_DPI`` environment variable (KDE / explicitly set).
    4. The ``tk scaling`` value Tk itself computed (fallback, always available).

    Parameters
    ----------
    root : tk.Tk
        A live root window (needed for the Tk fallback).

    Returns
    -------
    float
        Estimated screen DPI.
    """
    # 1. GDK scale factors (GNOME / Mutter)
    try:
        gdk_scale = float(os.environ.get("GDK_SCALE", "1") or "1")
        gdk_dpi_scale = float(os.environ.get("GDK_DPI_SCALE", "1") or "1")
        combined = gdk_scale * gdk_dpi_scale
        if combined != 1.0:
            return 96.0 * combined
    except ValueError:
        pass

    # 2. Xft.dpi from xrdb (set by GNOME, KDE, etc. at session start)
    try:
        import subprocess

        result = subprocess.run(
            ["xrdb", "-query"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
        for line in result.stdout.splitlines():
            if line.startswith("Xft.dpi:"):
                dpi = float(line.split(":", 1)[1].strip())
                if dpi > 0:
                    return dpi
    except Exception:
        pass

    # 3. QT_FONT_DPI (KDE / explicitly set by user)
    try:
        qt_dpi = float(os.environ.get("QT_FONT_DPI", "0") or "0")
        if qt_dpi > 0:
            return qt_dpi
    except ValueError:
        pass

    # 4. Fall back to what Tk computed: scaling * 72
    try:
        tk_scaling = float(root.tk.call("tk", "scaling"))
        return tk_scaling * 72.0
    except Exception:
        return 96.0


def _apply_dpi_scaling(root: Any) -> None:
    """Apply DPI-aware scaling to a ``tk.Tk`` root window.

    Two things are needed for correct HiDPI rendering:

    1. **tk scaling** — controls the physical pixel size of widgets and
       point-size fonts.  We set it to ``detected_dpi / 72``.

    2. **Scalable font family** — the pixi-bundled Tk defaults to the
       ``fixed`` bitmap font, which renders at a fixed pixel size regardless
       of scaling.  We replace all named fonts with ``liberation sans`` /
       ``liberation mono``, which are outline fonts and scale correctly.

    Parameters
    ----------
    root : tk.Tk
        The root window to configure.  Call before adding any widgets.
    """
    import tkinter.font as tkfont

    try:
        dpi = _detect_screen_dpi(root)
        if dpi > 0:
            root.tk.call("tk", "scaling", dpi / 72.0)
    except Exception:
        pass

    # Switch all named fonts from the non-scalable default ("fixed") to
    # outline fonts that render correctly at any tk scaling factor.
    _PROPORTIONAL = "liberation sans"
    _MONOSPACE = "liberation mono"
    _NAMED_FONTS = {
        "TkDefaultFont": (_PROPORTIONAL, 10),
        "TkTextFont": (_PROPORTIONAL, 10),
        "TkMenuFont": (_PROPORTIONAL, 10),
        "TkHeadingFont": (_PROPORTIONAL, 10),
        "TkCaptionFont": (_PROPORTIONAL, 10),
        "TkSmallCaptionFont": (_PROPORTIONAL, 8),
        "TkIconFont": (_PROPORTIONAL, 10),
        "TkTooltipFont": (_PROPORTIONAL, 10),
        "TkFixedFont": (_MONOSPACE, 10),
    }
    for name, (family, size) in _NAMED_FONTS.items():
        try:
            font = tkfont.nametofont(name)
            font.configure(family=family, size=size)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Transport worker — runs asyncio event loop on a background thread
# ---------------------------------------------------------------------------


class TransportWorker(threading.Thread):
    """Background thread that runs the asyncio transport loop.

    Incoming messages are dispatched to the Tk main thread via
    ``root.after(0, callback)``.  Each callback list can be populated
    after construction by appending to the appropriate ``_*_cbs`` list.

    Parameters
    ----------
    transport : RemoteTransport
        The transport instance to run.  Must **not** be connected yet.
    """

    def __init__(self, transport: RemoteTransport) -> None:
        super().__init__(daemon=True, name="mpl-remote-transport")
        self._transport = transport

        # Tk ``after`` scheduler — set to ``root.after`` once the window
        # exists (after the blocking connection wait in open_remote_figure).
        # Messages that arrive before this is set are dropped (should not
        # happen in normal usage after connection).
        self._schedule: Any = None  # Callable[[int, Callable], str] | None

        # Callback lists — appended to by the canvas after construction.
        self._binary_cbs: list[Any] = []
        self._json_cbs: list[Any] = []
        self._disconnected_cbs: list[Any] = []
        self._reconnecting_cbs: list[Any] = []
        self._reconnected_cbs: list[Any] = []

        # Connection synchronisation — used by open_remote_figure to wait
        # for the handshake before building the UI.
        self._connected_event = threading.Event()
        self._connected_config: ServerConfig | None = None
        self._connection_error: str | None = None

        # Wire transport message callbacks (called from the asyncio thread).
        transport._on_binary = self._handle_binary
        transport._on_json = self._handle_json
        transport._on_disconnect = self._handle_disconnect
        transport._on_reconnect = self._handle_reconnected
        transport._on_reconnecting = self._handle_reconnecting

    @property
    def transport(self) -> RemoteTransport:
        return self._transport

    # -- dispatch helpers ---------------------------------------------------

    def _dispatch(self, fn: Any) -> None:
        """Post *fn* to the Tk main thread."""
        if self._schedule is not None:
            self._schedule(0, fn)

    def _handle_binary(self, data: bytes) -> None:
        self._dispatch(lambda d=data: [cb(d) for cb in list(self._binary_cbs)])

    def _handle_json(self, msg: dict[str, Any]) -> None:
        self._dispatch(lambda m=msg: [cb(m) for cb in list(self._json_cbs)])

    def _handle_disconnect(self) -> None:
        self._dispatch(lambda: [cb() for cb in list(self._disconnected_cbs)])

    def _handle_reconnecting(self, attempt: int, max_attempts: int) -> None:
        self._dispatch(
            lambda a=attempt, m=max_attempts: [
                cb(a, m) for cb in list(self._reconnecting_cbs)
            ]
        )

    def _handle_reconnected(self, config: ServerConfig) -> None:
        self._dispatch(lambda c=config: [cb(c) for cb in list(self._reconnected_cbs)])

    # -- threading.Thread entry point ---------------------------------------

    def run(self) -> None:
        """Thread entry — run the transport's asyncio lifecycle."""
        try:
            asyncio.run(self._async_main())
        except Exception:
            logger.exception("TransportWorker crashed")

    async def _async_main(self) -> None:
        await run_transport_lifecycle(
            self._transport,
            on_connected=self._on_connected,
            on_error=self._on_error,
        )

    def _on_connected(self, config: ServerConfig) -> None:
        self._connected_config = config
        self._connected_event.set()

    def _on_error(self, error: str) -> None:
        self._connection_error = error
        self._connected_event.set()

    # -- public helpers (main thread) ---------------------------------------

    def stop(self) -> None:
        """Request a clean shutdown by disconnecting the transport.

        This method is safe to call from the Tk main thread.  It does NOT
        join the background thread because doing so would block the Tk event
        loop and deadlock the asyncio disconnect coroutine, which needs to
        deliver callbacks back to the main thread via ``after()``.

        The worker thread is a daemon thread and will be cleaned up by the
        OS when the process exits (or when the asyncio disconnect completes).
        """
        loop = self._transport._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._transport.disconnect(), loop)


# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------


class FigureCanvasTkRemote(FigureCanvasRemote, FigureCanvasTk):
    """Tk widget that displays images received from a remote mpl_fastapi server.

    Inherits from both :class:`FigureCanvasRemote` (protocol logic) and
    :class:`FigureCanvasTk` (Tk widget integration).

    Threading
    ---------
    * Painting runs on the **main** thread (blit only).
    * Transport IO runs on :class:`TransportWorker`.
    * Incoming messages arrive via ``root.after()`` → main-thread callbacks.
    * Outgoing sends post to the transport's asyncio loop from the main
      thread via ``asyncio.run_coroutine_threadsafe``.

    Parameters
    ----------
    figure : Figure
        Matplotlib figure (used as a placeholder; all rendering is remote).
    worker : TransportWorker
        The transport worker thread.
    server_config : ServerConfig
        Server configuration from the initial handshake.
    master : tk widget
        Parent window for the canvas widget.
    """

    toolbar: NavigationToolbar2TkRemote  # type: ignore[assignment]

    def __init__(
        self,
        figure: Figure,
        worker: TransportWorker,
        server_config: ServerConfig,
        master: Any,
    ) -> None:
        transport = worker.transport

        # Worker reference and Tk scheduling helper — must be set before
        # any method that might check them (e.g. schedule_repaint).
        self._worker = worker
        self._master = master

        # Tk-specific state.  These must exist before FigureCanvasTk.__init__
        # runs (it may trigger configure/map events that expect them).
        self._photo_ref: Any = None
        self._image_id: int | None = None
        self._overlay_rect_id: int | None = None
        self._overlay_text_id: int | None = None
        self._pending_resize: tuple[int, int] | None = None
        self._resize_after_id: str | None = None
        self._resize_interval_ms: int = 100
        self._draw_pending: bool = False

        # Initialise via the MRO:
        #   FigureCanvasTkRemote
        #     → FigureCanvasRemote.__init__  (sets _server_config, DPI, mpl_connect)
        #         → FigureCanvasTk.__init__(figure, master=master)  (creates _tkcanvas)
        #             → FigureCanvasBase.__init__(figure)
        # FigureCanvasRemote sets all remote protocol state (including _server_config
        # and DPI recovery) *before* calling super(), so FigureCanvasTk's call to
        # get_width_height() inside its __init__ works correctly.
        super().__init__(figure, transport, server_config, master=master)

        # Replace FigureCanvasTk's <Configure> binding with our debounced
        # resize handler so window resizes are rate-limited to the server.
        self._tkcanvas.bind("<Configure>", self._on_configure, add=False)

        # Wire worker callbacks so messages are dispatched on the main thread.
        worker._binary_cbs.append(self._on_binary_message)
        worker._json_cbs.append(self._on_json_message)
        worker._reconnecting_cbs.append(self._show_reconnect_overlay)
        worker._reconnected_cbs.append(self._on_reconnected)
        worker._disconnected_cbs.append(self._show_disconnected_overlay)

    # -- painting (main thread) --------------------------------------------

    def schedule_repaint(self) -> None:
        """Post a blit to the Tk event loop."""
        self._master.after(0, self._blit)

    def _blit(self) -> None:
        """Convert ``_remote_image`` (PIL) to a PhotoImage and display it."""
        img = self._remote_image
        if img is None:
            return

        try:
            from PIL import ImageTk

            if img.mode != "RGBA":
                img = img.convert("RGBA")
            photo = ImageTk.PhotoImage(img)
            # Keep a reference so the image is not garbage-collected.
            self._photo_ref = photo

            if self._image_id is None:
                self._image_id = self._tkcanvas.create_image(
                    0, 0, anchor="nw", image=photo, tags="_mpl_image"
                )
                # Ensure the image is below any overlay items.
                self._tkcanvas.lower("_mpl_image")
            else:
                self._tkcanvas.itemconfigure(self._image_id, image=photo)
        except Exception:
            logger.exception("Error blitting remote image")

    # -- draw / draw_idle --------------------------------------------------

    def draw_idle(self) -> None:
        """Schedule a ``draw()`` on the Tk event loop."""
        if not getattr(self, "_draw_pending", False) and not getattr(
            self, "_is_drawing", False
        ):
            self._draw_pending = True
            self._master.after(0, self._draw_idle_cb)

    def _draw_idle_cb(self) -> None:
        self._draw_pending = False
        if self._tkcanvas.winfo_width() <= 0 or self._tkcanvas.winfo_height() <= 0:
            return
        try:
            self.draw()
        except Exception:
            import traceback

            traceback.print_exc()

    # -- reconnection overlays ---------------------------------------------

    def _show_reconnect_overlay(self, attempt: int, max_attempts: int) -> None:
        """Show a reconnecting message on the canvas."""
        self._draw_overlay(f"Reconnecting\u2026  ({attempt}/{max_attempts})")

    def _hide_reconnect_overlay(self) -> None:
        """Remove the reconnecting overlay."""
        for item_id in (self._overlay_rect_id, self._overlay_text_id):
            if item_id is not None:
                self._tkcanvas.delete(item_id)
        self._overlay_rect_id = None
        self._overlay_text_id = None

    def _show_disconnected_overlay(self) -> None:
        """Show a permanent 'Disconnected' message on the canvas."""
        self._draw_overlay("Disconnected")

    def _draw_overlay(self, text: str) -> None:
        """Draw a centred overlay message on the canvas."""
        self._hide_reconnect_overlay()
        w = self._tkcanvas.winfo_width()
        h = self._tkcanvas.winfo_height()
        cx, cy = w // 2, h // 2
        pad = 12
        self._overlay_text_id = self._tkcanvas.create_text(
            cx, cy, text=text, fill="white", font=("TkDefaultFont", 14)
        )
        bbox = self._tkcanvas.bbox(self._overlay_text_id)
        if bbox:
            x0, y0, x1, y1 = bbox
            self._overlay_rect_id = self._tkcanvas.create_rectangle(
                x0 - pad,
                y0 - pad,
                x1 + pad,
                y1 + pad,
                fill="#000000",
                stipple="gray50",
                outline="",
            )
            # Raise text above background rectangle.
            self._tkcanvas.tag_raise(self._overlay_text_id, self._overlay_rect_id)

    def _on_reconnected(self, config: ServerConfig) -> None:
        """Reset Tk canvas state after a successful reconnection."""
        self._photo_ref = None
        self._image_id = None
        self._hide_reconnect_overlay()

        # Use the actual Tk widget size as the authoritative size.
        w = self._tkcanvas.winfo_width()
        h = self._tkcanvas.winfo_height()
        self._server_config = ServerConfig(
            **{**self._server_config.__dict__, "figure_size": (w, h)}
        )

        # Delegate to the toolkit-agnostic reset.
        super()._on_reconnected(config)

    # -- resize handling ---------------------------------------------------

    def _on_configure(self, event: Any) -> None:
        """Handle Tk <Configure> (resize) events with debouncing."""
        if self.figure is None:
            return
        w, h = event.width, event.height
        if w <= 0 or h <= 0:
            return

        # Update local figure geometry for coordinate transforms.
        dpr = self.device_pixel_ratio
        dpival = self.figure.dpi
        self.figure.set_size_inches(
            (w * dpr) / dpival, (h * dpr) / dpival, forward=False
        )

        # Fire local resize event for mpl_connect callbacks.
        ResizeEvent("resize_event", self)._process()  # type: ignore[attr-defined]

        # Debounce resize forwarding to the server.
        self._pending_resize = (w, h)
        if self._resize_after_id is not None:
            self._master.after_cancel(self._resize_after_id)
        self._resize_after_id = self._master.after(
            self._resize_interval_ms, self._flush_resize
        )

    def _flush_resize(self) -> None:
        """Send the most recent stashed resize to the server."""
        self._resize_after_id = None
        if self._pending_resize is not None:
            w, h = self._pending_resize
            self._pending_resize = None
            self._forward_resize(w, h)

    # -- cleanup -----------------------------------------------------------

    def close_event(self) -> None:
        """Handle canvas close (called by manager on window destroy)."""
        if self._resize_after_id is not None:
            self._master.after_cancel(self._resize_after_id)
            self._resize_after_id = None
        # Detach all worker callbacks BEFORE stopping so the asyncio thread
        # can never call after() on a widget that is already destroyed.
        self._worker._binary_cbs.clear()
        self._worker._json_cbs.clear()
        self._worker._reconnecting_cbs.clear()
        self._worker._reconnected_cbs.clear()
        self._worker._disconnected_cbs.clear()
        self._worker._schedule = None
        self._worker.stop()


# ---------------------------------------------------------------------------
# Navigation toolbar
# ---------------------------------------------------------------------------


class NavigationToolbar2TkRemote(RemoteNavigationToolbar2, NavigationToolbar2Tk):
    """Tk toolbar that sends navigation commands to the remote server.

    Button presses (pan, zoom, home, …) are forwarded to the server via
    the canvas's transport.  UI state (message, history buttons, toggle
    states) is driven by server push messages.

    Inherits from both :class:`RemoteNavigationToolbar2` (protocol logic,
    toolbar-button forwarding) and :class:`NavigationToolbar2Tk` (Tk widget
    integration).  The forwarding methods (``home``, ``back``, ``forward``,
    ``pan``, ``zoom``) come from the remote base; Tk-specific overrides
    (rubberband drawing, status label, save dialogs) are defined here.
    """

    canvas: FigureCanvasTkRemote  # type: ignore[assignment]

    def __init__(
        self,
        canvas: FigureCanvasTkRemote,
        window: Any,
        *,
        pack_toolbar: bool = True,
    ) -> None:
        super().__init__(canvas, window, pack_toolbar=pack_toolbar)

    # -- toolbar actions → server -------------------------------------------
    # home(), back(), forward(), pan(), zoom() are inherited from
    # RemoteNavigationToolbar2 — they forward via canvas._forward_toolbar_button().

    def save_figure(self, *args: Any) -> None:  # noqa: ARG002
        self._save_remote_figure()

    def download(self, *args: Any) -> None:  # noqa: ARG002
        self._save_remote_figure()

    def _save_remote_figure(self) -> None:
        """Prompt for a filename, then ask the server to save.

        The actual file download happens asynchronously in
        :meth:`_on_save_complete` once the server responds.
        """
        config = self.canvas._server_config
        save_formats = config.save_formats
        default_format = config.default_save_format

        startpath = Path(mpl.rcParams["savefig.directory"]).expanduser()
        initial_file = f"figure.{default_format}"

        # Build filetypes list for the dialog.
        filetypes = [(f"{fmt.upper()} files", f"*.{fmt}") for fmt in save_formats]
        filetypes.append(("All files", "*"))

        fname = tkinter.filedialog.asksaveasfilename(
            parent=self.canvas._master,
            title="Choose a filename to save to",
            initialdir=str(startpath),
            initialfile=initial_file,
            filetypes=filetypes,
            defaultextension=f".{default_format}",
        )
        if not fname:
            return

        # Update save directory for next time.
        if mpl.rcParams["savefig.directory"]:
            mpl.rcParams["savefig.directory"] = str(Path(fname).parent)

        fmt = Path(fname).suffix.lstrip(".").lower() or default_format
        self._pending_save_path = fname

        # Send save request; response arrives via canvas._on_json_message
        # → toolbar._on_save_complete.
        self.canvas._forward_save_figure(format=fmt)

    def _on_save_complete(self, msg: dict[str, Any]) -> None:
        """Download the saved file from the server."""
        local_path = getattr(self, "_pending_save_path", None)
        self._pending_save_path = None
        if local_path is None:
            return

        download_url = msg.get("download_url", "")
        if not download_url:
            tkinter.messagebox.showwarning(
                "Save error", "No download URL returned.", parent=self.canvas._master
            )
            return

        try:
            self.canvas._download_url_to_file(download_url, local_path)
        except Exception as exc:
            tkinter.messagebox.showerror(
                "Download error", str(exc), parent=self.canvas._master
            )

    def _on_save_error(self, msg: dict[str, Any]) -> None:
        """Show an error dialog for a failed server save."""
        self._pending_save_path = None
        tkinter.messagebox.showerror(
            "Save error",
            msg.get("message", "Unknown server error"),
            parent=self.canvas._master,
        )

    # -- server push state updates ------------------------------------------

    def set_message(self, s: str) -> None:
        """Update status bar text (inherited label from NavigationToolbar2Tk)."""
        msg_widget = getattr(self, "message", None)
        if msg_widget is not None and hasattr(msg_widget, "set"):
            msg_widget.set(s)

    def set_history_buttons(self) -> None:
        """No-op during init; later driven by server ``history_buttons``."""
        if getattr(self, "_initializing", True):
            return

    def _on_history_buttons(self, *, back: bool, forward: bool) -> None:
        """Enable/disable back/forward buttons from server message."""
        if hasattr(self, "_buttons"):
            if "back" in self._buttons:
                state = tk.NORMAL if back else tk.DISABLED
                self._buttons["back"]["state"] = state
            if "forward" in self._buttons:
                state = tk.NORMAL if forward else tk.DISABLED
                self._buttons["forward"]["state"] = state

    def _on_navigate_mode(self, mode: str) -> None:
        """Update pan/zoom toggle buttons from server navigate_mode message."""
        if not hasattr(self, "_buttons"):
            return
        for btn_name in ("pan", "zoom"):
            btn = self._buttons.get(btn_name)
            if btn is None:
                continue
            if mode.lower() == btn_name:
                btn.config(relief=tk.SUNKEN)
            else:
                btn.config(relief=tk.RAISED)

    def draw_rubberband(
        self,
        event: Any,  # noqa: ARG002
        x0: float,
        y0: float,
        x1: float,
        y1: float,
    ) -> None:
        """Draw rubberband overlay on the canvas.

        Coordinates arrive in matplotlib figure space (physical pixels,
        y from bottom).  Flip y using the Tk canvas height to convert to
        Tk's top-left origin convention.
        """
        height = self.canvas._tkcanvas.winfo_height()
        y0 = height - y0
        y1 = height - y1

        if hasattr(self, "_rubberband_id"):
            self.canvas._tkcanvas.delete(self._rubberband_id)

        self._rubberband_id: int = self.canvas._tkcanvas.create_rectangle(
            x0, y0, x1, y1, outline="black", dash=(4, 4)
        )

    def remove_rubberband(self) -> None:
        """Remove the rubberband rectangle."""
        if hasattr(self, "_rubberband_id"):
            self.canvas._tkcanvas.delete(self._rubberband_id)
            del self._rubberband_id


# ---------------------------------------------------------------------------
# Update parameters widget
# ---------------------------------------------------------------------------


class UpdateParametersFrame(ttk.Frame):
    """Collapsible parameter editor driven by a JSON Schema.

    When the server's ``update_schema`` is non-null, this frame provides
    one input widget per parameter.  Clicking **Update** sends an
    ``update_params`` message to the server, which re-renders the figure.

    Parameters
    ----------
    schema : dict
        JSON Schema for the update parameters.
    canvas : FigureCanvasTkRemote
        Canvas to send update messages through.
    master : tk widget
        Parent widget.
    """

    def __init__(
        self,
        schema: dict[str, Any],
        canvas: FigureCanvasTkRemote,
        master: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, **kwargs)
        self._schema = schema
        self._canvas = canvas
        self._inputs: dict[str, Any] = {}
        self._vars: dict[str, Any] = {}
        self._build_form()

    def _build_form(self) -> None:
        """Build form widgets from the JSON Schema."""
        properties = self._schema.get("properties", {})

        row = 0
        for name, prop in properties.items():
            label_text = prop.get("title", name)
            description = prop.get("description", "")
            if description:
                label_text += f"  ({description})"

            lbl = ttk.Label(self, text=label_text)
            lbl.grid(row=row, column=0, sticky="w", padx=4, pady=2)

            widget, var = self._create_input(name, prop)
            if widget is not None:
                widget.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
                self._inputs[name] = widget
                self._vars[name] = var
                row += 1

        self.columnconfigure(1, weight=1)

        # Buttons row.
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=row, column=0, columnspan=2, sticky="w", pady=4)

        ttk.Button(btn_frame, text="Update", command=self._on_submit).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(btn_frame, text="Reset", command=self._on_reset).pack(
            side=tk.LEFT, padx=4
        )

    def _create_input(self, name: str, prop: dict[str, Any]) -> tuple[Any, Any]:
        """Create an appropriate input widget for a JSON Schema property.

        Returns
        -------
        (widget, variable) or (None, None) for unsupported types.
        """
        prop_type = prop.get("type", "string")
        default = prop.get("default")

        if prop_type in ("number", "integer"):
            var = tk.DoubleVar(value=float(default) if default is not None else 0.0)
            minimum = prop.get("minimum", prop.get("exclusiveMinimum", -1e9))
            maximum = prop.get("maximum", prop.get("exclusiveMaximum", 1e9))
            increment = (
                (float(maximum) - float(minimum)) / 100
                if (minimum is not None and maximum is not None)
                else (0.1 if prop_type == "number" else 1.0)
            )
            decimals = 4 if prop_type == "number" else 0
            widget = ttk.Spinbox(
                self,
                textvariable=var,
                from_=float(minimum),
                to=float(maximum),
                increment=increment,
                format=f"%.{decimals}f",
                width=12,
            )
            return widget, var

        if prop_type == "boolean":
            var = tk.BooleanVar(value=bool(default) if default is not None else False)
            widget = ttk.Checkbutton(self, variable=var)
            return widget, var

        if prop_type == "string":
            enum_values = prop.get("enum")
            if enum_values is not None:
                var = tk.StringVar(
                    value=str(default) if default is not None else str(enum_values[0])
                )
                widget = ttk.Combobox(
                    self,
                    textvariable=var,
                    values=list(map(str, enum_values)),
                    state="readonly",
                )
                return widget, var
            var = tk.StringVar(value=str(default) if default is not None else "")
            widget = ttk.Entry(self, textvariable=var)
            return widget, var

        logger.warning("Unsupported schema type %r for parameter %r", prop_type, name)
        return None, None

    def get_values(self) -> dict[str, Any]:
        """Read current form values and return as a dict."""
        values: dict[str, Any] = {}
        properties = self._schema.get("properties", {})
        for name, var in self._vars.items():
            prop_type = properties[name].get("type", "string")
            raw = var.get()
            if prop_type == "integer":
                values[name] = int(float(raw))
            elif prop_type == "number":
                values[name] = float(raw)
            elif prop_type == "boolean":
                values[name] = bool(raw)
            else:
                values[name] = str(raw)
        return values

    def set_values(self, params: dict[str, Any]) -> None:
        """Programmatically set form values."""
        for name, value in params.items():
            var = self._vars.get(name)
            if var is not None:
                var.set(value)

    def _on_submit(self) -> None:
        """Read form values and send an ``update_params`` message."""
        params = self.get_values()
        logger.debug("Submitting update params: %s", params)
        self._canvas._forward_update_params(params)

    def _on_reset(self) -> None:
        """Reset all inputs to their schema default values."""
        properties = self._schema.get("properties", {})
        for name, var in self._vars.items():
            default = properties[name].get("default")
            if default is not None:
                var.set(default)


# ---------------------------------------------------------------------------
# Figure manager
# ---------------------------------------------------------------------------


class FigureManagerTkRemote:
    """Manager that wires a remote canvas to a Tk window with toolbar.

    Unlike ``FigureManagerBase``, this class directly manages the Tk
    window lifecycle without pyplot integration.

    Parameters
    ----------
    canvas : FigureCanvasTkRemote
        The remote canvas widget.
    num : int
        Figure number (unused, kept for API compatibility).
    """

    def __init__(self, canvas: FigureCanvasTkRemote, num: int) -> None:  # noqa: ARG002
        self.canvas = canvas
        self._window = canvas._master

        # Create toolbar.
        self.toolbar = NavigationToolbar2TkRemote(
            canvas, self._window, pack_toolbar=False
        )
        self.toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Pack the canvas widget.
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Associate toolbar with canvas for message dispatch.
        canvas.toolbar = self.toolbar  # type: ignore[assignment]

        # Create update parameters panel if the server supports it.
        self.update_frame: UpdateParametersFrame | None = None
        if canvas._server_config.update_schema is not None:
            sep = ttk.Separator(self._window, orient=tk.HORIZONTAL)
            sep.pack(side=tk.BOTTOM, fill=tk.X)
            self.update_frame = UpdateParametersFrame(
                canvas._server_config.update_schema, canvas, self._window
            )
            self.update_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=4, pady=4)
            # Pre-populate form with update params from the URL (if any).
            if canvas._server_config.update_params:
                self.update_frame.set_values(canvas._server_config.update_params)

        # Set window title.
        if canvas._server_config.figure_label:
            self._window.title(canvas._server_config.figure_label)

        # Handle window close.
        self._window.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _on_window_close(self) -> None:
        """Handle the window close event."""
        CloseEvent("close_event", self.canvas)._process()  # type: ignore[attr-defined]
        self.destroy()

    def show(self) -> None:
        """Deiconify and raise the window."""
        self._window.deiconify()
        self._window.lift()

    def destroy(self) -> None:
        """Stop transport, clean up, and close the window."""
        if self.update_frame is not None:
            self.update_frame.destroy()
            self.update_frame = None
        self.canvas.close_event()
        try:
            self._window.destroy()
        except tk.TclError:
            pass  # Already destroyed

    def mainloop(self) -> None:
        """Enter the Tk event loop (convenience for single-figure scripts)."""
        self._window.mainloop()

    def set_window_title(self, title: str) -> None:
        self._window.title(title)

    def get_window_title(self) -> str:
        return self._window.title()

    def resize(self, width: int, height: int) -> None:
        """Resize the window.

        Parameters
        ----------
        width, height : int
            Canvas size in CSS (logical) pixels.
        """
        extra_w = self._window.winfo_width() - self.canvas._tkcanvas.winfo_width()
        extra_h = self._window.winfo_height() - self.canvas._tkcanvas.winfo_height()
        self._window.geometry(f"{width + extra_w}x{height + extra_h}")


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def open_remote_figure(
    url: str,
    plot_name: str,
    init_params: dict[str, Any] | None = None,
    *,
    update_params: dict[str, Any] | None = None,
    token: str | None = None,
    device_pixel_ratio: float = 1.0,
    master: Any = None,
) -> FigureManagerTkRemote:
    """Connect to a remote plot and return a ready-to-show Tk figure manager.

    This is the primary entry point.  It creates a
    :class:`RemoteTransport`, starts a :class:`TransportWorker`, waits
    for the handshake to complete (blocking), then builds the canvas and
    manager.

    No ``pyplot``, no ``Gcf``, no ``matplotlib.use()`` — pure OO.

    Parameters
    ----------
    url : str
        Server base URL, e.g. ``"ws://localhost:8000/plots"``.
    plot_name : str
        Name of the plot to connect to.
    init_params : dict, optional
        Query-string parameters for plot initialisation.
    update_params : dict, optional
        Update parameters applied after initialisation.  Encoded as
        ``_update.<key>=<value>`` in the WebSocket query string.
    token : str, optional
        Authentication token appended to the WebSocket query string.
    device_pixel_ratio : float
        Device pixel ratio for the init handshake.  Defaults to 1.0.
    master : tk widget, optional
        Parent Tk window.  If *None* (the default), a new ``tk.Tk()``
        root is created.  Pass an existing ``tk.Tk()`` or ``tk.Toplevel``
        for multi-figure setups.

    Returns
    -------
    FigureManagerTkRemote
        Manager with canvas, toolbar, and window.  Call :meth:`show()` to
        display and :meth:`mainloop()` (on the last figure) to enter the
        Tk event loop.

    Raises
    ------
    RuntimeError
        If the connection fails or the handshake times out.
    """
    ws_url = build_ws_url(
        url, plot_name, init_params, update_params=update_params, token=token
    )

    transport = RemoteTransport(ws_url, device_pixel_ratio=device_pixel_ratio)
    worker = TransportWorker(transport)
    worker.start()

    # Block until connected or error (up to 10 s).
    if not worker._connected_event.wait(timeout=10.0):
        worker.stop()
        raise RuntimeError(f"Connection to {url!r} timed out after 10 s")

    if worker._connection_error is not None:
        worker.stop()
        raise RuntimeError(f"Failed to connect to {url!r}: {worker._connection_error}")

    config = worker._connected_config
    assert config is not None  # satisfied if _connected_event is set without error

    # Create Tk window.
    if master is None:
        window: Any = tk.Tk()
        _apply_dpi_scaling(window)
    else:
        window = master

    # Enable after() dispatch now that the window exists.
    worker._schedule = window.after

    figure = Figure()
    canvas = FigureCanvasTkRemote(figure, worker, config, master=window)
    manager = FigureManagerTkRemote(canvas, num=-1)
    manager.set_window_title(plot_name)

    # Request initial render.
    transport.send_json({"type": "refresh"})

    return manager


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def open_remote_figures(
    specs: Sequence[
        tuple[str, str]
        | tuple[str, str, dict[str, Any] | None]
        | tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]
    ],
    *,
    token: str | None = None,
    device_pixel_ratio: float = 1.0,
) -> list[FigureManagerTkRemote]:
    """Batch-open multiple remote figures.

    A ``tk.Tk()`` root is created for the first figure; subsequent figures
    use ``tk.Toplevel`` windows so they share the same event loop.

    Parameters
    ----------
    specs : list of tuples
        Each element is ``(url, plot_name)``,
        ``(url, plot_name, init_params)``, or
        ``(url, plot_name, init_params, update_params)``.
        *url* is the server base WebSocket URL, *plot_name* is the name
        of the plot, *init_params* is an optional dict of initialisation
        parameters, and *update_params* is an optional dict of update
        parameters (applied after init via ``_update.*`` query params).
    device_pixel_ratio : float
        Passed to :func:`open_remote_figure`.

    Returns
    -------
    list of FigureManagerTkRemote
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
        for mgr in managers:
            mgr.show()
        managers[-1].mainloop()  # or call run_tk_app directly
    """
    # Create the shared Tk root on first use.
    root: Any = tk.Tk()
    _apply_dpi_scaling(root)
    root.withdraw()  # Hidden root — each figure gets its own Toplevel.

    managers: list[FigureManagerTkRemote] = []
    for spec in specs:
        if len(spec) == 2:
            url, plot_name = spec  # type: ignore[misc]
            init_params: dict[str, Any] | None = None
            update_params: dict[str, Any] | None = None
        elif len(spec) == 3:
            url, plot_name, init_params = spec  # type: ignore[misc]
            update_params = None
        else:
            url, plot_name, init_params, update_params = spec  # type: ignore[misc]

        window = tk.Toplevel(root)
        window.withdraw()
        try:
            mgr = open_remote_figure(
                url=url,
                plot_name=plot_name,
                init_params=init_params,
                update_params=update_params,
                token=token,
                device_pixel_ratio=device_pixel_ratio,
                master=window,
            )
            managers.append(mgr)
        except RuntimeError:
            logger.warning("Could not open %r on %s — skipping", plot_name, url)
            window.destroy()

    if not managers and root.winfo_exists():
        root.destroy()

    return managers


def run_tk_app(
    specs: Sequence[
        tuple[str, str]
        | tuple[str, str, dict[str, Any] | None]
        | tuple[str, str, dict[str, Any] | None, dict[str, Any] | None]
    ],
    *,
    token: str | None = None,
    device_pixel_ratio: float = 1.0,
) -> None:
    """Open remote figures and enter the Tk event loop.

    This is the top-level entry point for scripts.  It creates a shared
    ``tk.Tk()`` root (hidden), opens each figure in its own
    ``tk.Toplevel``, shows all windows, and enters ``root.mainloop()``.

    Parameters
    ----------
    specs : list of tuples
        Each element is ``(url, plot_name)``,
        ``(url, plot_name, init_params)``, or
        ``(url, plot_name, init_params, update_params)``.
    device_pixel_ratio : float
        Passed to :func:`open_remote_figure`.

    Examples
    --------
    ::

        from mpl_fastapi.remote.backend_tkremote import run_tk_app

        run_tk_app([
            ("ws://localhost:8000/plots", "sine", {"frequency": 2.0}),
            ("ws://localhost:8000/plots", "cosine"),
        ])
    """
    managers = open_remote_figures(
        specs, token=token, device_pixel_ratio=device_pixel_ratio
    )
    if not managers:
        logger.warning("No figures opened.")
        return

    for mgr in managers:
        mgr.show()

    # Run the Tk event loop via the first window's master root.
    # All Toplevel windows share this root.
    first_window = managers[0]._window
    # Walk up to the Tk root.
    root = first_window
    while isinstance(root, tk.Toplevel):
        root = root.master  # type: ignore[assignment]
    root.mainloop()


# ---------------------------------------------------------------------------
# Schema form helper (used by FigureLauncherWindow)
# ---------------------------------------------------------------------------


class _LauncherFormFrame(ttk.Frame):
    """A read-only form frame that builds widgets from a JSON Schema.

    Unlike :class:`UpdateParametersFrame`, this frame has no submit button
    and no canvas reference.  It is used by :class:`FigureLauncherWindow`
    to display init- and update-parameter forms before a figure is opened.

    Parameters
    ----------
    schema : dict
        JSON Schema with a ``properties`` key.
    master : tk widget
        Parent widget.
    """

    def __init__(self, schema: dict[str, Any], master: Any, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._schema = schema
        self._vars: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        properties = self._schema.get("properties", {})
        for row, (name, prop) in enumerate(properties.items()):
            label_text = prop.get("title", name)
            description = prop.get("description", "")
            if description:
                label_text += f"  ({description})"
            ttk.Label(self, text=label_text).grid(
                row=row, column=0, sticky="w", padx=4, pady=2
            )
            widget, var = self._create_input(self, name, prop)
            if widget is not None:
                widget.grid(row=row, column=1, sticky="ew", padx=4, pady=2)
                self._vars[name] = var
        self.columnconfigure(1, weight=1)

    @staticmethod
    def _create_input(master: Any, name: str, prop: dict[str, Any]) -> tuple[Any, Any]:
        """Return (widget, tk variable) for the given JSON Schema property."""
        prop_type = prop.get("type", "string")
        default = prop.get("default")

        if prop_type in ("number", "integer"):
            var: Any = tk.DoubleVar(
                value=float(default) if default is not None else 0.0
            )
            minimum = prop.get("minimum", prop.get("exclusiveMinimum", -1e9))
            maximum = prop.get("maximum", prop.get("exclusiveMaximum", 1e9))
            span = float(maximum) - float(minimum)
            increment = (
                span / 100 if span > 0 else (0.1 if prop_type == "number" else 1.0)
            )
            decimals = 4 if prop_type == "number" else 0
            widget: Any = ttk.Spinbox(
                master,
                textvariable=var,
                from_=float(minimum),
                to=float(maximum),
                increment=increment,
                format=f"%.{decimals}f",
                width=12,
            )
            return widget, var

        if prop_type == "boolean":
            var = tk.BooleanVar(value=bool(default) if default is not None else False)
            widget = ttk.Checkbutton(master, variable=var)
            return widget, var

        if prop_type == "string":
            enum_values = prop.get("enum")
            if enum_values is not None:
                var = tk.StringVar(
                    value=str(default) if default is not None else str(enum_values[0])
                )
                widget = ttk.Combobox(
                    master,
                    textvariable=var,
                    values=list(map(str, enum_values)),
                    state="readonly",
                )
                return widget, var
            var = tk.StringVar(value=str(default) if default is not None else "")
            widget = ttk.Entry(master, textvariable=var)
            return widget, var

        logger.warning("Unsupported schema type %r for parameter %r", prop_type, name)
        return None, None

    def get_values(self) -> dict[str, Any]:
        """Read current form values and return as a dict."""
        values: dict[str, Any] = {}
        properties = self._schema.get("properties", {})
        for name, var in self._vars.items():
            prop_type = properties[name].get("type", "string")
            raw = var.get()
            if prop_type == "integer":
                values[name] = int(float(raw))
            elif prop_type == "number":
                values[name] = float(raw)
            elif prop_type == "boolean":
                values[name] = bool(raw)
            else:
                values[name] = str(raw)
        return values


# ---------------------------------------------------------------------------
# Figure launcher window
# ---------------------------------------------------------------------------


class FigureLauncherWindow(tk.Toplevel):
    """Launcher window that discovers remote plots and lets the user open them.

    On construction, queries the server's ``/plots`` endpoint in a
    background thread.  The left panel shows a scrollable list of available
    plots; selecting one populates the right panel with init- and
    update-parameter forms.  Clicking **Launch** opens the figure via
    :func:`open_remote_figure`.

    The window stays open after launching figures so the user can open more.

    Parameters
    ----------
    base_url : str
        Server base URL (``ws://`` or ``http://``).
    token : str, optional
        Authentication token.
    master : tk widget, optional
        Parent window.  If *None* a new ``tk.Tk()`` root is created (and
        stored as :attr:`_root`).
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        master: Any = None,
    ) -> None:
        if master is None:
            self._tk_root: Any = tk.Tk()
            _apply_dpi_scaling(self._tk_root)
            self._tk_root.withdraw()
            super().__init__(self._tk_root)
        else:
            self._tk_root = None
            super().__init__(master)

        self._base_url = base_url
        self._token = token
        self._plots: list[RemotePlotInfo] = []
        self._managers: list[FigureManagerTkRemote] = []
        self._init_form: _LauncherFormFrame | None = None
        self._update_form: _LauncherFormFrame | None = None

        self.title("mpl_fastapi \u2014 Figure Launcher")
        self.minsize(700, 450)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._start_discovery()

        # Cache for remote watermark (fetched in background)
        self._remote_versions: dict[str, str] = {}
        self._tk_version = ""
        try:
            self._tk_version = self.tk.eval("info patchlevel")
        except Exception:
            pass
        t = threading.Thread(
            target=self._fetch_watermark_bg, args=(base_url, token), daemon=True
        )
        t.start()

    # -- UI construction ----------------------------------------------------

    def _build_ui(self) -> None:
        """Build the split-pane launcher UI."""
        # --- Menu bar ---
        menu_bar = tk.Menu(self)
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(
            label="Version Info\u2026", command=self._show_version_dialog
        )
        menu_bar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=menu_bar)

        paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, sashwidth=5)
        paned.pack(fill=tk.BOTH, expand=True)

        # --- Left panel: plot list ---
        left = ttk.Frame(paned, padding=4)
        paned.add(left, minsize=200, stretch="never")

        ttk.Label(
            left, text="Available Figures", font=("TkDefaultFont", 10, "bold")
        ).pack(anchor="w", pady=(0, 4))

        list_frame = ttk.Frame(left)
        list_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        self._listbox = tk.Listbox(
            list_frame,
            yscrollcommand=scrollbar.set,
            selectmode=tk.SINGLE,
            activestyle="none",
        )
        scrollbar.config(command=self._listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_list_select)

        self._status_var = tk.StringVar(value="Loading\u2026")
        ttk.Label(left, textvariable=self._status_var, wraplength=180).pack(
            anchor="w", pady=(4, 0)
        )

        # --- Right panel: description + parameter forms + launch button ---
        right_outer = ttk.Frame(paned, padding=4)
        paned.add(right_outer, minsize=400, stretch="always")

        # Scrollable canvas for the right panel content
        right_canvas = tk.Canvas(right_outer, borderwidth=0, highlightthickness=0)
        right_scroll = ttk.Scrollbar(
            right_outer, orient=tk.VERTICAL, command=right_canvas.yview
        )
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        right_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._right_frame = ttk.Frame(right_canvas)
        self._right_frame_id = right_canvas.create_window(
            (0, 0), window=self._right_frame, anchor="nw"
        )

        def _on_configure(event: Any) -> None:
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
            right_canvas.itemconfigure(self._right_frame_id, width=event.width)

        right_canvas.bind("<Configure>", _on_configure)
        self._right_frame.bind(
            "<Configure>",
            lambda _e: right_canvas.configure(scrollregion=right_canvas.bbox("all")),
        )

        # Description label
        self._desc_var = tk.StringVar()
        ttk.Label(
            self._right_frame,
            textvariable=self._desc_var,
            wraplength=380,
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(0, 8))

        # Init params labelframe (hidden until a plot with params is selected)
        self._init_lf = ttk.LabelFrame(
            self._right_frame, text="Init Parameters", padding=4
        )
        self._update_lf = ttk.LabelFrame(
            self._right_frame, text="Update Parameters", padding=4
        )

        # Launch button (always visible, disabled until selection)
        btn_frame = ttk.Frame(self._right_frame)
        btn_frame.pack(side=tk.BOTTOM, anchor="e", pady=8)
        self._launch_btn = ttk.Button(
            btn_frame, text="Launch", command=self._on_launch, state=tk.DISABLED
        )
        self._launch_btn.pack()

    # -- discovery ----------------------------------------------------------

    def _start_discovery(self) -> None:
        """Start background thread to query server for available plots."""
        t = threading.Thread(target=self._discover_worker, daemon=True)
        t.start()

    def _discover_worker(self) -> None:
        """Run in background thread — fetch plot list and post result."""
        try:
            plots = list_remote_figures(self._base_url, token=self._token)
            self.after(0, lambda p=plots: self._on_discovery_finished(p))
        except Exception as exc:
            msg = str(exc)
            self.after(0, lambda m=msg: self._on_discovery_error(m))

    def _on_discovery_finished(self, plots: list[RemotePlotInfo]) -> None:
        self._plots = plots
        self._listbox.delete(0, tk.END)
        for plot in plots:
            self._listbox.insert(tk.END, f"{plot.name}  \u2014  {plot.description}")
        count = len(plots)
        self._status_var.set(f"{count} figure{'s' if count != 1 else ''} available")
        if plots:
            self._listbox.selection_set(0)
            self._listbox.event_generate("<<ListboxSelect>>")

    def _on_discovery_error(self, message: str) -> None:
        self._status_var.set(f"Error: {message}")
        logger.error("Discovery failed: %s", message)

    # -- plot selection -----------------------------------------------------

    def _on_list_select(self, event: Any) -> None:  # noqa: ARG002
        sel = self._listbox.curselection()
        if not sel or sel[0] >= len(self._plots):
            self._launch_btn.configure(state=tk.DISABLED)
            return

        plot = self._plots[sel[0]]
        self._desc_var.set(f"{plot.name}\n{plot.description}")

        # Rebuild init form
        if self._init_form is not None:
            self._init_form.destroy()
            self._init_form = None
        self._init_lf.pack_forget()

        if plot.init_schema and plot.init_schema.get("properties"):
            self._init_form = _LauncherFormFrame(plot.init_schema, self._init_lf)
            self._init_form.pack(fill=tk.X)
            self._init_lf.pack(fill=tk.X, pady=(0, 6))

        # Rebuild update form
        if self._update_form is not None:
            self._update_form.destroy()
            self._update_form = None
        self._update_lf.pack_forget()

        if plot.update_schema and plot.update_schema.get("properties"):
            self._update_form = _LauncherFormFrame(plot.update_schema, self._update_lf)
            self._update_form.pack(fill=tk.X)
            self._update_lf.pack(fill=tk.X, pady=(0, 6))

        self._launch_btn.configure(state=tk.NORMAL)

    # -- launch -------------------------------------------------------------

    def _on_launch(self) -> None:
        sel = self._listbox.curselection()
        if not sel or sel[0] >= len(self._plots):
            return

        plot = self._plots[sel[0]]
        init_params = self._init_form.get_values() if self._init_form else None
        update_params = self._update_form.get_values() if self._update_form else None

        # Derive ws base URL by stripping the /ws/v0/{name} suffix.
        ws_base = plot.ws_url
        suffix = f"/ws/v0/{plot.name}"
        if ws_base.endswith(suffix):
            ws_base = ws_base[: -len(suffix)]

        # Each figure gets its own Toplevel sharing our root.
        window = tk.Toplevel(self)
        window.withdraw()
        try:
            mgr = open_remote_figure(
                url=ws_base,
                plot_name=plot.name,
                init_params=init_params,
                update_params=update_params or None,
                token=self._token,
                master=window,
            )
            mgr.show()
            self._managers.append(mgr)
        except RuntimeError as exc:
            window.destroy()
            tkinter.messagebox.showerror(
                "Connection Failed",
                f"Could not open {plot.name!r}:\n{exc}",
                parent=self,
            )

    # -- cleanup ------------------------------------------------------------

    def _fetch_watermark_bg(
        self,
        base_url: str,
        token: str | None,
    ) -> None:
        """Background thread: fetch watermark and cache it."""
        self._remote_versions = fetch_watermark(base_url, token=token)

    def _show_version_dialog(self) -> None:
        """Open a modal dialog showing local and remote version info."""
        ui_versions = {"Tk": self._tk_version} if self._tk_version else {}
        text = _build_watermark_text(self._remote_versions, ui_versions)
        dlg = tk.Toplevel(self)
        dlg.title("Version Information")
        dlg.transient(self)
        dlg.grab_set()
        dlg.resizable(False, False)

        txt = tk.Text(
            dlg,
            wrap=tk.WORD,
            font=("liberation mono", 10),
            width=70,
            height=12,
            relief=tk.FLAT,
            bg=dlg.cget("bg"),
        )
        txt.insert("1.0", text)
        txt.configure(state=tk.DISABLED)
        txt.pack(padx=12, pady=(12, 4))

        ok_btn = ttk.Button(dlg, text="OK", command=dlg.destroy)
        ok_btn.pack(pady=(4, 12))
        dlg.bind("<Return>", lambda _e: dlg.destroy())
        dlg.bind("<Escape>", lambda _e: dlg.destroy())
        ok_btn.focus_set()

    def _on_close(self) -> None:
        """Handle window close — destroy all launched managers then ourselves."""
        for mgr in self._managers:
            try:
                mgr.destroy()
            except Exception:
                pass
        self._managers.clear()
        self.destroy()
        if self._tk_root is not None:
            self._tk_root.destroy()

    def mainloop(self, n: int = 0) -> None:  # type: ignore[override]
        """Enter the Tk event loop (convenience for standalone use)."""
        if self._tk_root is not None:
            self._tk_root.mainloop(n)
        else:
            super().mainloop()  # type: ignore[call-arg]


def open_launcher(
    base_url: str,
    *,
    token: str | None = None,
) -> FigureLauncherWindow:
    """Create and show a :class:`FigureLauncherWindow`.

    Creates a ``tk.Tk()`` root if there is no existing Tk instance.

    Parameters
    ----------
    base_url : str
        Server base URL (``ws://`` or ``http://``).
    token : str, optional
        Authentication token.

    Returns
    -------
    FigureLauncherWindow
        The launcher window.  Call :meth:`~FigureLauncherWindow.mainloop`
        to enter the Tk event loop.

    Examples
    --------
    ::

        from mpl_fastapi.remote.backend_tkremote import open_launcher

        launcher = open_launcher("ws://localhost:8000/plots")
        launcher.mainloop()
    """
    launcher = FigureLauncherWindow(base_url, token=token)
    launcher.deiconify()
    return launcher


def _main() -> None:
    """Entry point for the ``mpl-fastapi-tk`` console script.

    Also invoked when running the module directly::

        python -m mpl_fastapi.remote.backend_tkremote

    Authentication token resolution order:

    1. ``--token`` CLI argument.
    2. ``MPL_FASTAPI_TOKEN`` environment variable.
    3. No token (unauthenticated).
    """
    from mpl_fastapi.remote._launcher_cli import (
        build_arg_parser,
        parse_kv_pairs,
        resolve_token,
    )

    parser = build_arg_parser("Tk thin-client launcher for remote mpl_fastapi plots")
    args = parser.parse_args()
    token = resolve_token(args.token)
    init_params = parse_kv_pairs(args.params, parser, "Init parameters")
    update_params = parse_kv_pairs(args.update, parser, "Update parameters")

    launcher = open_launcher(args.url, token=token)

    if args.plot:
        try:
            mgr = open_remote_figure(
                url=args.url,
                plot_name=args.plot,
                init_params=init_params or None,
                update_params=update_params or None,
                token=token,
                master=launcher,
            )
            mgr.show()
            launcher._managers.append(mgr)
        except RuntimeError as exc:
            print(f"Warning: could not open {args.plot!r}: {exc}")

    launcher.mainloop()


if __name__ == "__main__":
    _main()
