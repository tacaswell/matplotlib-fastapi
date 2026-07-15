"""Tests for the Qt remote backend (backend_qtremote).

This module tests the Qt-specific integration layer, including:

* :class:`TransportThread` — QThread lifecycle and signal emission
* :class:`FigureCanvasQTRemote` — painting, resize, event forwarding
* :class:`NavigationToolbar2QTRemote` — button actions and server state
* :class:`FigureManagerQTRemote` — manager wiring and cleanup
* :func:`open_remote_figure` — end-to-end factory function
"""

from __future__ import annotations

import asyncio
import io
import struct
import threading
import time
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image
from pydantic import BaseModel, Field
from PySide6 import QtWidgets
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
)

from mpl_fastapi import (
    InitConfig,
    PlotConfig,
    UpdateConfig,
    create_mpl_router,
    install_mpl_router,
)
from mpl_fastapi.remote.backend_qtremote import (
    FigureCanvasQTRemote,
    FigureManagerQTRemote,
    NavigationToolbar2QTRemote,
    TransportThread,
    UpdateParametersWidget,
    open_remote_figure,
    open_remote_figures,
    run_qt_app,
)
from mpl_fastapi.remote.transport import RemoteTransport, ServerConfig
from mpl_fastapi.ws_client import ImageFormat, ImageTypeMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(**overrides: Any) -> ServerConfig:
    """Create a ServerConfig with sensible defaults."""
    defaults: dict[str, Any] = {
        "connection_id": "test-conn-1",
        "protocol_version": 1,
        "figure_size": (640, 480),
        "figure_dpi": 100.0,
        "figure_label": "Test Figure",
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)


def _make_mock_transport(
    server_config: ServerConfig | None = None,
    device_pixel_ratio: float = 1.0,
) -> MagicMock:
    """Create a mock RemoteTransport with standard attributes."""
    transport = MagicMock(spec=RemoteTransport)
    transport._loop = MagicMock()
    transport._ws = MagicMock()
    transport.server_config = server_config or _make_server_config()
    transport.is_connected = True
    transport._device_pixel_ratio = device_pixel_ratio
    return transport


def _make_test_image(
    width: int = 640,
    height: int = 480,
    color: tuple[int, ...] = (255, 0, 0, 255),
) -> bytes:
    """Create a test PNG image with a binary header prepended."""
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_data = buf.getvalue()

    # 8-byte header: type_mode(1) + format(1) + seq(2) + base_seq(2) + reserved(2)
    header = struct.pack(
        "<BBHHH",
        ImageTypeMode.FULL,  # type_mode
        ImageFormat.PNG,  # format
        1,  # seq_num
        0,  # base_seq
        0,  # reserved
    )
    return header + png_data


# ---------------------------------------------------------------------------
# Test server setup (module-scoped, shared across integration tests)
# ---------------------------------------------------------------------------


class SimpleParams(BaseModel):
    value: float = Field(default=1.0, ge=0.1, le=10.0)


class UpdateParams(BaseModel):
    """Update parameters for testing."""

    phase: float = Field(default=0.0, ge=0.0, le=6.28, description="Phase shift")


def _create_simple_figure(fig: Figure, params: SimpleParams) -> dict[str, object]:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 100)
    (line,) = ax.plot(x, np.sin(x * params.value))
    ax.set_title(f"Simple (value={params.value})")
    return {"ax": ax, "line": line, "x": x, "value": params.value}


def _update_simple_figure(
    state: dict[str, object], params: UpdateParams
) -> dict[str, object]:
    x = state["x"]
    value = state["value"]
    y = np.sin(np.asarray(x) * float(value) + params.phase)  # type: ignore[arg-type]
    cast(Line2D, state["line"]).set_ydata(y)
    return state


def _make_test_app() -> FastAPI:
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "simple": PlotConfig(
                description="test",
                init=InitConfig(
                    function=_create_simple_figure,
                    params_model=SimpleParams,
                ),
            ),
            "updatable": PlotConfig(
                description="updatable test plot",
                init=InitConfig(
                    function=_create_simple_figure,
                    params_model=SimpleParams,
                ),
                update=UpdateConfig(
                    function=_update_simple_figure,
                    params_model=UpdateParams,
                ),
            ),
        }
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


@pytest.fixture(scope="module")
def server_url() -> str:
    """Start a real mpl_fastapi server on a random port."""
    app = _make_test_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start in time")

    sockets = server.servers[0].sockets
    actual_port = sockets[0].getsockname()[1]
    return f"ws://127.0.0.1:{actual_port}/plots"


# ---------------------------------------------------------------------------
# Unit tests: FigureCanvasQTRemote (with mocked transport)
# ---------------------------------------------------------------------------


class TestFigureCanvasQTRemoteInit:
    def test_constructor_creates_widget(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        assert canvas.figure is fig
        assert canvas._remote_image is None
        assert canvas._remote_qpixmap is None

    def test_size_hint_from_config(self, qtbot: Any) -> None:
        config = _make_server_config(figure_size=(800, 600))
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(8, 6), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        assert canvas.sizeHint() == QSize(800, 600)


class TestFigureCanvasQTRemotePainting:
    def test_paint_with_no_image(self, qtbot: Any) -> None:
        """paintEvent should not crash when no image has been received."""
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        # Force a repaint — should not raise
        canvas.repaint()

    def test_paint_with_image(self, qtbot: Any) -> None:
        """paintEvent should render the received image."""
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        # Feed a test image
        canvas._on_binary_message(_make_test_image())
        assert canvas._remote_image is not None
        assert canvas._remote_qpixmap is not None

        # Force repaint — should not raise
        canvas.repaint()

    def test_schedule_repaint_updates_pixmap(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        canvas._on_binary_message(_make_test_image(color=(0, 255, 0, 255)))
        assert canvas._remote_qpixmap is not None


class TestFigureCanvasQTRemoteEvents:
    def test_mouse_press_forwards_to_server(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        # Simulate a click
        qtbot.mouseClick(canvas, Qt.MouseButton.LeftButton)

        # Verify transport.send_json was called with a button_press message
        calls = transport.send_json.call_args_list
        button_presses = [c for c in calls if c[0][0].get("type") == "button_press"]
        # We expect both press and release
        assert len(button_presses) >= 1, f"Expected button_press, got {calls}"
        # Wire protocol uses 0-indexed buttons (JS convention);
        # the server adds +1 to get matplotlib MouseButton values.
        # Left click → 0 on the wire.
        assert button_presses[0][0][0]["button"] == 0

    def test_key_press_forwards_to_server(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        qtbot.keyPress(canvas, Qt.Key.Key_A)

        calls = transport.send_json.call_args_list
        key_presses = [c for c in calls if c[0][0].get("type") == "key_press"]
        assert len(key_presses) >= 1, f"Expected key_press, got {calls}"

    def test_resize_forwards_to_server(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        canvas.resize(400, 300)
        QApplication.processEvents()

        # Resize is debounced — wait for the trailing-edge timer to fire.
        qtbot.waitUntil(
            lambda: any(
                c[0][0].get("type") == "resize"
                for c in transport.send_json.call_args_list
            ),
            timeout=1000,
        )

        calls = transport.send_json.call_args_list
        resizes = [c for c in calls if c[0][0].get("type") == "resize"]
        assert len(resizes) >= 1, f"Expected resize, got {calls}"

    def test_dpr_change_forwards_to_server(self, qtbot: Any) -> None:
        """Simulate a DPR change (e.g. Wayland screen mapping)."""
        config = _make_server_config(figure_dpi=100.0)
        transport = _make_mock_transport(config, device_pixel_ratio=1.0)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        # Before showing: the canvas has the handshake DPR (1.0).
        # ``Figure._original_dpi`` is a private matplotlib attribute (set
        # internally for HiDPI bookkeeping) that is absent from the stubs.
        assert canvas.device_pixel_ratio == 1.0
        assert fig._original_dpi == 100.0  # type: ignore[attr-defined]

        # Show the canvas.  On the test machine (DPR=2), showEvent fires
        # _update_pixel_ratio() which detects the real screen DPR and
        # sends set_device_pixel_ratio to the server automatically.
        transport.send_json.reset_mock()
        canvas.show()
        qtbot.waitExposed(canvas)
        QApplication.processEvents()

        screen_dpr = canvas.devicePixelRatioF() or 1
        if screen_dpr != 1.0:
            # The DPR changed on show — verify the message was sent
            assert canvas.device_pixel_ratio == screen_dpr
            assert fig.dpi == 100.0 * screen_dpr
            assert fig._original_dpi == 100.0  # type: ignore[attr-defined]
            calls = transport.send_json.call_args_list
            dpr_msgs = [
                c for c in calls if c[0][0].get("type") == "set_device_pixel_ratio"
            ]
            assert len(dpr_msgs) >= 1
            assert dpr_msgs[0][0][0]["device_pixel_ratio"] == screen_dpr
            assert transport._device_pixel_ratio == screen_dpr

        # Now simulate an explicit DPR change via _update_pixel_ratio
        # (e.g. dragging window to a different monitor)
        transport.send_json.reset_mock()
        from unittest.mock import patch

        target_dpr = 3.0  # Different from any current value
        with patch.object(canvas, "devicePixelRatioF", return_value=target_dpr):
            canvas._update_pixel_ratio()

        assert canvas.device_pixel_ratio == target_dpr
        assert fig.dpi == 100.0 * target_dpr
        assert fig._original_dpi == 100.0  # type: ignore[attr-defined]

        calls = transport.send_json.call_args_list
        dpr_msgs = [c for c in calls if c[0][0].get("type") == "set_device_pixel_ratio"]
        assert len(dpr_msgs) == 1
        assert dpr_msgs[0][0][0]["device_pixel_ratio"] == target_dpr
        assert transport._device_pixel_ratio == target_dpr

        # A resize should also have been sent (debounced — wait for timer)
        qtbot.waitUntil(
            lambda: any(
                c[0][0].get("type") == "resize"
                for c in transport.send_json.call_args_list
            ),
            timeout=1000,
        )
        calls = transport.send_json.call_args_list
        resizes = [c for c in calls if c[0][0].get("type") == "resize"]
        assert len(resizes) >= 1


# ---------------------------------------------------------------------------
# Unit tests: NavigationToolbar2QTRemote
# ---------------------------------------------------------------------------


class TestNavigationToolbar2QTRemote:
    def test_toolbar_creates_buttons(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        # Should have the standard navigation actions
        assert "home" in toolbar._actions
        assert "pan" in toolbar._actions
        assert "zoom" in toolbar._actions
        assert "back" in toolbar._actions
        assert "forward" in toolbar._actions
        assert "download" in toolbar._actions

    def test_home_sends_toolbar_button(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar.home()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "home"}
        )

    def test_pan_sends_toolbar_button(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar.pan()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "pan"}
        )

    def test_zoom_sends_toolbar_button(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar.zoom()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "zoom"}
        )

    def test_set_message_updates_label(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar.set_message("x=1.5 y=2.3")
        assert toolbar.locLabel.text() == "x=1.5 y=2.3"

    def test_history_buttons_enables_actions(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar._on_history_buttons(back=True, forward=False)
        assert toolbar._actions["back"].isEnabled()
        assert not toolbar._actions["forward"].isEnabled()

        toolbar._on_history_buttons(back=False, forward=True)
        assert not toolbar._actions["back"].isEnabled()
        assert toolbar._actions["forward"].isEnabled()

    def test_navigate_mode_updates_buttons(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        toolbar = NavigationToolbar2QTRemote(canvas)
        qtbot.addWidget(toolbar)

        toolbar._on_navigate_mode("PAN")
        assert toolbar._actions["pan"].isChecked()
        assert not toolbar._actions["zoom"].isChecked()

        toolbar._on_navigate_mode("ZOOM")
        assert not toolbar._actions["pan"].isChecked()
        assert toolbar._actions["zoom"].isChecked()

        toolbar._on_navigate_mode("")
        assert not toolbar._actions["pan"].isChecked()
        assert not toolbar._actions["zoom"].isChecked()


# ---------------------------------------------------------------------------
# Unit tests: FigureManagerQTRemote
# ---------------------------------------------------------------------------


class TestFigureManagerQTRemote:
    def test_creates_window_and_toolbar(self, qtbot: Any) -> None:
        config = _make_server_config(figure_label="My Plot")
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        assert manager.toolbar is not None
        assert isinstance(manager.toolbar, NavigationToolbar2QTRemote)
        assert manager.window.windowTitle() == "My Plot"

    def test_show_and_destroy(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas._transport_thread = MagicMock()
        mock_thread = canvas._transport_thread

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        manager.show()
        assert manager.window.isVisible()

        manager.destroy()
        # Transport thread should have been stopped
        mock_thread.stop.assert_called_once()

    def test_set_window_title(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        manager.set_window_title("New Title")
        assert manager.get_window_title() == "New Title"


# ---------------------------------------------------------------------------
# Unit tests: TransportThread
# ---------------------------------------------------------------------------


class TestTransportThread:
    def test_connected_signal_emitted(self, qtbot: Any) -> None:  # noqa: ARG002
        """TransportThread emits connected(ServerConfig) on successful handshake."""
        config = _make_server_config()

        # Create a transport with a mock _run that simulates connection
        transport = MagicMock(spec=RemoteTransport)
        transport._loop = None
        transport._receive_task = None

        async def fake_connect() -> ServerConfig:
            transport._loop = asyncio.get_running_loop()
            return config

        async def fake_start_receive_loop() -> None:
            pass

        transport.connect = fake_connect
        transport.start_receive_loop = fake_start_receive_loop

        thread = TransportThread(transport)

        configs_received: list[ServerConfig] = []
        thread.connected.connect(configs_received.append)
        thread.start()
        thread.wait(5000)

        # Process pending events so cross-thread signals are delivered
        QApplication.processEvents()

        assert len(configs_received) == 1
        assert configs_received[0].connection_id == "test-conn-1"

    def test_connection_error_signal(self, qtbot: Any) -> None:  # noqa: ARG002
        """TransportThread emits connection_error on failure."""
        transport = MagicMock(spec=RemoteTransport)
        transport._loop = None
        transport._receive_task = None

        async def fail_connect() -> ServerConfig:
            raise RuntimeError("Connection refused")

        transport.connect = fail_connect

        thread = TransportThread(transport)

        errors: list[str] = []
        thread.connection_error.connect(errors.append)
        thread.start()
        thread.wait(5000)

        # Process pending events so cross-thread signals are delivered
        QApplication.processEvents()

        assert len(errors) == 1
        assert "Connection refused" in errors[0]


# ---------------------------------------------------------------------------
# Integration test: open_remote_figure with real server
# ---------------------------------------------------------------------------


class TestOpenRemoteFigure:
    def test_open_remote_figure_creates_manager(
        self, qtbot: Any, server_url: str
    ) -> None:
        """open_remote_figure connects, creates canvas + manager."""
        manager = open_remote_figure(
            url=server_url,
            plot_name="simple",
            init_params={"value": 1.0},
        )
        qtbot.addWidget(manager.window)

        try:
            assert isinstance(manager, FigureManagerQTRemote)
            assert isinstance(manager.canvas, FigureCanvasQTRemote)
            assert isinstance(manager.toolbar, NavigationToolbar2QTRemote)

            # The canvas should have a transport thread
            assert manager.canvas._transport_thread is not None
            assert manager.canvas._transport_thread.isRunning()

            # Server config should be populated
            config = manager.canvas._server_config
            assert config.connection_id
            assert config.figure_size[0] > 0
            assert config.figure_size[1] > 0

            # Show the window briefly
            manager.show()
            qtbot.waitExposed(manager.window)

            # Wait for an image to arrive
            for _ in range(100):
                QApplication.processEvents()
                if manager.canvas._remote_image is not None:
                    break
                time.sleep(0.05)

            assert manager.canvas._remote_image is not None
        finally:
            manager.destroy()

    def test_toolbar_button_sends_to_server(self, qtbot: Any, server_url: str) -> None:
        """Pressing toolbar buttons sends messages via the transport."""
        manager = open_remote_figure(
            url=server_url,
            plot_name="simple",
            init_params={"value": 1.0},
        )
        qtbot.addWidget(manager.window)

        try:
            manager.show()
            qtbot.waitExposed(manager.window)

            # Wait for initial image
            for _ in range(100):
                QApplication.processEvents()
                if manager.canvas._remote_image is not None:
                    break
                time.sleep(0.05)

            # Click home button — should not raise
            manager.toolbar.home()
            QApplication.processEvents()

            # Click pan — should not raise
            manager.toolbar.pan()
            QApplication.processEvents()

            # Click zoom — should not raise
            manager.toolbar.zoom()
            QApplication.processEvents()

            # Let any pending server responses drain before teardown
            time.sleep(0.2)
            QApplication.processEvents()
        finally:
            manager.destroy()
            QApplication.processEvents()

    def test_open_unknown_plot_raises(
        self,
        qtbot: Any,  # noqa: ARG002
        server_url: str,
    ) -> None:
        """open_remote_figure raises for non-existent plot names."""
        with pytest.raises(RuntimeError, match=r"Failed to connect|timed out"):
            open_remote_figure(
                url=server_url,
                plot_name="nonexistent_plot_xyz",
            )


# ---------------------------------------------------------------------------
# Integration tests: open_remote_figures
# ---------------------------------------------------------------------------


class TestOpenRemoteFigures:
    def test_batch_open_multiple_figures(self, qtbot: Any, server_url: str) -> None:
        """open_remote_figures opens several figures at once."""
        specs = [
            (server_url, "simple", {"value": 1.0}),
            (server_url, "simple", {"value": 2.0}),
        ]
        managers = open_remote_figures(specs)

        try:
            assert len(managers) == 2
            for mgr in managers:
                qtbot.addWidget(mgr.window)
                assert isinstance(mgr, FigureManagerQTRemote)
                assert isinstance(mgr.canvas, FigureCanvasQTRemote)
        finally:
            for mgr in managers:
                mgr.destroy()
            QApplication.processEvents()

    def test_batch_open_two_tuple_form(self, qtbot: Any, server_url: str) -> None:
        """open_remote_figures accepts 2-tuple specs (url, plot_name)."""
        specs = [(server_url, "simple")]
        managers = open_remote_figures(specs)

        try:
            assert len(managers) == 1
            qtbot.addWidget(managers[0].window)
            assert isinstance(managers[0], FigureManagerQTRemote)
        finally:
            for mgr in managers:
                mgr.destroy()
            QApplication.processEvents()

    def test_batch_open_skips_failures(
        self,
        qtbot: Any,
        server_url: str,
    ) -> None:
        """open_remote_figures skips plots that fail to connect."""
        specs = [
            (server_url, "simple", {"value": 1.0}),
            (server_url, "nonexistent_plot_xyz", None),
        ]
        managers = open_remote_figures(specs)

        try:
            # Only the "simple" plot should have succeeded
            assert len(managers) == 1
            qtbot.addWidget(managers[0].window)
        finally:
            for mgr in managers:
                mgr.destroy()
            QApplication.processEvents()

    def test_batch_open_empty_specs(self) -> None:
        """open_remote_figures with empty specs returns empty list."""
        assert open_remote_figures([]) == []


# ---------------------------------------------------------------------------
# Tests: run_qt_app
# ---------------------------------------------------------------------------


class TestRunQtApp:
    def test_run_qt_app_with_no_specs_returns(self) -> None:
        """run_qt_app with empty specs returns immediately."""
        # Should not block or raise
        run_qt_app([])

    def test_run_qt_app_shows_managers(self, qtbot: Any, server_url: str) -> None:
        """run_qt_app calls show() on each manager.

        We can't actually enter app.exec() in a test (it would block),
        so we verify the show + exec machinery indirectly by patching
        app.exec and checking that managers get shown.
        """
        from unittest.mock import patch

        managers_shown: list[FigureManagerQTRemote] = []
        original_show = FigureManagerQTRemote.show

        def tracking_show(self: FigureManagerQTRemote) -> None:
            managers_shown.append(self)
            original_show(self)

        app = QApplication.instance()
        assert app is not None

        with (
            patch.object(app, "exec"),
            patch.object(FigureManagerQTRemote, "show", tracking_show),
        ):
            run_qt_app([(server_url, "simple")])

        assert len(managers_shown) == 1
        mgr = managers_shown[0]
        qtbot.addWidget(mgr.window)
        try:
            assert mgr.window.isVisible()
        finally:
            mgr.destroy()
            QApplication.processEvents()


# ---------------------------------------------------------------------------
# Helper: example update schemas for widget tests
# ---------------------------------------------------------------------------

_SIMPLE_UPDATE_SCHEMA: dict[str, Any] = {
    "properties": {
        "phase": {
            "type": "number",
            "title": "Phase",
            "description": "Phase shift in radians",
            "default": 0.0,
            "minimum": 0.0,
            "maximum": 6.28,
        },
    },
    "required": [],
}

_MULTI_TYPE_UPDATE_SCHEMA: dict[str, Any] = {
    "properties": {
        "frequency": {
            "type": "number",
            "title": "Frequency",
            "default": 1.0,
            "minimum": 0.1,
            "maximum": 10.0,
        },
        "points": {
            "type": "integer",
            "title": "Points",
            "default": 200,
            "minimum": 50,
            "maximum": 1000,
        },
        "show_grid": {
            "type": "boolean",
            "title": "Show Grid",
            "default": True,
        },
        "style": {
            "type": "string",
            "title": "Style",
            "enum": ["solid", "dashed", "dotted"],
            "default": "solid",
        },
        "label": {
            "type": "string",
            "title": "Label",
            "default": "My Plot",
        },
    },
}


# ---------------------------------------------------------------------------
# Unit tests: UpdateParametersWidget
# ---------------------------------------------------------------------------


class TestUpdateParametersWidgetConstruction:
    """Test widget creation from JSON Schema."""

    def test_creates_widget_from_simple_schema(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        assert "phase" in widget._inputs
        assert widget.windowTitle() == "Update Parameters"

    def test_creates_number_spinbox(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        phase_input = widget._inputs["phase"]
        assert isinstance(phase_input, QDoubleSpinBox)
        assert phase_input.value() == 0.0
        assert phase_input.minimum() == 0.0
        assert phase_input.maximum() == 6.28

    def test_creates_integer_spinbox(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_MULTI_TYPE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_MULTI_TYPE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        points_input = widget._inputs["points"]
        assert isinstance(points_input, QDoubleSpinBox)
        assert points_input.value() == 200.0
        assert points_input.decimals() == 0

    def test_creates_boolean_checkbox(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_MULTI_TYPE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_MULTI_TYPE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        grid_input = widget._inputs["show_grid"]
        assert isinstance(grid_input, QCheckBox)
        assert grid_input.isChecked()

    def test_creates_enum_combobox(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_MULTI_TYPE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_MULTI_TYPE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        style_input = widget._inputs["style"]
        assert isinstance(style_input, QComboBox)
        assert style_input.currentText() == "solid"
        assert style_input.count() == 3

    def test_creates_string_line_edit(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_MULTI_TYPE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_MULTI_TYPE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        label_input = widget._inputs["label"]
        assert isinstance(label_input, QLineEdit)
        assert label_input.text() == "My Plot"

    def test_empty_schema_creates_no_inputs(self, qtbot: Any) -> None:
        empty_schema: dict[str, Any] = {"properties": {}}
        config = _make_server_config(update_schema=empty_schema)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(empty_schema, canvas)
        qtbot.addWidget(widget)

        assert len(widget._inputs) == 0


class TestUpdateParametersWidgetValues:
    """Test get_values and set_values."""

    def test_get_values_returns_defaults(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        values = widget.get_values()
        assert "phase" in values
        assert values["phase"] == pytest.approx(0.0)

    def test_get_values_multi_type(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_MULTI_TYPE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_MULTI_TYPE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        values = widget.get_values()
        assert values["frequency"] == pytest.approx(1.0)
        assert values["points"] == 200  # integer type
        assert values["show_grid"] is True
        assert values["style"] == "solid"
        assert values["label"] == "My Plot"

    def test_set_values_updates_widgets(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        widget.set_values({"phase": 3.14})
        assert widget.get_values()["phase"] == pytest.approx(3.14)

    def test_set_values_ignores_unknown_keys(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        # Should not raise
        widget.set_values({"nonexistent": 42})
        assert widget.get_values()["phase"] == pytest.approx(0.0)


class TestUpdateParametersWidgetSubmit:
    """Test form submission."""

    def test_submit_sends_update_params(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        # Set a value and submit
        widget.set_values({"phase": 1.57})
        widget._on_submit()

        transport.send_json.assert_called_with(
            {"type": "update_params", "params": {"phase": pytest.approx(1.57)}}
        )

    def test_submit_emits_signal(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        received: list[dict[str, Any]] = []
        widget.params_submitted.connect(received.append)

        widget.set_values({"phase": 2.0})
        widget._on_submit()

        assert len(received) == 1
        assert received[0]["phase"] == pytest.approx(2.0)

    def test_reset_restores_defaults(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        widget = UpdateParametersWidget(_SIMPLE_UPDATE_SCHEMA, canvas)
        qtbot.addWidget(widget)

        widget.set_values({"phase": 5.0})
        assert widget.get_values()["phase"] == pytest.approx(5.0)

        widget._on_reset()
        assert widget.get_values()["phase"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Unit tests: FigureManagerQTRemote with update widget
# ---------------------------------------------------------------------------


class TestFigureManagerQTRemoteUpdateWidget:
    def test_no_update_widget_when_schema_none(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=None)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        assert manager.update_widget is None

    def test_update_widget_created_when_schema_present(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        assert manager.update_widget is not None
        assert isinstance(manager.update_widget, UpdateParametersWidget)
        assert "phase" in manager.update_widget._inputs

    def test_update_widget_docked_in_window(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        manager.show()
        qtbot.waitExposed(manager.window)

        # The dock widget should be a child of the main window
        assert manager.update_widget is not None
        assert manager.update_widget.parent() is manager.window

    def test_destroy_cleans_up_update_widget(self, qtbot: Any) -> None:
        config = _make_server_config(update_schema=_SIMPLE_UPDATE_SCHEMA)
        transport = _make_mock_transport(config)
        fig = Figure()
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas._transport_thread = MagicMock()

        manager = FigureManagerQTRemote(canvas, num=1)
        qtbot.addWidget(manager.window)

        assert manager.update_widget is not None
        manager.destroy()
        assert manager.update_widget is None


# ---------------------------------------------------------------------------
# Integration test: update parameters with real server
# ---------------------------------------------------------------------------


class TestUpdateParametersIntegration:
    def test_updatable_plot_has_update_widget(
        self, qtbot: Any, server_url: str
    ) -> None:
        """open_remote_figure for an updatable plot creates the dock widget."""
        manager = open_remote_figure(
            url=server_url,
            plot_name="updatable",
            init_params={"value": 1.0},
        )
        qtbot.addWidget(manager.window)

        try:
            assert manager.update_widget is not None
            assert isinstance(manager.update_widget, UpdateParametersWidget)

            # The schema should have a "phase" property
            assert "phase" in manager.update_widget._inputs
        finally:
            manager.destroy()
            QApplication.processEvents()

    def test_non_updatable_plot_has_no_update_widget(
        self, qtbot: Any, server_url: str
    ) -> None:
        """open_remote_figure for a non-updatable plot has no dock widget."""
        manager = open_remote_figure(
            url=server_url,
            plot_name="simple",
            init_params={"value": 1.0},
        )
        qtbot.addWidget(manager.window)

        try:
            assert manager.update_widget is None
        finally:
            manager.destroy()
            QApplication.processEvents()

    def test_submit_update_params_triggers_redraw(
        self, qtbot: Any, server_url: str
    ) -> None:
        """Submitting parameters sends update_params and gets a new image."""
        manager = open_remote_figure(
            url=server_url,
            plot_name="updatable",
            init_params={"value": 1.0},
        )
        qtbot.addWidget(manager.window)

        try:
            manager.show()
            qtbot.waitExposed(manager.window)

            # Wait for initial image
            for _ in range(100):
                QApplication.processEvents()
                if manager.canvas._remote_image is not None:
                    break
                time.sleep(0.05)
            assert manager.canvas._remote_image is not None

            # Remember the initial sequence number
            initial_seq = manager.canvas._last_seq_num

            # Submit an update
            assert manager.update_widget is not None
            manager.update_widget.set_values({"phase": 1.57})
            manager.update_widget._on_submit()

            # Wait for new image (sequence number should advance)
            for _ in range(100):
                QApplication.processEvents()
                if manager.canvas._last_seq_num > initial_seq:
                    break
                time.sleep(0.05)

            assert manager.canvas._last_seq_num > initial_seq
        finally:
            manager.destroy()
            QApplication.processEvents()


# ---------------------------------------------------------------------------
# Reconnection tests (unit, mocked transport)
# ---------------------------------------------------------------------------


class TestFigureCanvasQTRemoteReconnectOverlay:
    """Tests for the reconnect overlay on FigureCanvasQTRemote."""

    def test_overlay_initially_hidden(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        assert canvas._reconnect_overlay.isHidden()

    def test_show_reconnect_overlay(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        canvas._show_reconnect_overlay(2, 10)

        assert canvas._reconnect_overlay.isVisible()
        assert "2/10" in canvas._reconnect_overlay.text()

    def test_hide_reconnect_overlay(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        canvas._show_reconnect_overlay(1, 5)
        assert not canvas._reconnect_overlay.isHidden()

        canvas._hide_reconnect_overlay()
        assert canvas._reconnect_overlay.isHidden()

    def test_show_disconnected_overlay(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)
        canvas.show()
        qtbot.waitExposed(canvas)

        canvas._show_disconnected_overlay()

        assert canvas._reconnect_overlay.isVisible()
        assert "Disconnected" in canvas._reconnect_overlay.text()

    def test_on_reconnected_hides_overlay(self, qtbot: Any) -> None:
        config = _make_server_config()
        transport = _make_mock_transport(config)
        fig = Figure(figsize=(6.4, 4.8), dpi=100)
        canvas = FigureCanvasQTRemote(fig, transport, config)
        qtbot.addWidget(canvas)

        # Show the overlay
        canvas._show_reconnect_overlay(1, 3)
        assert not canvas._reconnect_overlay.isHidden()

        # Reconnect
        new_config = _make_server_config(connection_id="reconnected-1")
        canvas._on_reconnected(new_config)

        assert canvas._reconnect_overlay.isHidden()
        assert canvas._server_config.connection_id == "reconnected-1"
        assert canvas._remote_image is None
        assert canvas._remote_qpixmap is None
        # Should send resize (to current widget size) then refresh
        calls = transport.send_json.call_args_list
        assert any(c.args[0].get("type") == "resize" for c in calls)
        assert calls[-1].args[0] == {"type": "refresh"}


# ---------------------------------------------------------------------------
# SchemaFormBuilder
# ---------------------------------------------------------------------------


class TestSchemaFormBuilder:
    """Tests for SchemaFormBuilder."""

    def _make_schema(self, **properties: Any) -> dict[str, Any]:
        return {"type": "object", "properties": properties}

    def test_build_number_widget(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            freq={"type": "number", "default": 2.0, "minimum": 0.1, "maximum": 10.0}
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        assert "freq" in builder.inputs
        w = builder.inputs["freq"]
        assert isinstance(w, QDoubleSpinBox)
        assert w.value() == pytest.approx(2.0)
        assert w.minimum() == pytest.approx(0.1)
        assert w.maximum() == pytest.approx(10.0)

    def test_build_integer_widget(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            count={"type": "integer", "default": 5, "minimum": 1, "maximum": 100}
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        w = builder.inputs["count"]
        assert isinstance(w, QDoubleSpinBox)
        assert w.decimals() == 0
        assert w.value() == pytest.approx(5.0)

    def test_build_boolean_widget(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(enabled={"type": "boolean", "default": True})
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        w = builder.inputs["enabled"]
        assert isinstance(w, QCheckBox)
        assert w.isChecked() is True

    def test_build_string_widget(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(label={"type": "string", "default": "hello"})
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        w = builder.inputs["label"]
        assert isinstance(w, QLineEdit)
        assert w.text() == "hello"

    def test_build_enum_widget(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            color={
                "type": "string",
                "enum": ["red", "green", "blue"],
                "default": "green",
            }
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        w = builder.inputs["color"]
        assert isinstance(w, QComboBox)
        assert w.currentText() == "green"
        assert w.count() == 3

    def test_get_values(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            freq={"type": "number", "default": 2.0, "minimum": 0.0, "maximum": 10.0},
            enabled={"type": "boolean", "default": False},
            label={"type": "string", "default": "test"},
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        values = builder.get_values()
        assert values["freq"] == pytest.approx(2.0)
        assert values["enabled"] is False
        assert values["label"] == "test"

    def test_set_values(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            freq={"type": "number", "default": 2.0, "minimum": 0.0, "maximum": 10.0},
            enabled={"type": "boolean", "default": False},
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        builder.set_values({"freq": 7.5, "enabled": True})
        assert builder.inputs["freq"].value() == pytest.approx(7.5)
        assert builder.inputs["enabled"].isChecked() is True

    def test_reset_to_defaults(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = self._make_schema(
            freq={"type": "number", "default": 2.0, "minimum": 0.0, "maximum": 10.0},
        )
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        builder.set_values({"freq": 9.0})
        assert builder.inputs["freq"].value() == pytest.approx(9.0)

        builder.reset_to_defaults()
        assert builder.inputs["freq"].value() == pytest.approx(2.0)

    def test_empty_schema(self, qtbot: Any) -> None:
        from mpl_fastapi.remote.backend_qtremote import SchemaFormBuilder

        schema = {"type": "object", "properties": {}}
        builder = SchemaFormBuilder(schema)

        container = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(container)
        qtbot.addWidget(container)
        builder.build_form(layout)

        assert builder.inputs == {}
        assert builder.get_values() == {}


# ---------------------------------------------------------------------------
# FigureLauncherWindow
# ---------------------------------------------------------------------------


class TestFigureLauncherWindow:
    """Tests for FigureLauncherWindow."""

    def _make_plots(self) -> list:
        from mpl_fastapi.remote.backend_remote import RemotePlotInfo

        return [
            RemotePlotInfo(
                name="sine",
                description="A sine wave",
                ws_url="ws://localhost:8000/plots/ws/v0/sine",
                view_url="http://localhost:8000/plots/plot/sine",
                init_schema={
                    "type": "object",
                    "properties": {
                        "freq": {
                            "type": "number",
                            "default": 1.0,
                            "minimum": 0.1,
                            "maximum": 10.0,
                        }
                    },
                },
                update_schema=None,
            ),
            RemotePlotInfo(
                name="cosine",
                description="A cosine wave",
                ws_url="ws://localhost:8000/plots/ws/v0/cosine",
                view_url="http://localhost:8000/plots/plot/cosine",
                init_schema={"type": "object", "properties": {}},
                update_schema={
                    "type": "object",
                    "properties": {
                        "phase": {
                            "type": "number",
                            "default": 0.0,
                            "minimum": 0.0,
                            "maximum": 6.28,
                        }
                    },
                },
            ),
        ]

    def test_creates_ui(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        assert win._plot_list is not None
        assert win._launch_button is not None
        assert not win._launch_button.isEnabled()

    def test_discovery_populates_list(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        plots = self._make_plots()
        win._on_discovery_finished(plots)

        assert win._plot_list.count() == 2
        assert "sine" in win._plot_list.item(0).text()
        assert "cosine" in win._plot_list.item(1).text()

    def test_status_label_after_discovery(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_finished(self._make_plots())
        assert "2 figures available" in win._status_label.text()

    def test_discovery_error_shows_message(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_error("Connection refused")
        assert "Error" in win._status_label.text()
        assert "Connection refused" in win._status_label.text()

    def test_selecting_plot_enables_launch(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_finished(self._make_plots())

        # First item auto-selected, launch should be enabled
        assert win._launch_button.isEnabled()

    def test_selecting_plot_with_init_schema_shows_form(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_finished(self._make_plots())
        # sine (row 0) has init params
        win._plot_list.setCurrentRow(0)

        assert not win._init_group.isHidden()
        assert win._init_form is not None
        assert "freq" in win._init_form.inputs

    def test_selecting_plot_with_update_schema_shows_form(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_finished(self._make_plots())
        # cosine (row 1) has update params
        win._plot_list.setCurrentRow(1)

        assert not win._update_group.isHidden()
        assert win._update_form is not None
        assert "phase" in win._update_form.inputs

    def test_selecting_plot_hides_irrelevant_forms(self, qtbot: Any) -> None:
        from unittest.mock import patch

        from mpl_fastapi.remote.backend_qtremote import FigureLauncherWindow

        with patch("mpl_fastapi.remote.backend_qtremote._DiscoveryWorker"):
            win = FigureLauncherWindow("ws://localhost:8000/plots")
            qtbot.addWidget(win)

        win._on_discovery_finished(self._make_plots())
        # sine (row 0) has init but no update
        win._plot_list.setCurrentRow(0)

        assert not win._init_group.isHidden()
        assert win._update_group.isHidden()

        # cosine (row 1) has update but empty init
        win._plot_list.setCurrentRow(1)

        assert win._init_group.isHidden()
        assert not win._update_group.isHidden()
