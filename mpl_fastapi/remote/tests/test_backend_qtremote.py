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
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel, Field
from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import QApplication

from mpl_fastapi import InitConfig, PlotConfig, create_mpl_router, install_mpl_router
from mpl_fastapi.remote.backend_qtremote import (
    FigureCanvasQTRemote,
    FigureManagerQTRemote,
    NavigationToolbar2QTRemote,
    TransportThread,
    open_remote_figure,
)
from mpl_fastapi.remote.transport import RemoteTransport, ServerConfig
from mpl_fastapi.ws_client import ImageFormat, ImageTypeMode

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(**overrides: Any) -> ServerConfig:
    """Create a ServerConfig with sensible defaults."""
    defaults = {
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


def _create_simple_figure(fig: Figure, params: SimpleParams) -> dict[str, object]:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 10, 100)
    ax.plot(x, np.sin(x * params.value))
    ax.set_title(f"Simple (value={params.value})")
    return {"ax": ax}


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

    sockets = server.servers[0].sockets  # type: ignore[union-attr]
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

        # Before showing: the canvas has the handshake DPR (1.0)
        assert canvas.device_pixel_ratio == 1.0
        assert fig._original_dpi == 100.0

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
            assert fig._original_dpi == 100.0
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
        assert fig._original_dpi == 100.0

        calls = transport.send_json.call_args_list
        dpr_msgs = [c for c in calls if c[0][0].get("type") == "set_device_pixel_ratio"]
        assert len(dpr_msgs) == 1
        assert dpr_msgs[0][0][0]["device_pixel_ratio"] == target_dpr
        assert transport._device_pixel_ratio == target_dpr

        # A resize should also have been sent
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
