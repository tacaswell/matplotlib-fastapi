"""Tests for FigureCanvasRemote — the toolkit-agnostic remote canvas."""

from __future__ import annotations

import io
import struct
from typing import Any
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from PIL import Image

from mpl_fastapi.remote.backend_remote import (
    FigureCanvasRemote,
    FigureManagerRemote,
    RemoteNavigationToolbar2,
)
from mpl_fastapi.remote.transport import RemoteTransport, ServerConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_server_config(**overrides: Any) -> ServerConfig:
    defaults: dict[str, Any] = {
        "connection_id": "test-conn-123",
        "protocol_version": 0,
        "figure_size": (640, 480),
        "figure_dpi": 100.0,
        "figure_label": "Test",
        "toolbar_items": [
            ["Home", "Reset", "home", "home"],
            ["Back", "Back", "back", "back"],
            ["Forward", "Forward", "forward", "forward"],
        ],
        "save_formats": ["png", "pdf"],
        "default_save_format": "png",
        "image_format": "png",
        "update_schema": None,
    }
    defaults.update(overrides)
    return ServerConfig(**defaults)


def _make_full_image_message(
    width: int = 64,
    height: int = 48,
    seq_num: int = 1,
    color: tuple[int, int, int, int] = (255, 0, 0, 255),
) -> bytes:
    """Build a binary message with 8-byte header + PNG data."""
    img = Image.new("RGBA", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # Header: type_mode=FULL(0x00), format=PNG(0x01), seq_num, base_seq=0, flags=0
    header = struct.pack(">BBHHH", 0x00, 0x01, seq_num, 0, 0)
    return header + png_bytes


def _make_diff_image_message(
    width: int = 64,
    height: int = 48,
    seq_num: int = 2,
    base_seq: int = 1,
) -> bytes:
    """Build a diff binary message (transparent except a red square)."""
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    # Draw a small non-transparent region
    for x in range(10, 20):
        for y in range(10, 20):
            img.putpixel((x, y), (0, 255, 0, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    header = struct.pack(">BBHHH", 0x01, 0x01, seq_num, base_seq, 0)
    return header + png_bytes


def _make_mock_transport(device_pixel_ratio: float = 1.0) -> MagicMock:
    """Create a mock RemoteTransport."""
    transport = MagicMock(spec=RemoteTransport)
    transport.send_json = MagicMock()
    transport.send_json_async = MagicMock()
    transport.is_connected = True
    transport._device_pixel_ratio = device_pixel_ratio
    return transport


# ---------------------------------------------------------------------------
# FigureCanvasRemote tests
# ---------------------------------------------------------------------------


class TestFigureCanvasRemoteInit:
    """Test canvas construction and initial state."""

    def test_constructor_syncs_figure(self) -> None:
        fig = Figure()
        config = _make_server_config(
            figure_size=(800, 600), figure_dpi=150.0, figure_label="MyPlot"
        )
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        assert canvas._remote_image is None
        assert canvas._server_config is config
        assert canvas.get_width_height() == (800, 600)
        # Figure should be synced
        assert fig.get_dpi() == 150.0
        w, h = fig.get_size_inches()
        assert abs(w - 800 / 150.0) < 0.01
        assert abs(h - 600 / 150.0) < 0.01
        assert fig.get_label() == "MyPlot"

    def test_transport_property(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        assert canvas.transport is transport

    def test_hidpi_figure_geometry(self) -> None:
        """With DPR=2, figure size_inches should match original (not halved)."""
        fig = Figure()
        # Server sends CSS pixels (640x480) and *scaled* DPI (200 = 100 * 2)
        config = _make_server_config(figure_size=(640, 480), figure_dpi=200.0)
        transport = _make_mock_transport(device_pixel_ratio=2.0)
        canvas = FigureCanvasRemote(fig, transport, config)

        # Original DPI = 200 / 2 = 100
        # Size inches = 640/100 = 6.4, 480/100 = 4.8
        w, h = fig.get_size_inches()
        assert abs(w - 6.4) < 0.01
        assert abs(h - 4.8) < 0.01
        # DPI should be the scaled DPI
        assert fig.get_dpi() == 200.0
        # Canvas device_pixel_ratio should be set
        assert canvas.device_pixel_ratio == 2.0
        # get_width_height returns CSS pixels
        assert canvas.get_width_height() == (640, 480)

    def test_hidpi_resize_message(self) -> None:
        """Server resize message with DPR=2 should compute inches correctly."""
        fig = Figure()
        config = _make_server_config(figure_size=(640, 480), figure_dpi=200.0)
        transport = _make_mock_transport(device_pixel_ratio=2.0)
        canvas = FigureCanvasRemote(fig, transport, config)

        # Server sends a resize back (CSS pixels)
        canvas._on_json_message({"type": "resize", "size": [800, 600]})
        w, h = fig.get_size_inches()
        # original_dpi = 200 / 2 = 100, size = 800/100 x 600/100
        assert abs(w - 8.0) < 0.01
        assert abs(h - 6.0) < 0.01
        assert canvas.get_width_height() == (800, 600)


class TestFigureCanvasRemoteBinaryMessages:
    """Test image compositing from binary messages."""

    def test_full_image(self) -> None:
        fig = Figure()
        config = _make_server_config(figure_size=(64, 48))
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        data = _make_full_image_message(64, 48, seq_num=1, color=(255, 0, 0, 255))
        canvas._on_binary_message(data)

        assert canvas._remote_image is not None
        assert canvas._remote_image.size == (64, 48)
        assert canvas._last_seq_num == 1

    def test_diff_compositing(self) -> None:
        fig = Figure()
        config = _make_server_config(figure_size=(64, 48))
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        # First: full image (red)
        full_data = _make_full_image_message(64, 48, seq_num=1, color=(255, 0, 0, 255))
        canvas._on_binary_message(full_data)
        assert canvas._remote_image is not None

        # Second: diff (green patch)
        diff_data = _make_diff_image_message(64, 48, seq_num=2, base_seq=1)
        canvas._on_binary_message(diff_data)

        assert canvas._last_seq_num == 2
        # The pixel at (15, 15) should be green from the diff
        pixel = canvas._remote_image.getpixel((15, 15))
        assert pixel == (0, 255, 0, 255)
        # A pixel outside the diff patch should still be red
        pixel_outside = canvas._remote_image.getpixel((0, 0))
        assert pixel_outside == (255, 0, 0, 255)

    def test_diff_without_base_treats_as_full(self) -> None:
        fig = Figure()
        config = _make_server_config(figure_size=(64, 48))
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        diff_data = _make_diff_image_message(64, 48, seq_num=1, base_seq=0)
        canvas._on_binary_message(diff_data)
        # Should not crash — treated as full
        assert canvas._remote_image is not None


class TestFigureCanvasRemoteJsonMessages:
    """Test JSON message dispatch."""

    def test_invalidate_sends_render(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._on_json_message({"type": "invalidate"})

        transport.send_json.assert_called_once_with({"type": "render"})

    def test_resize_updates_config(self) -> None:
        fig = Figure()
        config = _make_server_config(figure_size=(640, 480))
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._on_json_message({"type": "resize", "size": [800, 600], "forward": True})

        assert canvas._server_config.figure_size == (800, 600)

    def test_rubberband_sets_rect(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._on_json_message(
            {"type": "rubberband", "x0": 10, "y0": 20, "x1": 100, "y1": 200}
        )
        assert canvas._rubberband_rect == (10, 20, 100, 200)

    def test_rubberband_clears(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._rubberband_rect = (10, 20, 100, 200)
        canvas._on_json_message(
            {"type": "rubberband", "x0": -1, "y0": -1, "x1": -1, "y1": -1}
        )
        assert canvas._rubberband_rect is None

    def test_error_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        with caplog.at_level("ERROR"):
            canvas._on_json_message({"type": "error", "message": "test error 42"})
        assert "test error 42" in caplog.text


class TestFigureCanvasRemoteEventForwarding:
    """Test event forwarding to the transport."""

    def test_forward_mouse_event(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_mouse_event("button_press", 100, 200, button=0)
        transport.send_json.assert_called_once_with(
            {"type": "button_press", "x": 100, "y": 200, "button": 0}
        )

    def test_forward_scroll_includes_step(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_mouse_event("scroll", 50, 60, button=0, step=3)
        call_args = transport.send_json.call_args[0][0]
        assert call_args["type"] == "scroll"
        assert call_args["step"] == 3

    def test_forward_key_event(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_key_event("key_press", "a")
        transport.send_json.assert_called_once_with({"type": "key_press", "key": "a"})

    def test_forward_resize(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_resize(800, 600)
        transport.send_json.assert_called_once_with(
            {"type": "resize", "width": 800, "height": 600}
        )

    def test_forward_toolbar_button(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_toolbar_button("home")
        transport.send_json.assert_called_once_with(
            {"type": "toolbar_button", "name": "home"}
        )


# ---------------------------------------------------------------------------
# RemoteNavigationToolbar2 tests
# ---------------------------------------------------------------------------


class TestRemoteNavigationToolbar2:
    """Test the remote toolbar."""

    def _make_toolbar(
        self,
    ) -> tuple[RemoteNavigationToolbar2, MagicMock]:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        toolbar = RemoteNavigationToolbar2(canvas)
        canvas.toolbar = toolbar
        return toolbar, transport

    def test_home_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.home()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "home"}
        )

    def test_pan_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.pan()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "pan"}
        )

    def test_zoom_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.zoom()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "zoom"}
        )

    def test_back_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.back()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "back"}
        )

    def test_forward_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.forward()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "forward"}
        )

    def test_download_sends_toolbar_button(self) -> None:
        toolbar, transport = self._make_toolbar()
        toolbar.download()
        transport.send_json.assert_called_with(
            {"type": "toolbar_button", "name": "download"}
        )

    def test_set_message(self) -> None:
        toolbar, _ = self._make_toolbar()
        toolbar.set_message("x=1.23, y=4.56")
        assert toolbar.message == "x=1.23, y=4.56"

    def test_set_history_buttons_noop_during_init(self) -> None:
        """set_history_buttons should not crash during __init__."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        # This should not raise
        toolbar = RemoteNavigationToolbar2(canvas)
        assert toolbar is not None


# ---------------------------------------------------------------------------
# FigureManagerRemote tests
# ---------------------------------------------------------------------------


class TestFigureManagerRemote:
    def test_creates_toolbar(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        manager = FigureManagerRemote(canvas, num=-1)

        assert manager.toolbar is not None
        assert isinstance(manager.toolbar, RemoteNavigationToolbar2)
        assert manager.canvas is canvas
