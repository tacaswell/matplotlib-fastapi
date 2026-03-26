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


def _make_mock_transport(
    device_pixel_ratio: float = 1.0,
    url: str = "ws://localhost:8000/ws/test-123",
) -> MagicMock:
    """Create a mock RemoteTransport."""
    transport = MagicMock(spec=RemoteTransport)
    transport.send_json = MagicMock()
    transport.send_json_async = MagicMock()
    transport.is_connected = True
    transport._device_pixel_ratio = device_pixel_ratio
    transport._url = url
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

    def test_rubberband_dispatches_to_toolbar_draw(self) -> None:
        """Rubberband message calls toolbar.draw_rubberband when toolbar exists."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        toolbar = RemoteNavigationToolbar2(canvas)
        canvas.toolbar = toolbar
        toolbar.draw_rubberband = MagicMock()  # type: ignore[method-assign]

        canvas._on_json_message(
            {"type": "rubberband", "x0": 10, "y0": 20, "x1": 100, "y1": 200}
        )
        toolbar.draw_rubberband.assert_called_once_with(None, 10, 20, 100, 200)

    def test_rubberband_dispatches_to_toolbar_remove(self) -> None:
        """Clear-rubberband message calls toolbar.remove_rubberband."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        toolbar = RemoteNavigationToolbar2(canvas)
        canvas.toolbar = toolbar
        toolbar.remove_rubberband = MagicMock()  # type: ignore[method-assign]

        canvas._rubberband_rect = (10, 20, 100, 200)
        canvas._on_json_message(
            {"type": "rubberband", "x0": -1, "y0": -1, "x1": -1, "y1": -1}
        )
        toolbar.remove_rubberband.assert_called_once()

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


# ---------------------------------------------------------------------------
# Save-flow tests
# ---------------------------------------------------------------------------


class TestForwardSaveFigure:
    """Test the _forward_save_figure message format."""

    def test_default_args(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_save_figure()
        transport.send_json.assert_called_once_with(
            {
                "type": "save_figure",
                "format": "png",
                "dpi": 100.0,
                "transparent": False,
            }
        )

    def test_custom_args(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_save_figure(format="pdf", dpi=300.0, transparent=True)
        transport.send_json.assert_called_once_with(
            {
                "type": "save_figure",
                "format": "pdf",
                "dpi": 300.0,
                "transparent": True,
            }
        )


class TestSaveCompleteDispatch:
    """Test _on_json_message routing for save_complete / save_error."""

    def _make_canvas_with_toolbar(
        self,
    ) -> tuple[FigureCanvasRemote, RemoteNavigationToolbar2, MagicMock]:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        toolbar = RemoteNavigationToolbar2(canvas)
        canvas.toolbar = toolbar
        return canvas, toolbar, transport

    def test_save_complete_routes_to_toolbar(self) -> None:
        """Without a pending callback, save_complete goes to toolbar."""
        canvas, toolbar, _ = self._make_canvas_with_toolbar()
        toolbar._on_save_complete = MagicMock()  # type: ignore[method-assign]

        msg = {"type": "save_complete", "download_url": "/download/test.png"}
        canvas._on_json_message(msg)

        toolbar._on_save_complete.assert_called_once_with(msg)

    def test_save_error_routes_to_toolbar(self) -> None:
        """Without a pending callback, save_error goes to toolbar."""
        canvas, toolbar, _ = self._make_canvas_with_toolbar()
        toolbar._on_save_error = MagicMock()  # type: ignore[method-assign]

        msg = {"type": "save_error", "message": "format not supported"}
        canvas._on_json_message(msg)

        toolbar._on_save_error.assert_called_once_with(msg)

    def test_save_complete_routes_to_pending_callback(self) -> None:
        """With a pending callback, save_complete goes there instead."""
        canvas, toolbar, _ = self._make_canvas_with_toolbar()
        toolbar._on_save_complete = MagicMock()  # type: ignore[method-assign]
        canvas._handle_print_figure_complete = MagicMock()  # type: ignore[method-assign]

        received: list[dict[str, Any]] = []
        canvas._pending_print_figure_callback = (
            "/tmp/test.png",
            received.append,
        )

        msg = {"type": "save_complete", "download_url": "/download/test.png"}
        canvas._on_json_message(msg)

        # Pending callback should have been called
        assert len(received) == 1
        assert received[0] is msg
        # _handle_print_figure_complete should have been called
        canvas._handle_print_figure_complete.assert_called_once_with(
            "/tmp/test.png", msg
        )
        # Toolbar should NOT have been called
        toolbar._on_save_complete.assert_not_called()
        # Pending callback should be consumed
        assert canvas._pending_print_figure_callback is None

    def test_save_error_routes_to_pending_callback(self) -> None:
        """With a pending callback, save_error goes there instead."""
        canvas, toolbar, _ = self._make_canvas_with_toolbar()
        toolbar._on_save_error = MagicMock()  # type: ignore[method-assign]

        received: list[dict[str, Any]] = []
        canvas._pending_print_figure_callback = (
            "/tmp/test.png",
            received.append,
        )

        msg = {"type": "save_error", "message": "bad format"}
        canvas._on_json_message(msg)

        # Pending callback should have been called
        assert len(received) == 1
        assert received[0] is msg
        # Toolbar should NOT have been called
        toolbar._on_save_error.assert_not_called()
        # Pending callback should be consumed
        assert canvas._pending_print_figure_callback is None


class TestPrintFigure:
    """Test the print_figure method (fig.savefig integration)."""

    def test_sends_save_figure_with_format_from_extension(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas.print_figure("/tmp/plot.pdf")

        call_args = transport.send_json.call_args[0][0]
        assert call_args["type"] == "save_figure"
        assert call_args["format"] == "pdf"
        assert canvas._pending_print_figure_callback is not None

    def test_sends_save_figure_with_explicit_format(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas.print_figure("/tmp/plot.dat", format="svg")

        call_args = transport.send_json.call_args[0][0]
        assert call_args["format"] == "svg"

    def test_dpi_figure_uses_original_dpi(self) -> None:
        """When dpi='figure', use _original_dpi if available."""
        fig = Figure()
        # Server sends scaled DPI with DPR=2
        config = _make_server_config(figure_dpi=200.0)
        transport = _make_mock_transport(device_pixel_ratio=2.0)
        canvas = FigureCanvasRemote(fig, transport, config)
        # _original_dpi should be 200/2 = 100
        assert getattr(fig, "_original_dpi", None) == 100.0

        canvas.print_figure("/tmp/plot.png", dpi="figure")

        call_args = transport.send_json.call_args[0][0]
        assert call_args["dpi"] == 100.0

    def test_explicit_dpi(self) -> None:
        fig = Figure()
        config = _make_server_config(figure_dpi=100.0)
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas.print_figure("/tmp/plot.png", dpi=300)

        call_args = transport.send_json.call_args[0][0]
        assert call_args["dpi"] == 300.0

    def test_transparent_flag(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas.print_figure("/tmp/plot.png", transparent=True)

        call_args = transport.send_json.call_args[0][0]
        assert call_args["transparent"] is True

    def test_file_like_raises(self) -> None:
        """print_figure should reject file-like objects."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        with pytest.raises(ValueError, match="file-like"):
            canvas.print_figure(io.BytesIO())  # type: ignore[arg-type]

    def test_registers_pending_callback(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        assert canvas._pending_print_figure_callback is None

        canvas.print_figure("/tmp/out.png")
        assert canvas._pending_print_figure_callback is not None
        path, cb = canvas._pending_print_figure_callback
        assert path == "/tmp/out.png"
        assert callable(cb)


class TestHandlePrintFigureComplete:
    """Test the download logic when a save_complete arrives for print_figure."""

    def test_calls_download_url_to_file(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        canvas._download_url_to_file = MagicMock()  # type: ignore[method-assign]

        msg = {"type": "save_complete", "download_url": "/download/test.png"}
        canvas._handle_print_figure_complete("/tmp/plot.png", msg)

        canvas._download_url_to_file.assert_called_once_with(
            "/download/test.png", "/tmp/plot.png"
        )

    def test_missing_download_url_logs_error(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        with caplog.at_level("ERROR"):
            canvas._handle_print_figure_complete(
                "/tmp/plot.png",
                {"type": "save_complete"},
            )
        assert "download_url" in caplog.text

    def test_download_failure_logs_exception(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        canvas._download_url_to_file = MagicMock(  # type: ignore[method-assign]
            side_effect=OSError("connection refused")
        )

        with caplog.at_level("ERROR"):
            canvas._handle_print_figure_complete(
                "/tmp/plot.png",
                {"type": "save_complete", "download_url": "/download/x.png"},
            )
        assert "Failed to download" in caplog.text


class TestDownloadUrlToFile:
    """Test _download_url_to_file URL construction."""

    def test_url_construction_ws(self) -> None:
        """ws:// should map to http://."""
        from unittest.mock import patch

        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport(url="ws://host:9000/ws/fig1")
        canvas = FigureCanvasRemote(fig, transport, config)

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            canvas._download_url_to_file("/download/out.png", "/tmp/out.png")
            mock_retrieve.assert_called_once_with(
                "http://host:9000/download/out.png", "/tmp/out.png"
            )

    def test_url_construction_wss(self) -> None:
        """wss:// should map to https://."""
        from unittest.mock import patch

        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport(url="wss://secure.host:443/ws/fig1")
        canvas = FigureCanvasRemote(fig, transport, config)

        with patch("urllib.request.urlretrieve") as mock_retrieve:
            canvas._download_url_to_file("/download/out.pdf", "/tmp/out.pdf")
            mock_retrieve.assert_called_once_with(
                "https://secure.host:443/download/out.pdf", "/tmp/out.pdf"
            )


class TestSaveFlowEndToEnd:
    """End-to-end test: print_figure → save_complete → download."""

    def test_print_figure_then_save_complete(self) -> None:
        """Simulate the full print_figure flow with a mock download."""
        from unittest.mock import patch

        fig = Figure()
        config = _make_server_config(figure_dpi=100.0)
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        FigureManagerRemote(canvas, num=-1)

        # 1) Call print_figure — registers callback, sends message
        canvas.print_figure("/tmp/end_to_end.png", dpi=150)
        assert canvas._pending_print_figure_callback is not None
        call_args = transport.send_json.call_args[0][0]
        assert call_args == {
            "type": "save_figure",
            "format": "png",
            "dpi": 150.0,
            "transparent": False,
        }

        # 2) Simulate server responding with save_complete
        with patch.object(canvas, "_download_url_to_file") as mock_dl:
            canvas._on_json_message(
                {"type": "save_complete", "download_url": "/download/e2e.png"}
            )
            mock_dl.assert_called_once_with("/download/e2e.png", "/tmp/end_to_end.png")

        # 3) Callback should be consumed
        assert canvas._pending_print_figure_callback is None

    def test_print_figure_then_save_error(self) -> None:
        """Simulate print_figure when the server returns save_error."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        _ = FigureManagerRemote(canvas, num=-1)

        canvas.print_figure("/tmp/will_fail.svg", format="svg")
        assert canvas._pending_print_figure_callback is not None

        # The on_complete callback raises on save_error,
        # but _on_json_message catches nothing — the callback runs directly.
        # Since the callback raises RuntimeError, we should see it propagate.
        with pytest.raises(RuntimeError, match="Server save failed"):
            canvas._on_json_message(
                {"type": "save_error", "message": "unsupported format"}
            )

        assert canvas._pending_print_figure_callback is None


# ---------------------------------------------------------------------------
# Tests: Update Parameters
# ---------------------------------------------------------------------------


class TestUpdateParams:
    """Test _forward_update_params and has_update_params."""

    def test_forward_update_params_sends_message(self) -> None:
        """_forward_update_params sends update_params via transport."""
        fig = Figure()
        config = _make_server_config(
            update_schema={"properties": {"phase": {"type": "number", "default": 0.0}}}
        )
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_update_params({"phase": 1.57})
        transport.send_json.assert_called_with(
            {"type": "update_params", "params": {"phase": 1.57}}
        )

    def test_forward_update_params_multiple_values(self) -> None:
        """_forward_update_params sends all parameter values."""
        fig = Figure()
        config = _make_server_config(
            update_schema={
                "properties": {
                    "phase": {"type": "number"},
                    "amplitude": {"type": "number"},
                }
            }
        )
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._forward_update_params({"phase": 1.0, "amplitude": 2.5})
        transport.send_json.assert_called_with(
            {"type": "update_params", "params": {"phase": 1.0, "amplitude": 2.5}}
        )

    def test_has_update_params_true_when_schema_present(self) -> None:
        """has_update_params is True when update_schema is non-null."""
        fig = Figure()
        config = _make_server_config(
            update_schema={"properties": {"phase": {"type": "number"}}}
        )
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        assert canvas.has_update_params is True

    def test_has_update_params_false_when_schema_none(self) -> None:
        """has_update_params is False when update_schema is None."""
        fig = Figure()
        config = _make_server_config(update_schema=None)
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        assert canvas.has_update_params is False


class TestRemoteNavigationToolbar2UpdateParams:
    """Test _on_update_params_submitted on the agnostic toolbar."""

    def test_on_update_params_submitted_forwards_to_canvas(self) -> None:
        fig = Figure()
        config = _make_server_config(
            update_schema={"properties": {"phase": {"type": "number"}}}
        )
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)
        toolbar = RemoteNavigationToolbar2(canvas)

        toolbar._on_update_params_submitted({"phase": 3.14})
        transport.send_json.assert_called_with(
            {"type": "update_params", "params": {"phase": 3.14}}
        )


class TestReconnection:
    """Tests for FigureCanvasRemote._on_reconnected."""

    def test_on_reconnected_clears_image(self) -> None:
        """_on_reconnected clears _remote_image."""
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        # Simulate having an image
        canvas._remote_image = Image.new("RGBA", (64, 48), (255, 0, 0, 255))
        assert canvas._remote_image is not None

        new_config = _make_server_config(connection_id="reconnected-456")
        canvas._on_reconnected(new_config)

        assert canvas._remote_image is None
        assert canvas._server_config.connection_id == "reconnected-456"

    def test_on_reconnected_clears_rubberband(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._rubberband_rect = (10, 20, 30, 40)
        new_config = _make_server_config(connection_id="reconnected-789")
        canvas._on_reconnected(new_config)

        assert canvas._rubberband_rect is None

    def test_on_reconnected_sends_resize_then_refresh(self) -> None:
        """_on_reconnected sends a resize (to client size) then refresh."""
        fig = Figure()
        config = _make_server_config(figure_size=(800, 600))
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        # Server reconnects with default 640×480, but client was at 800×600
        new_config = _make_server_config(
            connection_id="new-conn", figure_size=(640, 480)
        )
        canvas._on_reconnected(new_config)

        # Should send resize to the *client's* old size, then refresh
        calls = transport.send_json.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] == {
            "type": "resize",
            "width": 800,
            "height": 600,
        }
        assert calls[1].args[0] == {"type": "refresh"}

    def test_on_reconnected_resets_seq_num(self) -> None:
        fig = Figure()
        config = _make_server_config()
        transport = _make_mock_transport()
        canvas = FigureCanvasRemote(fig, transport, config)

        canvas._last_seq_num = 42
        new_config = _make_server_config(connection_id="new-conn")
        canvas._on_reconnected(new_config)

        assert canvas._last_seq_num == 0

    def test_on_reconnected_preserves_client_geometry(self) -> None:
        """_on_reconnected keeps the client's figure size, not the server default."""
        fig = Figure()
        config = _make_server_config(figure_size=(800, 600), figure_dpi=200.0)
        transport = _make_mock_transport(device_pixel_ratio=2.0)
        canvas = FigureCanvasRemote(fig, transport, config)

        # Record the figure geometry after initial setup
        orig_dpi = fig.dpi
        orig_w, orig_h = fig.get_size_inches()

        # Server reconnects with a different default size/dpi
        new_config = _make_server_config(
            connection_id="new",
            figure_size=(640, 480),
            figure_dpi=100.0,
        )
        canvas._on_reconnected(new_config)

        # The client's figure geometry should be unchanged — we keep
        # the client's size and tell the server to match it.
        assert fig.dpi == pytest.approx(orig_dpi)
        w, h = fig.get_size_inches()
        assert w == pytest.approx(orig_w)
        assert h == pytest.approx(orig_h)

        # _server_config.figure_size should reflect the client size
        assert canvas._server_config.figure_size == (800, 600)
