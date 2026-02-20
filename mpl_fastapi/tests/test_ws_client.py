"""Tests for Python WebSocket client.

These tests verify that the MatplotlibWebSocketClient correctly implements
the full matplotlib-fastapi WebSocket protocol using the TestClient adapter.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mpl_fastapi.ws_client import (
    MatplotlibWebSocketClient,
    create_fastapi_test_client_adapter,
)


class TestMatplotlibWebSocketClient:
    """Tests for the Python WebSocket client."""

    def test_basic_connection_and_initialization(self, client: TestClient) -> None:
        """Test basic connection and initialization flow."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Verify server-provided configuration was received
            assert ws_client.connection_id is not None
            assert ws_client.image_mode in ("full", "diff")
            assert len(ws_client.toolbar_items) > 0
            assert len(ws_client.save_formats) > 0
            assert ws_client.default_save_format is not None
            assert ws_client._server_protocol_version == 0
            assert ws_client._initialized is True

    def test_connection_with_init_params(self, client: TestClient) -> None:
        """Test connection with initialization parameters."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 2.0},
        )

        with ws_client.connect():
            assert ws_client.connection_id is not None
            assert ws_client._initialized is True

    def test_draw_request_and_image_retrieval(self, client: TestClient) -> None:
        """Test requesting a draw and receiving image data."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Initial draw was already processed during connection
            # Request another draw
            ws_client.send_draw()
            image_data = ws_client.wait_for_image()

            # Verify it's valid PNG data
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0
            img = Image.open(io.BytesIO(image_data))
            assert img.format == "PNG"
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_refresh_request(self, client: TestClient) -> None:
        """Test refresh request."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send refresh
            ws_client.send_refresh()

            # Should receive figure_label and draw messages
            msg = ws_client.receive_message()
            assert isinstance(msg, dict)
            assert msg["type"] == "figure_label"

            msg = ws_client.receive_message()
            assert isinstance(msg, dict)
            assert msg["type"] == "draw"

            # Now request the actual draw
            ws_client.send_draw()
            image_data = ws_client.wait_for_image()
            assert isinstance(image_data, bytes)

    def test_resize_request(self, client: TestClient) -> None:
        """Test resize request."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send resize
            ws_client.send_resize(width=800, height=600)

            # Should receive resize confirmation
            msg = ws_client.receive_message()
            assert isinstance(msg, dict)
            assert msg["type"] == "resize"
            assert msg["size"] == [800, 600]
            assert msg["forward"] is True

    def test_toolbar_button_actions(self, client: TestClient) -> None:
        """Test toolbar button actions."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Press pan button
            ws_client.send_toolbar_button("pan")

            # Should receive navigate_mode message (at minimum)
            # Toolbar buttons queue messages that are sent after drain_queue
            msg = ws_client.receive_message()
            assert isinstance(msg, dict)
            # Could be navigate_mode, message, or history_buttons
            assert msg["type"] in ("navigate_mode", "message", "history_buttons")

    def test_update_params(self, client: TestClient) -> None:
        """Test parameter update."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="updatable",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # Update parameters
            ws_client.send_update_params({"phase": 1.57})

            # Should receive draw message
            msg = ws_client.wait_for_message_type("draw")
            assert msg["type"] == "draw"

            # Request the actual draw
            ws_client.send_draw()
            image_data = ws_client.wait_for_image()
            assert isinstance(image_data, bytes)

    def test_mouse_events(self, client: TestClient) -> None:
        """Test mouse event sending."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send mouse events
            ws_client.send_mouse_event("button_press", x=100, y=100, button=1)
            ws_client.send_mouse_event("motion_notify", x=150, y=150)
            ws_client.send_mouse_event("button_release", x=150, y=150, button=1)
            # Events processed successfully (no response expected for these)

    def test_keyboard_events(self, client: TestClient) -> None:
        """Test keyboard event sending."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send keyboard events
            ws_client.send_keyboard_event("key_press", key="a")
            ws_client.send_keyboard_event("key_release", key="a")
            # Events processed successfully (no response expected for these)

    def test_save_figure(self, client: TestClient) -> None:
        """Test save figure request."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Request save
            save_info = ws_client.send_save_figure(format="png", dpi=150)

            # Verify response
            assert save_info["file_id"] is not None
            assert save_info["download_url"] is not None
            assert save_info["filename"] == "simple.png"
            assert save_info["format"] == "png"

    def test_device_pixel_ratio(self, client: TestClient) -> None:
        """Test device pixel ratio setting."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            device_pixel_ratio=2.0,
        )

        with ws_client.connect():
            # Connection should succeed with DPI set
            # The initialization handles the DPI change and initial draw
            assert ws_client.connection_id is not None
            assert ws_client._initialized is True

    def test_wait_for_message_type(self, client: TestClient) -> None:
        """Test waiting for specific message type."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send refresh which triggers figure_label
            ws_client.send_refresh()

            # Wait for figure_label specifically
            msg = ws_client.wait_for_message_type("figure_label")
            assert msg["type"] == "figure_label"

            # After finding the message, we're done
            # (there's also a draw message but we don't need to drain it for this test)

    def test_context_manager_cleanup(self, client: TestClient) -> None:
        """Test that context manager properly cleans up."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        # Before connection
        assert not ws_client._initialized
        assert not adapter.is_connected()

        with ws_client.connect():
            # During connection
            assert ws_client._initialized
            assert adapter.is_connected()

        # After connection
        assert not ws_client._initialized
        assert not adapter.is_connected()

    def test_error_when_not_connected(self, client: TestClient) -> None:
        """Test that methods raise error when not connected."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        # Should raise error when calling methods without connection
        with pytest.raises(RuntimeError, match="Client not initialized"):
            ws_client.send_draw()

        with pytest.raises(RuntimeError, match="Client not initialized"):
            ws_client.send_refresh()

        with pytest.raises(RuntimeError, match="Client not initialized"):
            ws_client.receive_message()

    def test_invalid_plot_name(self, client: TestClient) -> None:
        """Test connection to invalid plot name."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="nonexistent",
        )

        # Should fail to connect - TestClient closes connection with code 1008
        with pytest.raises(Exception), ws_client.connect():  # noqa: B017, PT011
            pass

    def test_multiple_sequential_connections(self, client: TestClient) -> None:
        """Test multiple sequential connections to same plot."""
        adapter1 = create_fastapi_test_client_adapter(client)
        ws_client1 = MatplotlibWebSocketClient(
            adapter=adapter1,
            base_url="/plots",
            plot_name="simple",
        )

        adapter2 = create_fastapi_test_client_adapter(client)
        ws_client2 = MatplotlibWebSocketClient(
            adapter=adapter2,
            base_url="/plots",
            plot_name="simple",
        )

        # First connection
        with ws_client1.connect():
            conn_id_1 = ws_client1.connection_id

        # Second connection (should get new connection_id)
        with ws_client2.connect():
            conn_id_2 = ws_client2.connection_id

        # Connection IDs should be different
        assert conn_id_1 != conn_id_2


class TestWebSocketAdapters:
    """Tests for WebSocket adapter implementations."""

    def test_test_client_adapter_basic(self, client: TestClient) -> None:
        """Test TestClientAdapter basic operations."""
        adapter = create_fastapi_test_client_adapter(client)

        # Before connection
        assert not adapter.is_connected()

        # Connect
        adapter.connect("/plots/ws/simple")
        assert adapter.is_connected()

        # Send and receive
        adapter.send_json({"type": "test"})
        # Note: actual test would need proper server response

        # Disconnect
        adapter.disconnect()
        assert not adapter.is_connected()

    def test_test_client_adapter_error_when_not_connected(
        self, client: TestClient
    ) -> None:
        """Test that adapter raises error when operations attempted without connection."""
        adapter = create_fastapi_test_client_adapter(client)

        with pytest.raises(RuntimeError, match="Not connected"):
            adapter.send_json({"type": "test"})

        with pytest.raises(RuntimeError, match="Not connected"):
            adapter.receive_json()

        with pytest.raises(RuntimeError, match="Not connected"):
            adapter.send_bytes(b"data")

        with pytest.raises(RuntimeError, match="Not connected"):
            adapter.receive_bytes()
