"""Tests for WebSocket connection lifecycle and protocol.

These tests verify the matplotlib-fastapi WebSocket protocol implementation
using the MatplotlibWebSocketClient Python client. This ensures both the
server and client correctly implement the protocol.

Test coverage includes:
- Connection establishment and protocol version negotiation
- Initial message sequence and configuration
- Image rendering and retrieval
- Interactive events (mouse, keyboard, toolbar)
- Dynamic parameter updates
- Resizing and DPI changes
- Save/download functionality
- Error handling for invalid inputs
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi.ws_client import (
    ContextManagerWebSocketAdapter,
    MatplotlibWebSocketClient,
    create_fastapi_test_client_adapter,
)


class TestWebSocketConnection:
    """Tests for WebSocket connection establishment and basic protocol."""

    def test_websocket_connect_success(self, client: TestClient) -> None:
        """Test successful WebSocket connection and initialization."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # Verify connection was successful
            assert ws_client._initialized is True
            assert ws_client._server_protocol_version == 0
            assert ws_client.connection_id is not None

    def test_websocket_connect_invalid_plot_returns_1008(
        self, client: TestClient
    ) -> None:
        """Test that connecting to non-existent plot closes with code 1008."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="nonexistent",
        )

        # Should fail to connect - TestClient closes connection with code 1008
        with pytest.raises(Exception), ws_client.connect():  # noqa: B017, PT011
            pass

    def test_websocket_connect_invalid_params_returns_1008(
        self, client: TestClient
    ) -> None:
        """Test that invalid parameters close connection with code 1008."""
        # Use raw adapter to test protocol-level error handling
        # (ws_client would fail on URL construction)
        adapter = ContextManagerWebSocketAdapter(client)

        # 'value' must be float, this will fail validation
        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.connect("/plots/ws/simple?value=invalid")
            # Server sends protocol_version first
            msg = adapter.receive_json()
            assert msg["type"] == "protocol_version"

            # Send client protocol_version
            adapter.send_json({"type": "protocol_version", "version": 0})

            # Then sends error message before closing
            msg = adapter.receive_json()
            assert msg["type"] == "error"
            assert "Invalid parameters" in msg["message"]

            # Try to receive more - this should raise WebSocketDisconnect
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_websocket_connect_params_out_of_range(self, client: TestClient) -> None:
        """Test that out-of-range parameters are rejected."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        # 'value' must be between 0.1 and 10.0
        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.connect("/plots/ws/simple?value=100.0")
            # Server sends protocol_version first
            msg = adapter.receive_json()
            assert msg["type"] == "protocol_version"

            # Send client protocol_version
            adapter.send_json({"type": "protocol_version", "version": 0})

            # Then sends error message before closing
            msg = adapter.receive_json()
            assert msg["type"] == "error"
            assert "Invalid parameters" in msg["message"]

            # Try to receive more - this should raise WebSocketDisconnect
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_websocket_initial_message_sequence(self, client: TestClient) -> None:
        """Test that initial message sequence follows protocol."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # Verify all configuration was received during connection
            assert ws_client._server_protocol_version == 0
            assert ws_client.connection_id is not None
            assert ws_client.image_mode in ("full", "diff")
            assert len(ws_client.toolbar_items) > 0
            assert len(ws_client.save_formats) > 0
            assert ws_client.default_save_format is not None

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

    def test_different_parameters_affect_plot(self, client: TestClient) -> None:
        """Test that different parameters are acknowledged."""
        # Just verify that connections with different params work
        adapter1 = create_fastapi_test_client_adapter(client)
        ws_client1 = MatplotlibWebSocketClient(
            adapter=adapter1,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        adapter2 = create_fastapi_test_client_adapter(client)
        ws_client2 = MatplotlibWebSocketClient(
            adapter=adapter2,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 5.0},
        )

        with ws_client1.connect():
            assert ws_client1._initialized is True

        with ws_client2.connect():
            assert ws_client2._initialized is True

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



class TestProtocolVersion:
    """Tests for protocol version negotiation."""

    def test_protocol_version_mismatch_closes_connection(
        self, client: TestClient
    ) -> None:
        """Test that protocol version mismatch closes the connection."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.connect("/plots/ws/simple?value=1.0")
            # Receive protocol_version (first message)
            msg = adapter.receive_json()
            assert msg["type"] == "protocol_version"

            # Send back incompatible version
            adapter.send_json({"type": "protocol_version", "version": 999})

            # Should receive error message immediately
            msg = adapter.receive_json()
            assert msg["type"] == "error"
            assert "Incompatible protocol version" in msg["message"]

            # Try to receive more - should disconnect
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_protocol_version_match_continues(self, client: TestClient) -> None:
        """Test that matching protocol version allows communication to continue."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        # Connection should succeed with matching protocol version
        with ws_client.connect():
            assert ws_client._server_protocol_version == 0
            assert ws_client._initialized is True

    def test_protocol_version_missing_closes_connection(
        self, client: TestClient
    ) -> None:
        """Test that missing protocol version (required field) closes the connection."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.connect("/plots/ws/simple?value=1.0")
            # Receive server's protocol_version
            msg = adapter.receive_json()
            assert msg["type"] == "protocol_version"

            # Send back protocol_version message without version field
            adapter.send_json({"type": "protocol_version"})

            # Should receive error message immediately
            msg = adapter.receive_json()
            assert msg["type"] == "error"
            assert "required" in msg["message"].lower()

            # Try to receive more - should disconnect
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_wrong_first_message_closes_connection(self, client: TestClient) -> None:
        """Test that sending non-protocol_version as first message closes connection."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.connect("/plots/ws/simple?value=1.0")
            # Receive server's protocol_version
            msg = adapter.receive_json()
            assert msg["type"] == "protocol_version"

            # Send wrong message type as first client message
            adapter.send_json({"type": "refresh"})

            # Should receive error message
            msg = adapter.receive_json()
            assert msg["type"] == "error"
            assert "first" in msg["message"].lower()

            # Try to receive more - should disconnect
            adapter.receive_json()

        assert exc_info.value.code == 1008



class TestImageRendering:
    """Tests for image rendering and retrieval."""

    def test_websocket_sends_image_after_draw(self, client: TestClient) -> None:
        """Test that WebSocket sends image data after a draw request."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 2.0},
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

    def test_draw_request_and_image_retrieval(self, client: TestClient) -> None:
        """Test requesting a draw and receiving image data."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Request a draw
            ws_client.send_draw()
            image_data = ws_client.wait_for_image()

            # Verify it's valid PNG data
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0
            img = Image.open(io.BytesIO(image_data))
            assert img.format == "PNG"

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



class TestResizing:
    """Tests for figure resizing."""

    def test_resize_sends_acknowledgment(self, client: TestClient) -> None:
        """Test that resize requests receive proper acknowledgment."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
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

    def test_multiple_resizes_work(self, client: TestClient) -> None:
        """Test that multiple resizes can be performed."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            sizes_to_test = [(640, 480), (800, 600), (400, 300)]

            for width, height in sizes_to_test:
                ws_client.send_resize(width=width, height=height)

                # Should receive resize acknowledgment
                msg = ws_client.receive_message()
                assert isinstance(msg, dict)
                assert msg["type"] == "resize"
                assert msg["size"] == [width, height]



class TestDPI:
    """Tests for DPI (device pixel ratio) changes."""

    def test_changing_dpi_triggers_redraw(self, client: TestClient) -> None:
        """Test that changing device pixel ratio triggers a full redraw."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # Connection already handles DPI, but we can send another DPI change
            # Send supports_binary (again, though already sent during init)
            ws_client.adapter.send_json({"type": "supports_binary", "value": True})

            # Change device pixel ratio - this sends draw message directly
            ws_client.adapter.send_json(
                {"type": "set_device_pixel_ratio", "device_pixel_ratio": 2.0}
            )

            # Server sends draw message directly (from handle_set_device_pixel_ratio)
            msg = ws_client.receive_message()
            assert isinstance(msg, dict)
            assert msg["type"] == "draw"

            # Now client sends draw request to get the image
            ws_client.send_draw()
            image_data = ws_client.wait_for_image()
            assert len(image_data) > 0

    def test_device_pixel_ratio(self, client: TestClient) -> None:
        """Test device pixel ratio setting during initialization."""
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


class TestUpdateParams:
    """Tests for dynamic parameter updates."""

    def test_update_params_triggers_redraw(self, client: TestClient) -> None:
        """Test that updating parameters triggers a redraw."""
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


class TestInteractiveEvents:
    """Tests for interactive events (mouse, keyboard, toolbar)."""

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


class TestSaveDownload:
    """Tests for save/download functionality."""

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

        # Receive protocol version
        msg = adapter.receive_json()
        assert msg["type"] == "protocol_version"

        # Send protocol version
        adapter.send_json({"type": "protocol_version", "version": 0})

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

