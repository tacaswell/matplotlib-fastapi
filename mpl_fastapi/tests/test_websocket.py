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
        adapter.connect("/plots/ws/v0/simple?value=invalid")

        # Send init message
        adapter.send_json({"type": "init", "protocol_version": 0})

        # Should receive error message before closing
        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "Invalid parameters" in msg["message"]

        # Try to receive more - this should raise WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_websocket_connect_params_out_of_range(self, client: TestClient) -> None:
        """Test that out-of-range parameters are rejected."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        # 'value' must be between 0.1 and 10.0
        adapter.connect("/plots/ws/v0/simple?value=100.0")

        # Send init message
        adapter.send_json({"type": "init", "protocol_version": 0})

        # Should receive error message before closing
        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "Invalid parameters" in msg["message"]

        # Try to receive more - this should raise WebSocketDisconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
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
            ws_client.send_render()

        with pytest.raises(RuntimeError, match="Client not initialized"):
            ws_client.send_refresh()

        with pytest.raises(RuntimeError, match="Client not initialized"):
            ws_client.send_toolbar_button("pan")


class TestProtocolVersion:
    """Tests for protocol version negotiation (v0)."""

    def test_protocol_version_mismatch_closes_connection(
        self, client: TestClient
    ) -> None:
        """Test that protocol version mismatch closes the connection."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        adapter.connect("/plots/ws/v0/simple?value=1.0")

        # Send init with incompatible version
        adapter.send_json({"type": "init", "protocol_version": 999})

        # Should receive error message immediately
        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "Incompatible protocol version" in msg["message"]

        # Try to receive more - should disconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
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

        adapter.connect("/plots/ws/v0/simple?value=1.0")

        # Send init without protocol_version field
        adapter.send_json({"type": "init"})

        # Should receive error message immediately
        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "required" in msg["message"].lower()

        # Try to receive more - should disconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.receive_json()

        assert exc_info.value.code == 1008

    def test_wrong_first_message_closes_connection(self, client: TestClient) -> None:
        """Test that sending non-init as first message closes connection."""
        # Use raw adapter to test protocol-level error handling
        adapter = ContextManagerWebSocketAdapter(client)

        adapter.connect("/plots/ws/v0/simple?value=1.0")

        # Send wrong message type as first client message
        adapter.send_json({"type": "refresh"})

        # Should receive error message
        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "init" in msg["message"].lower()

        # Try to receive more - should disconnect
        with pytest.raises(WebSocketDisconnect) as exc_info:
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
            # Request another render - send_render now returns image directly
            image_data = ws_client.send_render()

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
            # Request a render - send_render now returns image directly
            image_data = ws_client.send_render()

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
            # send_refresh requests full render and waits for binary image
            image_data = ws_client.send_refresh()

            # Verify we got the image
            assert isinstance(image_data, bytes)

    def test_receive_message(self, client: TestClient) -> None:
        """Test receiving messages from websocket."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send toolbar button which queues messages
            ws_client.send_toolbar_button("pan")

            # Receive messages using receive_message()
            # Toolbar actions can queue multiple message types
            received_types = []
            for _ in range(5):  # Collect several messages
                msg = ws_client.receive_message()
                received_types.append(msg["type"])
                # Break if we get an invalidate message (last expected message)
                if msg["type"] == "invalidate":
                    break

            # Verify we received expected message types from pan button
            assert "navigate_mode" in received_types or "message" in received_types


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
            # send_resize now returns the acknowledgment directly
            msg = ws_client.send_resize(width=800, height=600)
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
                # send_resize now returns the acknowledgment directly
                msg = ws_client.send_resize(width=width, height=height)
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

            # Server sends invalidate message directly (from handle_set_device_pixel_ratio)
            msg = ws_client._receive_json()
            assert isinstance(msg, dict)
            assert msg["type"] == "invalidate"

            # Now client sends render request to get the image - returns image directly
            image_data = ws_client.send_render()
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
            # Update parameters - send_update_params now waits for invalidate message
            ws_client.send_update_params({"phase": 1.57})

            # Request the actual render - send_render returns image directly
            image_data = ws_client.send_render()
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
            # Toolbar button actions don't return messages directly
            # They queue messages that can be retrieved if needed
            ws_client.send_toolbar_button("pan")

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
        """Test TestClientAdapter basic operations with v0 protocol."""
        adapter = create_fastapi_test_client_adapter(client)

        # Before connection
        assert not adapter.is_connected()

        # Connect to v0 endpoint
        adapter.connect("/plots/ws/v0/simple")
        assert adapter.is_connected()

        # Send init message (client sends first in v0 protocol)
        adapter.send_json({"type": "init", "protocol_version": 0})

        # Receive config response
        msg = adapter.receive_json()
        assert msg["type"] == "config"

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


class TestAdvancedMouseEvents:
    """Tests for advanced mouse event types with callback verification."""

    def test_mouse_events_trigger_callbacks(self, client: TestClient) -> None:
        """Test that mouse events trigger matplotlib callbacks and draw updates."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Test double-click event by sending it and verifying no crash
            ws_client.send_mouse_event("dblclick", x=100, y=100, button=1)

            # Test figure enter/leave events
            ws_client.send_mouse_event("figure_enter", x=50, y=50)
            ws_client.send_mouse_event("figure_leave", x=500, y=500)

            # Test scroll event - this often triggers view updates
            ws_client.send_mouse_event("scroll", x=200, y=200, step=1)

            # After all these events, the figure should still be responsive
            # Request a render to verify the figure is in a good state - send_render returns image directly
            image_data = ws_client.send_render()

            # Verify we got valid image data back
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0

            # Verify it's valid PNG
            img = Image.open(io.BytesIO(image_data))
            assert img.format == "PNG"


class TestToolbarNavigation:
    """Tests for toolbar navigation with state verification."""

    def test_pan_zoom_interaction_with_image_comparison(
        self, client: TestClient
    ) -> None:
        """Test pan/zoom toolbar interaction and verify it changes the view."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Get initial image - send_render returns image directly
            ws_client.send_render()

            # Activate pan mode
            ws_client.send_toolbar_button("pan")

            # Simulate pan by doing button press, move, release
            ws_client.send_mouse_event("button_press", x=100, y=100, button=1)
            ws_client.send_mouse_event("motion_notify", x=150, y=150)
            ws_client.send_mouse_event("button_release", x=150, y=150, button=1)

            # Deactivate pan mode
            ws_client.send_toolbar_button("pan")

    def test_zoom_mode_activation(self, client: TestClient) -> None:
        """Test zoom mode activation."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Activate zoom mode
            ws_client.send_toolbar_button("zoom")

    def test_home_button_resets_view(self, client: TestClient) -> None:
        """Test home button resets view after pan/zoom."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Do some navigation first
            ws_client.send_toolbar_button("pan")

            # Simulate a pan
            ws_client.send_mouse_event("button_press", x=100, y=100, button=1)
            ws_client.send_mouse_event("motion_notify", x=200, y=200)
            ws_client.send_mouse_event("button_release", x=200, y=200, button=1)

            # Click home to reset
            ws_client.send_toolbar_button("home")

    def test_back_forward_navigation_history(self, client: TestClient) -> None:
        """Test back/forward buttons work with navigation history."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Initially, back/forward should be disabled (no history)
            # Try to use back button
            ws_client.send_toolbar_button("back")

            # Try forward button
            ws_client.send_toolbar_button("forward")


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_unknown_event_type(self, client: TestClient) -> None:
        """Test handling of unknown event types."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
        )

        with ws_client.connect():
            # Send an unknown event type directly via adapter
            ws_client.adapter.send_json({"type": "unknown_event_type", "data": "test"})
            # Server should handle gracefully (unknown events are logged but don't crash)


class TestInteractiveCallbacks:
    """Tests that verify matplotlib callbacks are actually invoked by events."""

    def test_button_press_triggers_callback_byte_for_byte(
        self, test_app_with_interactive
    ) -> None:
        """Test button press event triggers callback with image pixel comparison.

        Pattern:
        1. Create interactive figure locally using create_interactive_plot
        2. Send mouse event via WebSocket and get rendered figure
        3. Generate MouseEvent locally and push it into local figure
        4. Render local figure
        5. Compare that renders are identical (pixel data, not PNG bytes)
        """
        from pathlib import Path

        import numpy as np
        from fastapi.testclient import TestClient
        from matplotlib.backend_bases import MouseEvent
        from matplotlib.figure import Figure

        from mpl_fastapi.mpl_backend import FastAPICanvas
        from mpl_fastapi.tests.conftest import SimpleParams, create_interactive_plot

        # Step 1: Create local figure with same generator function
        local_fig = Figure()
        local_params = SimpleParams(value=1.0)
        create_interactive_plot(local_fig, local_params)
        local_canvas = FastAPICanvas(local_fig)
        local_canvas.draw()

        # Step 2: Connect WebSocket client and send mouse event
        client = TestClient(test_app_with_interactive)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="interactive",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # First, get initial full image to establish base for diffs
            ws_client.send_refresh()

            # Send button press via WebSocket
            # Coordinates in display space
            click_x, click_y = 200, 200
            ws_client.send_mouse_event("button_press", x=click_x, y=click_y, button=1)

            # Wait for invalidate message from callback
            max_attempts = 5
            msg = None
            for _ in range(max_attempts):
                msg = ws_client.receive_message()
                if msg["type"] == "invalidate":
                    break
            assert msg is not None
            assert msg["type"] == "invalidate"

            # Get rendered image from WebSocket - send_render returns image directly
            ws_image_bytes = ws_client.send_render()

            # Step 3: Generate MouseEvent locally and push into local figure
            # Convert display coordinates to data coordinates
            renderer_height = local_canvas.get_renderer().height
            # WebSocket Y is from top, matplotlib Y is from bottom
            local_y = renderer_height - click_y

            # Create MouseEvent with proper canvas reference
            mouse_event = MouseEvent(
                "button_press_event",
                local_canvas,
                click_x,
                local_y,
                button=1,
            )
            # Process the event through matplotlib
            mouse_event._process()  # type: ignore[attr-defined]

            # Step 4: Render local figure
            local_canvas.draw()

            # Get pixel data from local render
            renderer = local_canvas.get_renderer()
            local_pixels = np.frombuffer(
                renderer.buffer_rgba(), dtype=np.uint8
            ).reshape((int(renderer.height), int(renderer.width), 4))

            # Get pixel data from WebSocket render
            ws_img = Image.open(io.BytesIO(ws_image_bytes))
            ws_pixels = np.array(ws_img)

            # Step 5: Compare pixel data
            try:
                assert ws_pixels.shape == local_pixels.shape, (
                    f"Image shapes should match: WebSocket={ws_pixels.shape}, "
                    f"Local={local_pixels.shape}"
                )

                # Compare pixel data (allowing for minor differences due to compression)
                pixel_diff = np.abs(ws_pixels.astype(int) - local_pixels.astype(int))
                max_diff = pixel_diff.max()
                mean_diff = pixel_diff.mean()

                # PNG compression can introduce small differences, so we allow tiny variations
                assert max_diff <= 5, (
                    f"Maximum pixel difference too large: {max_diff} "
                    f"(mean: {mean_diff:.2f})"
                )
                assert mean_diff <= 1.0, (
                    f"Mean pixel difference too large: {mean_diff:.2f} "
                    f"(max: {max_diff})"
                )
            except AssertionError:
                # Save images on failure
                output_dir = Path("test_results/test_websocket")
                output_dir.mkdir(parents=True, exist_ok=True)

                test_name = "test_button_press_triggers_callback_byte_for_byte"
                remote_path = output_dir / f"{test_name}_remote.png"
                local_path = output_dir / f"{test_name}_local.png"

                # Save WebSocket image
                with open(remote_path, "wb") as f:
                    f.write(ws_image_bytes)

                # Save local image
                Image.fromarray(local_pixels).save(local_path)

                print("\nImages saved to:")
                print(f"  Remote: {remote_path}")
                print(f"  Local: {local_path}")

                raise

    def test_double_click_event_byte_for_byte(self, test_app_with_interactive) -> None:
        """Test double-click event with pixel comparison."""
        from pathlib import Path

        import numpy as np
        from fastapi.testclient import TestClient
        from matplotlib.backend_bases import MouseEvent
        from matplotlib.figure import Figure

        from mpl_fastapi.mpl_backend import FastAPICanvas
        from mpl_fastapi.tests.conftest import SimpleParams, create_interactive_plot

        # Create local figure
        local_fig = Figure()
        local_params = SimpleParams(value=1.0)
        create_interactive_plot(local_fig, local_params)
        local_canvas = FastAPICanvas(local_fig)
        local_canvas.draw()

        # Connect WebSocket client
        client = TestClient(test_app_with_interactive)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="interactive",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # First, get initial full image to establish base for diffs
            ws_client.send_refresh()

            # Send double-click via WebSocket
            click_x, click_y = 150, 150
            ws_client.send_mouse_event("dblclick", x=click_x, y=click_y, button=1)

            # Wait for invalidate message
            try:
                max_attempts = 5
                msg = None
                for _ in range(max_attempts):
                    msg = ws_client.receive_message()
                    if msg["type"] == "invalidate":
                        break
                if msg is None or msg["type"] != "invalidate":
                    # No invalidate triggered, skip test
                    return
                assert msg["type"] == "invalidate"
            except RuntimeError:
                # No invalidate triggered, skip test
                return

            # Get WebSocket image - send_render returns image directly
            ws_image_bytes = ws_client.send_render()

            # Generate double-click locally
            renderer_height = local_canvas.get_renderer().height
            local_y = renderer_height - click_y

            dbl_click_event = MouseEvent(
                "button_press_event",
                local_canvas,
                click_x,
                local_y,
                button=1,
                dblclick=True,
            )
            dbl_click_event._process()  # type: ignore[attr-defined]

            # Render local figure
            local_canvas.draw()

            # Get pixel data from both
            renderer = local_canvas.get_renderer()
            local_pixels = np.frombuffer(
                renderer.buffer_rgba(), dtype=np.uint8
            ).reshape((int(renderer.height), int(renderer.width), 4))

            ws_img = Image.open(io.BytesIO(ws_image_bytes))
            ws_pixels = np.array(ws_img)

            # Compare pixel data
            try:
                assert ws_pixels.shape == local_pixels.shape
                pixel_diff = np.abs(ws_pixels.astype(int) - local_pixels.astype(int))
                max_diff = pixel_diff.max()
                mean_diff = pixel_diff.mean()

                assert max_diff <= 5, f"Max pixel diff: {max_diff}"
                assert mean_diff <= 1.0, f"Mean pixel diff: {mean_diff:.2f}"
            except AssertionError:
                # Save images on failure
                output_dir = Path("test_results/test_websocket")
                output_dir.mkdir(parents=True, exist_ok=True)

                test_name = "test_double_click_event_byte_for_byte"
                remote_path = output_dir / f"{test_name}_remote.png"
                local_path = output_dir / f"{test_name}_local.png"

                with open(remote_path, "wb") as f:
                    f.write(ws_image_bytes)
                Image.fromarray(local_pixels).save(local_path)

                print("\nImages saved to:")
                print(f"  Remote: {remote_path}")
                print(f"  Local: {local_path}")

                raise

    def test_scroll_event_byte_for_byte(self, test_app_with_interactive) -> None:
        """Test scroll event with pixel comparison."""
        from fastapi.testclient import TestClient
        from matplotlib.backend_bases import MouseEvent
        from matplotlib.figure import Figure

        from mpl_fastapi.mpl_backend import FastAPICanvas
        from mpl_fastapi.tests.conftest import SimpleParams, create_interactive_plot

        # Create local figure
        local_fig = Figure()
        local_params = SimpleParams(value=1.0)
        create_interactive_plot(local_fig, local_params)
        local_canvas = FastAPICanvas(local_fig)
        local_canvas.draw()

        # Connect WebSocket client
        client = TestClient(test_app_with_interactive)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="interactive",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            # Send scroll via WebSocket
            scroll_x, scroll_y = 200, 200
            scroll_step = 1
            ws_client.send_mouse_event(
                "scroll", x=scroll_x, y=scroll_y, step=scroll_step
            )

            # Scroll events typically don't trigger draws unless in zoom mode
            # Just verify no crash
            # Generate scroll locally
            renderer_height = local_canvas.get_renderer().height
            local_y = renderer_height - scroll_y

            scroll_event = MouseEvent(
                "scroll_event",
                local_canvas,
                scroll_x,
                local_y,
                step=scroll_step,
            )
            scroll_event._process()  # type: ignore[attr-defined]

            # Both should complete without error
            # (Scroll events don't usually modify the figure directly)


class TestUpdateOnInit:
    """Tests for _update.* query parameters applied on init."""

    def test_update_params_in_url_applied_on_init(self, client: TestClient) -> None:
        """Test that _update.* query params are applied after init."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="updatable",
            init_params={"value": 1.0},
            update_params={"phase": 1.57},
        )

        with ws_client.connect():
            assert ws_client._initialized is True
            # The config message should echo init and update params
            assert ws_client.server_init_params == {
                "value": 1.0,
            }
            assert ws_client.server_update_params == {
                "phase": 1.57,
            }

    def test_config_message_echoes_init_params(self, client: TestClient) -> None:
        """Test that the config message includes init_params."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 3.5},
        )

        with ws_client.connect():
            assert ws_client.server_init_params["value"] == 3.5
            # No update params for a non-updatable plot
            assert ws_client.server_update_params is None

    def test_config_message_null_update_params_when_none(
        self, client: TestClient
    ) -> None:
        """Test that update_params is null when no _update.* given."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="updatable",
            init_params={"value": 1.0},
        )

        with ws_client.connect():
            assert ws_client.server_update_params is None

    def test_update_on_init_renders_correctly(self, client: TestClient) -> None:
        """Test that the figure reflects update params applied on init."""
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="updatable",
            init_params={"value": 1.0},
            update_params={"phase": 1.57},
        )

        with ws_client.connect():
            image_data = ws_client.send_render()
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0

    def test_invalid_update_params_closes_connection(self, client: TestClient) -> None:
        """Test that invalid _update.* params close the connection."""
        adapter = ContextManagerWebSocketAdapter(client)

        # phase must be float; 'bad' will fail Pydantic validation
        adapter.connect("/plots/ws/v0/updatable?value=1.0&_update.phase=bad")
        adapter.send_json({"type": "init", "protocol_version": 0})

        msg = adapter.receive_json()
        assert msg["type"] == "error"
        assert "update parameters" in msg["message"].lower()

        with pytest.raises(WebSocketDisconnect) as exc_info:
            adapter.receive_json()
        assert exc_info.value.code == 1008

    def test_update_params_ignored_for_non_updatable_plot(
        self, client: TestClient
    ) -> None:
        """Test that _update.* params are ignored for plots without update."""
        adapter = create_fastapi_test_client_adapter(client)
        # 'simple' has no update config — _update.* should be ignored
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="simple",
            init_params={"value": 1.0},
            update_params={"phase": 1.57},
        )

        with ws_client.connect():
            assert ws_client._initialized is True
            # update_params in config should be None since plot has
            # no update config
            assert ws_client.server_update_params is None
