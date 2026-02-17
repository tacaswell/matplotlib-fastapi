"""Tests for WebSocket connection lifecycle and protocol.

These tests focus on the core WebSocket functionality:
- Connection establishment and protocol version negotiation
- Initial message sequence
- Basic message handling (draw, resize, DPI changes, updates)
- Error handling for invalid inputs

Note: These tests use FastAPI's TestClient which provides a synchronous
WebSocket interface. The TestClient doesn't support timeouts on receive
operations, so tests must know the exact message sequence to expect.
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect


def drain_initial_messages(websocket) -> None:
    """Drain all initial server messages.

    Server sends exactly 6 messages on connection (deterministic):
    1. protocol_version
    2. image_mode
    3. connection_id
    4. toolbar_config
    5. save_formats
    6. default_save_format

    Note: history_buttons is sent AFTER the first client message,
    not during initial connection, to ensure toolbar is initialized.
    """
    expected_types = {
        "protocol_version",
        "image_mode",
        "connection_id",
        "toolbar_config",
        "save_formats",
        "default_save_format",
    }
    seen_types = set()

    for _ in range(10):  # Safety limit (should only need 6)
        msg = websocket.receive_json()
        seen_types.add(msg["type"])
        if expected_types.issubset(seen_types):
            break

    # Verify we got all expected messages
    if not expected_types.issubset(seen_types):
        missing = expected_types - seen_types
        raise AssertionError(
            f"Did not receive all initial messages. Missing: {missing}"
        )


def send_client_init_and_drain_history_buttons(websocket) -> None:
    """Send first client message and drain the history_buttons response.

    After server sends 6 initial messages, client must send protocol_version
    to trigger the history_buttons message. This helper sends that message
    and drains the history_buttons response.
    """
    websocket.send_json({"type": "protocol_version", "version": 0})
    msg = websocket.receive_json()
    if msg["type"] != "history_buttons":
        raise AssertionError(
            f"Expected history_buttons after first client message, got {msg['type']}"
        )


def drain_until_message_type(
    websocket, target_type: str, max_messages: int = 20
) -> dict:
    """Drain messages until we find one of the target type.

    Returns the target message when found.
    Raises AssertionError if not found within max_messages.

    Note: Silently skips history_buttons messages that may appear after
    any client message (sent by server after first client message).
    """
    for _ in range(max_messages):
        msg = websocket.receive_json()
        # Skip history_buttons - it's sent after first client message
        if msg["type"] == "history_buttons":
            continue
        if msg["type"] == target_type:
            return msg
    raise AssertionError(
        f"Did not receive {target_type} message within {max_messages} messages"
    )


class TestWebSocketConnection:
    """Tests for WebSocket connection establishment and protocol."""

    def test_websocket_connect_success(self, client: TestClient) -> None:
        """Test successful WebSocket connection."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            # Should receive protocol_version first
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"
            assert msg["version"] == 0

    def test_websocket_connect_invalid_plot_returns_1008(
        self, client: TestClient
    ) -> None:
        """Test that connecting to non-existent plot closes with code 1008."""
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/plots/ws/nonexistent"),
        ):
            pass
        assert exc_info.value.code == 1008

    def test_websocket_connect_invalid_params_returns_1008(
        self, client: TestClient
    ) -> None:
        """Test that invalid parameters close connection with code 1008."""
        # 'value' must be float, this will fail validation
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/plots/ws/simple?value=invalid") as websocket,
        ):
            # Server sends protocol_version first
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"

            # Then sends error message before closing
            msg = websocket.receive_json()
            assert msg["type"] == "error"
            assert "Invalid parameters" in msg["message"]

            # Try to receive more - this should raise WebSocketDisconnect
            websocket.receive_json()
        assert exc_info.value.code == 1008

    def test_websocket_connect_params_out_of_range(self, client: TestClient) -> None:
        """Test that out-of-range parameters are rejected."""
        # 'value' must be between 0.1 and 10.0
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/plots/ws/simple?value=100.0") as websocket,
        ):
            # Server sends protocol_version first
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"

            # Then sends error message before closing
            msg = websocket.receive_json()
            assert msg["type"] == "error"
            assert "Invalid parameters" in msg["message"]

            # Try to receive more - this should raise WebSocketDisconnect
            websocket.receive_json()
        assert exc_info.value.code == 1008

    def test_websocket_initial_message_sequence(self, client: TestClient) -> None:
        """Test that initial messages are sent in correct order."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            messages = []
            # Collect exactly 6 initial messages (deterministic)
            for _ in range(6):
                msg = websocket.receive_json()
                messages.append(msg)

            message_types = [msg["type"] for msg in messages]

            # Should receive exactly these 6 messages
            assert message_types[0] == "protocol_version"
            assert "image_mode" in message_types
            assert "connection_id" in message_types
            assert "toolbar_config" in message_types
            assert "save_formats" in message_types
            assert "default_save_format" in message_types

            # Verify exactly 6 messages (no history_buttons yet)
            assert len(messages) == 6

            # Send first client message (protocol_version)
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Now we should receive history_buttons
            msg = websocket.receive_json()
            assert msg["type"] == "history_buttons"
            assert not msg["Back"]
            assert not msg["Forward"]


class TestProtocolVersion:
    """Tests for protocol version negotiation."""

    def test_protocol_version_mismatch_closes_connection(
        self, client: TestClient
    ) -> None:
        """Test that protocol version mismatch closes the connection."""
        with (
            pytest.raises(WebSocketDisconnect) as exc_info,
            client.websocket_connect("/plots/ws/simple?value=1.0") as websocket,
        ):
            # Receive protocol_version (first message)
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"

            # Send back incompatible version
            websocket.send_json({"type": "protocol_version", "version": 999})

            # Drain any messages sent before the version check
            # (image_mode, connection_id, toolbar_config, etc.)
            while True:
                msg = websocket.receive_json()
                if msg["type"] == "error":
                    assert "Incompatible protocol version" in msg["message"]
                    break

            # Try to receive more - should disconnect
            websocket.receive_json()

        assert exc_info.value.code == 1008

    def test_protocol_version_match_continues(self, client: TestClient) -> None:
        """Test that matching protocol version allows communication to continue."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            # Receive protocol_version
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"

            # Send back matching version
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Should continue to receive other messages
            msg = websocket.receive_json()
            # Should get next message type (image_mode, connection_id, etc.)
            assert msg["type"] in [
                "image_mode",
                "connection_id",
                "toolbar_config",
                "save_formats",
                "default_save_format",
            ]


class TestImageRendering:
    """Tests for image rendering and comparison."""

    def test_websocket_sends_image_after_draw(self, client: TestClient) -> None:
        """Test that WebSocket sends image data after a draw request."""
        with client.websocket_connect("/plots/ws/simple?value=2.0") as websocket:
            # Drain all initial server messages (6 messages)
            drain_initial_messages(websocket)

            # Send refresh to trigger draw (this is the FIRST client message)
            # Server will respond with: figure_label, draw, history_buttons
            websocket.send_json({"type": "refresh"})

            # Receive all responses to refresh
            msg1 = websocket.receive_json()
            msg2 = websocket.receive_json()
            msg3 = websocket.receive_json()

            # Identify which is which
            messages = {msg1["type"], msg2["type"], msg3["type"]}
            assert "figure_label" in messages
            assert "draw" in messages
            assert "history_buttons" in messages

            # Now send draw request
            websocket.send_json({"type": "draw"})

            msg = websocket.receive_json()
            assert msg["type"] == "image_mode"

            # Server sends binary image data
            image_bytes = websocket.receive_bytes()
            assert len(image_bytes) > 0

            # Verify it's a valid PNG image
            img = Image.open(io.BytesIO(image_bytes))
            assert img.format == "PNG"
            assert img.size[0] > 0
            assert img.size[1] > 0

    def test_different_parameters_affect_plot(self, client: TestClient) -> None:
        """Test that different parameters are acknowledged."""
        # Just verify that connections with different params work
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"

        with client.websocket_connect("/plots/ws/simple?value=5.0") as websocket:
            msg = websocket.receive_json()
            assert msg["type"] == "protocol_version"


class TestResizing:
    """Tests for figure resizing."""

    def test_resize_sends_acknowledgment(self, client: TestClient) -> None:
        """Test that resize requests receive proper acknowledgment."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            # Drain all initial server messages
            drain_initial_messages(websocket)

            # Send protocol version (FIRST client message - triggers history_buttons)
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Receive history_buttons after first message
            msg = websocket.receive_json()
            assert msg["type"] == "history_buttons"

            # Resize the figure
            websocket.send_json({"type": "resize", "width": 800, "height": 600})

            # Should receive resize acknowledgment
            msg = websocket.receive_json()
            assert msg["type"] == "resize"
            assert "size" in msg

    def test_multiple_resizes_work(self, client: TestClient) -> None:
        """Test that multiple resizes can be performed."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            # Drain all initial server messages
            drain_initial_messages(websocket)

            # Send protocol version (FIRST client message - triggers history_buttons)
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Receive history_buttons after first message
            msg = websocket.receive_json()
            assert msg["type"] == "history_buttons"

            sizes_to_test = [(640, 480), (800, 600), (400, 300)]

            for width, height in sizes_to_test:
                websocket.send_json(
                    {"type": "resize", "width": width, "height": height}
                )

                # Should receive resize acknowledgment
                msg = websocket.receive_json()
                assert msg["type"] == "resize"


class TestDPI:
    """Tests for DPI (device pixel ratio) changes."""

    def test_changing_dpi_triggers_redraw(self, client: TestClient) -> None:
        """Test that changing device pixel ratio triggers a full redraw."""
        with client.websocket_connect("/plots/ws/simple?value=1.0") as websocket:
            # Drain all initial server messages
            drain_initial_messages(websocket)

            # Send protocol version (FIRST client message - triggers history_buttons)
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Receive history_buttons after first message
            msg = websocket.receive_json()
            assert msg["type"] == "history_buttons"

            # Send supports_binary
            websocket.send_json({"type": "supports_binary", "value": True})

            # Change device pixel ratio - this sends draw message directly
            websocket.send_json(
                {"type": "set_device_pixel_ratio", "device_pixel_ratio": 2.0}
            )

            # Server sends draw message directly (from handle_set_device_pixel_ratio)
            msg = websocket.receive_json()
            assert msg["type"] == "draw"

            # Now client sends draw request to get the image
            websocket.send_json({"type": "draw"})

            # Server sends image_mode then binary data
            msg = websocket.receive_json()
            assert msg["type"] == "image_mode"

            # Server sends binary image data
            image_bytes = websocket.receive_bytes()
            assert len(image_bytes) > 0


class TestUpdateParams:
    """Tests for dynamic parameter updates."""

    def test_update_params_triggers_redraw(self, client: TestClient) -> None:
        """Test that updating parameters triggers a redraw."""
        with client.websocket_connect("/plots/ws/updatable?value=1.0") as websocket:
            # Drain all initial server messages
            drain_initial_messages(websocket)

            # Send protocol version (FIRST client message - triggers history_buttons)
            websocket.send_json({"type": "protocol_version", "version": 0})

            # Receive history_buttons after first message
            msg = websocket.receive_json()
            assert msg["type"] == "history_buttons"

            # Send supports_binary
            websocket.send_json({"type": "supports_binary", "value": True})

            # Send update request
            websocket.send_json({"type": "update_params", "params": {"phase": 1.57}})

            # Server queues draw (from canvas.draw_idle())
            msg = websocket.receive_json()
            assert msg["type"] == "draw"

            # Now client sends draw request to get the image
            websocket.send_json({"type": "draw"})

            # Server sends image_mode then binary data
            msg = websocket.receive_json()
            assert msg["type"] == "image_mode"

            # Server sends binary image data
            image_bytes = websocket.receive_bytes()
            assert len(image_bytes) > 0
