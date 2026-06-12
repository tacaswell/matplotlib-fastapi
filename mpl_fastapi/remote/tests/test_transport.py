"""Tests for RemoteTransport — the async WebSocket transport layer.

These tests run a real FastAPI server on a background thread and connect
to it with the async transport.
"""

from __future__ import annotations

import asyncio
import io
import threading
import time
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import uvicorn
from fastapi import FastAPI
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel, Field

from mpl_fastapi import (
    InitConfig,
    PlotConfig,
    UpdateConfig,
    create_mpl_router,
    install_mpl_router,
)
from mpl_fastapi.remote.transport import (
    RemoteTransport,
    ServerConfig,
    build_ws_url,
)
from mpl_fastapi.ws_client import ImageTypeMode, parse_binary_image

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class SimpleParams(BaseModel):
    value: float = Field(default=1.0, ge=0.1, le=10.0)


class UpdateParams(BaseModel):
    phase: float = Field(default=0.0, ge=0.0, le=6.28)


def _create_simple_plot(fig: Figure, params: SimpleParams) -> dict[str, object]:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 100)
    y = params.value * np.sin(x)
    ax.plot(x, y)
    return {"ax": ax, "x": x, "value": params.value}


def _update_simple_plot(
    state: dict[str, object], _params: UpdateParams
) -> dict[str, object]:
    return state


def _make_test_app() -> FastAPI:
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "simple": PlotConfig(
                description="test",
                init=InitConfig(
                    function=_create_simple_plot,
                    params_model=SimpleParams,
                ),
            ),
            "updatable": PlotConfig(
                description="updatable test",
                init=InitConfig(
                    function=_create_simple_plot,
                    params_model=SimpleParams,
                ),
                update=UpdateConfig(
                    function=_update_simple_plot,
                    params_model=UpdateParams,
                ),
            ),
        }
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


@pytest.fixture(scope="module")
def server_url() -> Iterator[str]:
    """Start a real uvicorn server and return its base URL."""
    app = _make_test_app()
    host = "127.0.0.1"
    port = 0  # Will be assigned by uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Run server in a daemon thread
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to start
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start in time")

    # Get the actual port
    sockets = server.servers[0].sockets
    actual_port = sockets[0].getsockname()[1]
    yield f"ws://{host}:{actual_port}/plots"

    server.should_exit = True
    thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _collected_messages() -> tuple[list[bytes], list[dict[str, Any]], list[None]]:
    """Return three lists that callbacks append to."""
    binaries: list[bytes] = []
    jsons: list[dict[str, Any]] = []
    disconnects: list[None] = []
    return binaries, jsons, disconnects


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildWsUrl:
    """Tests for the URL builder helper."""

    def test_simple(self) -> None:
        url = build_ws_url("http://localhost:8000/plots", "sine")
        assert url == "http://localhost:8000/plots/ws/v0/sine"

    def test_with_params(self) -> None:
        url = build_ws_url(
            "ws://host:9000/mpl/",
            "wave",
            {"frequency": 2.0, "amplitude": 1.5},
        )
        assert url.startswith("ws://host:9000/mpl/ws/v0/wave?")
        assert "frequency=2.0" in url
        assert "amplitude=1.5" in url

    def test_trailing_slash_stripped(self) -> None:
        url = build_ws_url("http://host/prefix///", "p")
        assert "/prefix/ws/v0/p" in url

    def test_with_update_params(self) -> None:
        url = build_ws_url(
            "ws://host:9000/plots",
            "interactive_sine",
            {"frequency": 2.0},
            update_params={"phase": 1.57},
        )
        assert url.startswith("ws://host:9000/plots/ws/v0/interactive_sine?")
        assert "frequency=2.0" in url
        assert "_update.phase=1.57" in url

    def test_update_params_only(self) -> None:
        url = build_ws_url(
            "ws://host/plots",
            "wave",
            update_params={"phase": 0.5},
        )
        assert url.startswith("ws://host/plots/ws/v0/wave?")
        assert "_update.phase=0.5" in url


class TestTransportHandshake:
    """Tests for the connect / handshake flow."""

    @pytest.mark.asyncio
    async def test_connect_returns_server_config(self, server_url: str) -> None:
        binaries, jsons, disconnects = _collected_messages()
        url = build_ws_url(server_url, "simple", {"value": 1.0})
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        config = await transport.connect()
        try:
            assert isinstance(config, ServerConfig)
            assert config.connection_id  # non-empty string
            assert config.protocol_version == 0
            assert config.figure_size[0] > 0
            assert config.figure_size[1] > 0
            assert config.figure_dpi > 0
            assert len(config.toolbar_items) > 0
            assert "png" in config.save_formats
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_connect_unknown_plot_raises(self, server_url: str) -> None:
        binaries, jsons, disconnects = _collected_messages()
        url = build_ws_url(server_url, "nonexistent_plot_xyz")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        with pytest.raises(Exception, match=r"rejected|403|close|1008"):
            await transport.connect()

    @pytest.mark.asyncio
    async def test_config_echoes_init_params(self, server_url: str) -> None:
        """Config message should echo back the init params."""
        binaries, jsons, disconnects = _collected_messages()
        url = build_ws_url(server_url, "simple", {"value": 3.5})
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        config = await transport.connect()
        try:
            assert config.init_params == {"value": 3.5}
            assert config.update_params is None
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_config_echoes_update_params(self, server_url: str) -> None:
        """Config message should echo back _update.* params."""
        binaries, jsons, disconnects = _collected_messages()
        url = build_ws_url(
            server_url,
            "updatable",
            {"value": 1.0},
            update_params={"phase": 1.57},
        )
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        config = await transport.connect()
        try:
            assert config.init_params == {"value": 1.0}
            assert config.update_params == {"phase": 1.57}
        finally:
            await transport.disconnect()


class TestTransportMessaging:
    """Tests for sending and receiving messages after handshake."""

    @pytest.mark.asyncio
    async def test_refresh_returns_full_image(self, server_url: str) -> None:
        binaries: list[bytes] = []
        jsons: list[dict[str, Any]] = []
        disconnects: list[None] = []
        url = build_ws_url(server_url, "simple", {"value": 1.0})
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        await transport.connect()
        await transport.start_receive_loop()

        try:
            # Send refresh (requests full image)
            assert transport._ws is not None
            import json

            await transport._ws.send(json.dumps({"type": "refresh"}))

            # Wait for binary image
            for _ in range(50):
                if binaries:
                    break
                await asyncio.sleep(0.05)

            assert len(binaries) >= 1
            header, image_data = parse_binary_image(binaries[0])
            assert header.type_mode == ImageTypeMode.FULL
            assert header.seq_num >= 0
            assert len(image_data) > 0

            # Verify it's a valid PNG
            img = Image.open(io.BytesIO(image_data))
            assert img.size[0] > 0
            assert img.size[1] > 0
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_json_thread_safe(self, server_url: str) -> None:
        """send_json is thread-safe and works from the running loop."""
        binaries: list[bytes] = []
        jsons: list[dict[str, Any]] = []
        disconnects: list[None] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        await transport.connect()
        await transport.start_receive_loop()

        try:
            # Use the thread-safe send_json
            transport.send_json({"type": "refresh"})

            # Wait for image
            for _ in range(50):
                if binaries:
                    break
                await asyncio.sleep(0.05)

            assert len(binaries) >= 1
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_resize_gets_ack(self, server_url: str) -> None:
        """Resize message should produce a resize ack."""
        binaries: list[bytes] = []
        jsons: list[dict[str, Any]] = []
        disconnects: list[None] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        await transport.connect()
        await transport.start_receive_loop()

        try:
            transport.send_json({"type": "resize", "width": 400, "height": 300})

            # Wait for resize ack
            for _ in range(50):
                if any(m.get("type") == "resize" for m in jsons):
                    break
                await asyncio.sleep(0.05)

            resize_msgs = [m for m in jsons if m.get("type") == "resize"]
            assert len(resize_msgs) >= 1
            assert "size" in resize_msgs[0]
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect_callback_fires(self, server_url: str) -> None:
        binaries: list[bytes] = []
        jsons: list[dict[str, Any]] = []
        disconnects: list[None] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
        )
        await transport.connect()
        await transport.start_receive_loop()
        await transport.disconnect()

        # Give the receive loop a moment to fire the callback
        await asyncio.sleep(0.2)
        assert len(disconnects) >= 1


class TestTransportReconnect:
    """Tests for the reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnects_after_server_closes_ws(self, server_url: str) -> None:
        """Transport reconnects when the server closes the WebSocket."""
        binaries: list[bytes] = []
        jsons: list[dict[str, Any]] = []
        disconnects: list[None] = []
        reconnects: list[ServerConfig] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=jsons.append,
            on_disconnect=lambda: disconnects.append(None),
            on_reconnect=reconnects.append,
            reconnect_max_attempts=3,
            reconnect_initial_delay=0.1,
            reconnect_max_delay=0.5,
        )
        await transport.connect()
        await transport.start_receive_loop()

        # Forcibly close the underlying WS to simulate a drop
        assert transport._ws is not None
        await transport._ws.close()

        # Wait for the reconnect to complete
        for _ in range(100):
            if reconnects:
                break
            await asyncio.sleep(0.1)

        try:
            assert len(reconnects) == 1
            assert isinstance(reconnects[0], ServerConfig)
            assert transport.is_connected
            assert len(disconnects) == 0  # Reconnected, so no disconnect
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_reconnect_disabled_when_zero_attempts(self, server_url: str) -> None:
        """No reconnect when reconnect_max_attempts=0."""
        disconnects: list[None] = []
        reconnects: list[ServerConfig] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=lambda _: None,
            on_json=lambda _: None,
            on_disconnect=lambda: disconnects.append(None),
            on_reconnect=reconnects.append,
            reconnect_max_attempts=0,
        )
        await transport.connect()
        await transport.start_receive_loop()

        assert transport._ws is not None
        await transport._ws.close()

        # Wait for disconnect callback
        for _ in range(50):
            if disconnects:
                break
            await asyncio.sleep(0.05)

        assert len(disconnects) == 1
        assert len(reconnects) == 0

    @pytest.mark.asyncio
    async def test_explicit_disconnect_does_not_reconnect(
        self, server_url: str
    ) -> None:
        """Calling disconnect() should not trigger reconnection."""
        reconnects: list[ServerConfig] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=lambda _: None,
            on_json=lambda _: None,
            on_disconnect=lambda: None,
            on_reconnect=reconnects.append,
            reconnect_max_attempts=3,
            reconnect_initial_delay=0.1,
        )
        await transport.connect()
        await transport.start_receive_loop()
        await transport.disconnect()

        # Give time for any spurious reconnect to start
        await asyncio.sleep(0.5)
        assert len(reconnects) == 0

    @pytest.mark.asyncio
    async def test_reconnect_exhausted_calls_on_disconnect(
        self, server_url: str
    ) -> None:
        """After all attempts fail, on_disconnect is called."""
        disconnects: list[None] = []
        reconnects: list[ServerConfig] = []
        # Use a URL that will fail to connect (bad plot name)
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=lambda _: None,
            on_json=lambda _: None,
            on_disconnect=lambda: disconnects.append(None),
            on_reconnect=reconnects.append,
            reconnect_max_attempts=2,
            reconnect_initial_delay=0.05,
            reconnect_max_delay=0.1,
        )
        await transport.connect()
        await transport.start_receive_loop()

        # Forcibly close, then change URL so reconnect fails
        assert transport._ws is not None
        await transport._ws.close()
        transport._url = build_ws_url(server_url, "nonexistent_plot_xyz")

        # Wait for all attempts to be exhausted
        for _ in range(100):
            if disconnects:
                break
            await asyncio.sleep(0.1)

        assert len(disconnects) == 1
        assert len(reconnects) == 0

    @pytest.mark.asyncio
    async def test_reconnect_resumes_message_flow(self, server_url: str) -> None:
        """After reconnect, messages flow again."""
        binaries: list[bytes] = []
        reconnects: list[ServerConfig] = []
        url = build_ws_url(server_url, "simple")
        transport = RemoteTransport(
            url,
            on_binary=binaries.append,
            on_json=lambda _: None,
            on_disconnect=lambda: None,
            on_reconnect=reconnects.append,
            reconnect_max_attempts=3,
            reconnect_initial_delay=0.1,
        )
        await transport.connect()
        await transport.start_receive_loop()

        # Get initial image
        transport.send_json({"type": "refresh"})
        for _ in range(50):
            if binaries:
                break
            await asyncio.sleep(0.05)
        assert len(binaries) >= 1
        initial_count = len(binaries)

        # Force-close WS
        assert transport._ws is not None
        await transport._ws.close()

        # Wait for reconnect
        for _ in range(100):
            if reconnects:
                break
            await asyncio.sleep(0.1)
        assert len(reconnects) == 1

        # Request a new image after reconnect
        transport.send_json({"type": "refresh"})
        for _ in range(50):
            if len(binaries) > initial_count:
                break
            await asyncio.sleep(0.05)

        try:
            assert len(binaries) > initial_count
        finally:
            await transport.disconnect()
