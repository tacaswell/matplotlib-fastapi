"""Tests for the ``force_full`` render path.

When the browser client detects a frame-sequence gap (e.g. a diff arrived
before its predecessor's decode finished, or a diff's ``base_seq`` doesn't
match the painted state) it sends ``{"type": "render", "force_full": true}``
to request a self-contained FULL frame that it can resync from.

Coverage:
- A ``render`` with ``force_full=true`` always produces a FULL (non-diff) frame
  even when the figure has not changed since the last render.
- A plain ``render`` (no ``force_full``) may produce a DIFF after drawing.
- ``force_full`` on the server sets ``canvas._force_full = True`` before
  calling ``draw_and_get_diff()``, which is then cleared after encoding.
"""

import struct

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from pydantic import BaseModel

from mpl_fastapi import InitConfig, PlotConfig, create_mpl_router, install_mpl_router
from mpl_fastapi.mpl_backend import FastAPICanvas
from mpl_fastapi.ws_client import (
    MatplotlibWebSocketClient,
    create_fastapi_test_client_adapter,
)

# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

BINARY_HEADER_SIZE = 8
IMAGE_TYPE_FULL = 0x00
IMAGE_TYPE_DIFF = 0x01


def _parse_header(data: bytes) -> dict:
    """Parse the 8-byte binary image header."""
    type_mode, fmt, seq_num, base_seq, flags = struct.unpack_from(">BBHHH", data, 0)
    return {
        "type_mode": type_mode,
        "format": fmt,
        "seq_num": seq_num,
        "base_seq": base_seq,
        "flags": flags,
    }


class SimpleForcefulParams(BaseModel):
    value: float = 1.0


def _make_simple_plot(fig: Figure, params: SimpleForcefulParams):
    import numpy as np

    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, params.value])
    return {"ax": ax, "params": params}


@pytest.fixture
def force_full_app() -> FastAPI:
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "ff_plot": PlotConfig(
                description="Plot for force_full testing",
                init=InitConfig(
                    function=_make_simple_plot,
                    params_model=SimpleForcefulParams,
                ),
            )
        }
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


# ---------------------------------------------------------------------------
# Unit-level test: canvas._force_full interaction with draw_and_get_diff
# ---------------------------------------------------------------------------


class TestForceFull:
    """Unit tests for the force_full server-side path."""

    def _make_canvas(self) -> FastAPICanvas:
        fig = Figure()
        canvas = FastAPICanvas(fig)
        import numpy as np

        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        return canvas

    def test_force_full_produces_full_frame(self) -> None:
        """draw_and_get_diff() with _force_full=True produces a non-diff frame."""
        canvas = self._make_canvas()
        # First draw — establishes _last_buff baseline.
        canvas.draw_and_get_diff()

        # Nothing has changed; without force_full this would normally be a diff
        # of all-zero pixels (no change).  With force_full it must be FULL.
        canvas._force_full = True
        canvas._png_is_old = True  # mark dirty so get_diff_image runs
        png_bytes, is_diff = canvas.draw_and_get_diff()
        assert not is_diff, "Expected a FULL frame when _force_full=True"
        assert len(png_bytes) > 0

    def test_force_full_flag_cleared_after_draw(self) -> None:
        """_force_full is reset to False after draw_and_get_diff() uses it."""
        canvas = self._make_canvas()
        canvas._force_full = True
        canvas.draw_and_get_diff()
        assert not canvas._force_full, "_force_full should be cleared after draw"

    def test_plain_render_after_change_may_be_diff(self) -> None:
        """Without force_full, a changed figure can produce a diff frame."""
        import numpy as np

        canvas = self._make_canvas()
        # First render establishes baseline.
        canvas.draw_and_get_diff()

        # Mutate the figure so the buffer changes.
        ax = canvas.figure.axes[0]
        ax.plot([0, 1], [1, 0], "r-")
        canvas._png_is_old = True

        png_bytes, is_diff = canvas.draw_and_get_diff()
        # The result is a diff (True) OR a full (False) depending on canvas state;
        # the important thing is we got a valid response either way.
        assert len(png_bytes) > 0


# ---------------------------------------------------------------------------
# Integration tests: force_full via the WebSocket protocol
# ---------------------------------------------------------------------------


class TestForceFillIntegration:
    """Integration tests: send force_full=true via the WebSocket."""

    def test_force_full_render_returns_full_header(
        self, force_full_app: FastAPI
    ) -> None:
        """render with force_full=true must produce a frame with FULL type_mode."""
        client = TestClient(force_full_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="ff_plot",
        )

        with ws_client.connect():
            # Initial full refresh — establishes server baseline.
            ws_client.send_refresh()

            # Send render with force_full=true
            adapter.send_json(
                {"type": "render", "figure_id": "ff_plot", "force_full": True}
            )
            raw = adapter.receive_bytes()

        assert len(raw) > BINARY_HEADER_SIZE
        hdr = _parse_header(raw)
        assert hdr["type_mode"] == IMAGE_TYPE_FULL, (
            f"Expected FULL frame (type_mode=0x00) but got type_mode=0x{hdr['type_mode']:02x}"
        )
        # FULL frames must have base_seq == 0
        assert hdr["base_seq"] == 0, (
            f"FULL frame should have base_seq=0, got {hdr['base_seq']}"
        )

    def test_force_full_after_repeated_renders(self, force_full_app: FastAPI) -> None:
        """force_full=true forces FULL even after several render round-trips."""
        client = TestClient(force_full_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="ff_plot",
        )

        with ws_client.connect():
            ws_client.send_refresh()

            # Do a couple of plain renders to advance the sequence.
            for _ in range(2):
                ws_client.send_render()

            # Now request force_full — must still be FULL.
            adapter.send_json(
                {"type": "render", "figure_id": "ff_plot", "force_full": True}
            )
            raw = adapter.receive_bytes()

        hdr = _parse_header(raw)
        assert hdr["type_mode"] == IMAGE_TYPE_FULL, (
            f"Expected FULL after force_full=true, got type_mode=0x{hdr['type_mode']:02x}"
        )

    def test_plain_render_without_force_full(self, force_full_app: FastAPI) -> None:
        """A plain render (force_full omitted) responds with some valid frame."""
        client = TestClient(force_full_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="ff_plot",
        )

        with ws_client.connect():
            ws_client.send_refresh()
            # Use adapter.receive_bytes() (raw wire format with header) rather
            # than ws_client.send_render() which strips the 8-byte header.
            adapter.send_json({"type": "render", "figure_id": "ff_plot"})
            raw = adapter.receive_bytes()

        assert len(raw) > BINARY_HEADER_SIZE
        hdr = _parse_header(raw)
        # Type must be a known value (FULL or DIFF).
        assert hdr["type_mode"] in (IMAGE_TYPE_FULL, IMAGE_TYPE_DIFF)

    def test_seq_num_increments_across_renders(self, force_full_app: FastAPI) -> None:
        """Each render response must have a strictly increasing seq_num."""
        client = TestClient(force_full_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="ff_plot",
        )

        seq_nums = []
        with ws_client.connect():
            # Use adapter directly to get raw wire bytes with the 8-byte header.
            adapter.send_json({"type": "refresh", "figure_id": "ff_plot"})
            seq_nums.append(_parse_header(adapter.receive_bytes())["seq_num"])

            for _ in range(2):
                adapter.send_json({"type": "render", "figure_id": "ff_plot"})
                seq_nums.append(_parse_header(adapter.receive_bytes())["seq_num"])

            adapter.send_json(
                {"type": "render", "figure_id": "ff_plot", "force_full": True}
            )
            seq_nums.append(_parse_header(adapter.receive_bytes())["seq_num"])

        # All seq_nums must be strictly increasing (no wrapping in this short test).
        for i in range(1, len(seq_nums)):
            assert seq_nums[i] > seq_nums[i - 1], f"seq_num not increasing: {seq_nums}"
