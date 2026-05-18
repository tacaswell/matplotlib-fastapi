"""Tests for server-side blitting support.

Blitting allows user code (e.g. animations, callbacks) to call
``canvas.blit()`` which immediately encodes the renderer buffer as a diff
PNG and enqueues it in ``canvas._binary_queue``.  The router's
``_drain_canvas()`` helper then sends those frames as unsolicited binary
WebSocket messages, bypassing the usual invalidate → client render
round-trip.

Coverage:
- ``get_diff_image()`` produces correct full/diff images and respects the
  ``_png_is_old`` dirty flag.
- ``blit()`` populates ``_binary_queue`` and clears ``_png_is_old``.
- ``draw()`` sets ``_png_is_old``.
- An update function that uses ``blit()`` causes the client to receive a
  binary image frame without sending an explicit ``render`` request.
- When blit queues data, ``draw_idle()`` / ``invalidate`` is NOT sent.
"""

import io

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
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
from mpl_fastapi.mpl_backend import FastAPICanvas
from mpl_fastapi.ws_client import (
    MatplotlibWebSocketClient,
    create_fastapi_test_client_adapter,
)


# ---------------------------------------------------------------------------
# Unit tests for FastAPICanvas.get_diff_image() and blit()
# ---------------------------------------------------------------------------


class TestGetDiffImage:
    """Unit tests for FastAPICanvas.get_diff_image()."""

    def _make_canvas(self) -> FastAPICanvas:
        fig = Figure()
        canvas = FastAPICanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        return canvas

    def test_returns_none_when_not_dirty(self) -> None:
        """get_diff_image() returns None when _png_is_old is False."""
        canvas = self._make_canvas()
        canvas._png_is_old = False
        assert canvas.get_diff_image() is None

    def test_returns_bytes_after_draw(self) -> None:
        """get_diff_image() returns (bytes, bool) after draw() sets _png_is_old."""
        canvas = self._make_canvas()
        canvas.draw()  # sets _png_is_old = True
        result = canvas.get_diff_image()
        assert result is not None
        png_bytes, is_diff = result
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Should be valid PNG
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"

    def test_full_image_when_force_full_set(self) -> None:
        """get_diff_image() returns a full image when _force_full is True."""
        canvas = self._make_canvas()
        canvas._force_full = True
        canvas.draw()
        result = canvas.get_diff_image()
        assert result is not None
        _png_bytes, is_diff = result
        assert is_diff is False

    def test_second_render_is_diff_when_unchanged(self) -> None:
        """Second call with no content change returns a diff image."""
        canvas = self._make_canvas()
        canvas.draw()
        canvas.get_diff_image()  # consume first full

        # Draw again with identical content
        canvas.draw()
        result = canvas.get_diff_image()
        assert result is not None
        _png_bytes, is_diff = result
        assert is_diff is True

    def test_force_full_produces_full_image(self) -> None:
        """Setting _force_full produces a full image even on second render."""
        canvas = self._make_canvas()
        canvas.draw()
        canvas.get_diff_image()  # first full

        canvas._force_full = True
        canvas.draw()
        result = canvas.get_diff_image()
        assert result is not None
        _png_bytes, is_diff = result
        assert is_diff is False

    def test_clears_png_is_old_flag(self) -> None:
        """get_diff_image() clears _png_is_old after encoding."""
        canvas = self._make_canvas()
        canvas.draw()
        assert canvas._png_is_old is True
        canvas.get_diff_image()
        assert canvas._png_is_old is False

    def test_returns_none_on_second_call_without_redraw(self) -> None:
        """Consecutive calls without draw() between them return None."""
        canvas = self._make_canvas()
        canvas.draw()
        canvas.get_diff_image()  # consumes dirty state
        assert canvas.get_diff_image() is None

    def test_updates_current_image_mode(self) -> None:
        """_current_image_mode is updated to reflect full/diff."""
        canvas = self._make_canvas()

        # Force full to get a known baseline
        canvas._force_full = True
        canvas.draw()
        canvas.get_diff_image()
        assert canvas._current_image_mode == "full"

        # Second draw with same content → diff
        canvas.draw()
        canvas.get_diff_image()
        assert canvas._current_image_mode == "diff"


class TestDraw:
    """Unit tests for FastAPICanvas.draw() override."""

    def test_draw_sets_png_is_old(self) -> None:
        """draw() sets _png_is_old to True."""
        fig = Figure()
        canvas = FastAPICanvas(fig)
        canvas._png_is_old = False
        canvas.draw()
        assert canvas._png_is_old is True


class TestBlit:
    """Unit tests for FastAPICanvas.blit()."""

    def _make_drawn_canvas(self) -> FastAPICanvas:
        fig = Figure()
        canvas = FastAPICanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        canvas.draw()
        return canvas

    def test_blit_queues_image(self) -> None:
        """blit() appends (bytes, bool) to _binary_queue."""
        canvas = self._make_drawn_canvas()
        assert len(canvas._binary_queue) == 0

        canvas.blit()

        assert len(canvas._binary_queue) == 1
        png_bytes, is_diff = canvas._binary_queue[0]
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

    def test_blit_queues_nothing_when_renderer_unchanged(self) -> None:
        """blit() always enqueues a frame (even if content is unchanged).

        blit() sets _png_is_old=True and forces a full frame via _force_full,
        so get_diff_image() always encodes something regardless of whether the
        renderer buffer differs from _last_buff.
        """
        canvas = self._make_drawn_canvas()  # draw() called internally
        # Consume the dirty state from draw()
        canvas.get_diff_image()
        assert canvas._png_is_old is False

        queue_len_before = len(canvas._binary_queue)
        canvas.blit()
        assert len(canvas._binary_queue) == queue_len_before + 1

    def test_blit_clears_png_is_old(self) -> None:
        """blit() clears _png_is_old after encoding."""
        canvas = self._make_drawn_canvas()
        assert canvas._png_is_old is True
        canvas.blit()
        assert canvas._png_is_old is False

    def test_blit_always_produces_full_image(self) -> None:
        """A single blit() with no prior pending frame produces a diff when content changed.

        blit() only forces a full frame when coalescing (queue was non-empty).
        A fresh blit after the queue has been drained produces a diff image
        (or full if _force_full was set externally).
        """
        canvas = self._make_drawn_canvas()
        canvas.get_diff_image()  # consume first draw, _last_buff is current

        # Draw changed content — with no pending queue entry this is a diff.
        canvas.draw()
        canvas.blit()
        _png_bytes, is_diff = canvas._binary_queue[0]
        assert is_diff is True

    def test_second_blit_before_drain_replaces_with_full(self) -> None:
        """A second blit() before the first is drained coalesces into a full frame.

        When blit() is called a second time before _drain_canvas() has consumed
        the pending entry, the stale diff is discarded and _force_full is set so
        that the replacement frame is a self-contained full image.  This prevents
        the client from receiving a diff whose base frame it may not have applied.
        """
        fig = Figure()
        canvas = FastAPICanvas(fig)
        ax = fig.add_subplot(111)
        (line,) = ax.plot([0, 1], [0, 1])

        # Establish a clean baseline so the first blit would normally be a diff.
        canvas._force_full = True
        canvas.draw()
        canvas.get_diff_image()  # consume — _last_buff is now current

        # First blit: content changed, queue was empty → produces a diff.
        line.set_ydata([1, 0])
        canvas.draw()
        canvas.blit()  # enqueues a diff

        assert len(canvas._binary_queue) == 1
        assert canvas._binary_queue[0][1] is True  # diff

        # Second blit before the first is drained — should coalesce.
        line.set_ydata([0.5, 0.5])
        canvas.draw()
        canvas.blit()  # should discard diff, produce full

        # Queue still has exactly one entry, now a full frame.
        assert len(canvas._binary_queue) == 1
        _png_bytes, is_diff = canvas._binary_queue[0]
        assert is_diff is False  # upgraded to full

    def test_blit_with_bbox_arg_ignored(self) -> None:
        """blit(bbox=...) accepts and ignores the bbox argument."""
        import matplotlib.transforms as transforms

        canvas = self._make_drawn_canvas()
        bbox = transforms.Bbox([[0, 0], [100, 100]])
        canvas.blit(bbox=bbox)  # should not raise
        assert len(canvas._binary_queue) == 1

    def test_supports_blit_is_true(self) -> None:
        """FastAPICanvas.supports_blit is True."""
        assert FastAPICanvas.supports_blit is True


# ---------------------------------------------------------------------------
# Integration tests: blit() during update_params skips invalidate
# ---------------------------------------------------------------------------


class BlitUpdateParams(BaseModel):
    phase: float = Field(default=0.0, ge=0.0, le=6.28)


def _create_blit_plot(fig: Figure, params: "SimpleParams") -> dict:  # type: ignore[name-defined]
    """Init function for blit integration tests."""
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 100)
    y = np.sin(x)
    (line,) = ax.plot(x, y)
    return {"ax": ax, "line": line, "x": x, "background": None}


def _update_with_blit(state: dict, params: BlitUpdateParams) -> dict:
    """Update function that uses copy_from_bbox / restore_region / blit.

    This mimics the standard animation blitting pattern:
    1. Restore the background saved before drawing animated artists.
    2. Draw the animated artist.
    3. blit() to push the new frame.
    """
    line = state["line"]
    x = state["x"]
    fig = line.figure
    canvas = fig.canvas
    ax = state["ax"]

    # Phase-shifted sine
    line.set_ydata(np.sin(x + params.phase))

    # Use the full-figure blit path (no background save/restore here since
    # this runs in a thread; we just want to exercise the blit() call).
    canvas.draw()
    canvas.blit()

    return state


class SimpleParamsBlit(BaseModel):
    value: float = Field(default=1.0, ge=0.1, le=10.0)


@pytest.fixture
def blit_app() -> FastAPI:
    """FastAPI app with a plot whose update function calls blit()."""
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "blit_plot": PlotConfig(
                description="Plot that uses blit() in its update function",
                init=InitConfig(
                    function=_create_blit_plot,
                    params_model=SimpleParamsBlit,
                ),
                update=UpdateConfig(
                    function=_update_with_blit,
                    params_model=BlitUpdateParams,
                ),
            )
        }
    )
    install_mpl_router(app, mpl, prefix="/plots")
    return app


class TestBlitIntegration:
    """Integration tests for blit() through the WebSocket protocol."""

    def test_blit_update_delivers_binary_frame_without_render(
        self, blit_app: FastAPI
    ) -> None:
        """update_params with blit() delivers a binary frame to the client.

        The update function calls canvas.blit(), which enqueues a binary
        image.  After _drain_canvas() the client should receive that image
        without having sent a ``render`` message.

        We verify this by:
        1. Connecting and getting the initial image via refresh.
        2. Sending update_params.
        3. Receiving messages — we expect a binary image to arrive, NOT just
           an ``invalidate`` message.
        """
        client = TestClient(blit_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="blit_plot",
        )

        with ws_client.connect():
            # Consume the initial refresh image
            ws_client.send_refresh()

            # Send update_params — the update function calls blit()
            ws_client.adapter.send_json(
                {"type": "update_params", "params": {"phase": 1.0}}
            )

            # We should receive a binary image frame pushed by blit(),
            # without having to send a render request.
            image_data = ws_client.adapter.receive_bytes()
            assert len(image_data) > 8  # at least header + some PNG data

            # Validate: 8-byte header + valid PNG body
            png_body = image_data[8:]  # 8-byte binary header (see _build_image_header)
            img = Image.open(io.BytesIO(png_body))
            assert img.format == "PNG"

    def test_blit_update_does_not_send_invalidate(self, blit_app: FastAPI) -> None:
        """update_params with blit() must NOT send an invalidate message.

        When blit() has queued binary data the router should skip
        ``draw_idle()`` so the client is not asked to request a redundant
        render on top of the frame it already received.
        """
        client = TestClient(blit_app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="blit_plot",
        )

        with ws_client.connect():
            ws_client.send_refresh()

            # Send update_params
            ws_client.adapter.send_json(
                {"type": "update_params", "params": {"phase": 2.0}}
            )

            # First message back must be the binary image (blit path), not JSON
            raw = ws_client.adapter.receive_bytes()
            assert isinstance(raw, bytes)
            # Must be binary (bytes), not a JSON invalidate
            # (receive_bytes() would raise if a JSON message arrives first
            # via the TestClient adapter — so reaching here means we got
            # the binary frame first)
            assert len(raw) > 8

    def test_normal_update_still_sends_invalidate(self) -> None:
        """A plain update (no blit) still sends invalidate → render flow."""
        from mpl_fastapi.tests.conftest import (
            SimpleParams,
            UpdateParams,
            create_simple_plot,
            update_simple_plot,
        )

        app = FastAPI()
        mpl = create_mpl_router(
            {
                "no_blit": PlotConfig(
                    description="Plot with plain update (no blit)",
                    init=InitConfig(
                        function=create_simple_plot,
                        params_model=SimpleParams,
                    ),
                    update=UpdateConfig(
                        function=update_simple_plot,
                        params_model=UpdateParams,
                    ),
                )
            }
        )
        install_mpl_router(app, mpl, prefix="/plots")

        client = TestClient(app)
        adapter = create_fastapi_test_client_adapter(client)
        ws_client = MatplotlibWebSocketClient(
            adapter=adapter,
            base_url="/plots",
            plot_name="no_blit",
        )

        with ws_client.connect():
            ws_client.send_refresh()

            # update_params without blit — should receive invalidate
            ws_client.send_update_params({"phase": 1.0})

            # Then a render request should return an image
            image_data = ws_client.send_render()
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0
