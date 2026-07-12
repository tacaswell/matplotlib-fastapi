"""
Demo plots for mpl-fastapi.

This file contains only plot definitions — no web-server boilerplate.
Run it with:

    python -m mpl_fastapi demos/plots.py

Then visit the URL printed to the console.
"""

import numpy as np
from typing import Literal
from matplotlib.artist import Artist
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from pydantic import BaseModel, Field

from mpl_fastapi import ConnectionInfo, InitConfig, PlotConfig, UpdateConfig

# ---------------------------------------------------------------------------
# Sine wave
# ---------------------------------------------------------------------------


class SinePlotParams(BaseModel):
    """Parameters for sine wave visualization."""

    frequency: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Frequency of the sine wave"
    )
    amplitude: float = Field(
        default=1.0, ge=0.1, le=5.0, description="Amplitude of the sine wave"
    )
    phase: float = Field(
        default=0.0, ge=0.0, le=6.28, description="Phase shift in radians"
    )
    points: int = Field(
        default=200, ge=50, le=1000, description="Number of points to plot"
    )


def create_sine_plot(fig: Figure, params: SinePlotParams) -> None:
    ax = fig.subplots()
    x = np.linspace(0, 4 * np.pi, params.points)
    y = params.amplitude * np.sin(params.frequency * x + params.phase)
    ax.plot(x, y, linewidth=2, label=f"A={params.amplitude}, f={params.frequency}")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"Sine Wave: y = ${params.amplitude}\sin({params.frequency}x + {params.phase:.2f})$"
    )
    ax.legend()


# ---------------------------------------------------------------------------
# Interactive sine wave (with live updates)
# ---------------------------------------------------------------------------


class SineUpdateParams(BaseModel):
    """Update parameters — only phase can be changed live."""

    phase: float = Field(
        default=0.0, ge=0.0, le=6.28, description="Phase shift in radians"
    )


def create_interactive_sine(fig: Figure, params: SinePlotParams) -> dict:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, params.points)
    y = params.amplitude * np.sin(params.frequency * x)
    (line,) = ax.plot(x, y, linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-params.amplitude * 1.2, params.amplitude * 1.2)
    title = ax.set_title(
        rf"Interactive Sine: $y = {params.amplitude}\sin({params.frequency}x)$"
    )
    return {
        "line": line,
        "title": title,
        "x": x,
        "amplitude": params.amplitude,
        "frequency": params.frequency,
    }


def update_interactive_sine(state: dict, params: SineUpdateParams) -> dict:
    y = state["amplitude"] * np.sin(state["frequency"] * state["x"] + params.phase)
    state["line"].set_ydata(y)
    state["title"].set_text(
        rf"Interactive Sine: $y = {state['amplitude']}\sin({state['frequency']}x + {params.phase:.2f})$"
    )
    return state


# ---------------------------------------------------------------------------
# Cosine wave with optional damping
# ---------------------------------------------------------------------------


class CosinePlotParams(BaseModel):
    frequency: float = Field(default=1.0, ge=0.1, le=10.0)
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0)
    damping: float = Field(default=0.0, ge=0.0, le=1.0, description="Damping factor")


def create_cosine_plot(fig: Figure, params: CosinePlotParams) -> None:
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 200)
    envelope = np.exp(-params.damping * x) if params.damping > 0 else 1.0
    y = params.amplitude * envelope * np.cos(params.frequency * x)
    ax.plot(x, y, linewidth=2, label="Cosine")
    if params.damping > 0:
        ax.plot(x, params.amplitude * envelope, "r--", alpha=0.5, label="Envelope")
        ax.plot(x, -params.amplitude * envelope, "r--", alpha=0.5)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    rhs = rf"{params.amplitude}\cos({params.frequency}x)"
    if params.damping > 0:
        rhs += rf"e^{{-{params.damping}x}}"
    ax.set_title(rf"Cosine Wave: y = ${rhs}$")
    ax.legend()


# ---------------------------------------------------------------------------
# Lissajous curves
# ---------------------------------------------------------------------------


class LissajousParams(BaseModel):
    freq_x: float = Field(default=3.0, ge=1.0, le=10.0, description="X frequency")
    freq_y: float = Field(default=2.0, ge=1.0, le=10.0, description="Y frequency")
    delta: float = Field(default=1.57, ge=0.0, le=6.28, description="Phase difference")


def create_lissajous_plot(fig: Figure, params: LissajousParams) -> None:
    ax = fig.add_subplot(111)
    t = np.linspace(0, 2 * np.pi, 1000)
    x = np.sin(params.freq_x * t + params.delta)
    y = np.sin(params.freq_y * t)
    ax.plot(x, y, linewidth=2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Lissajous Curve: {params.freq_x}:{params.freq_y}")


# ---------------------------------------------------------------------------
# Polygon editor
# Copied from https://matplotlib.org/stable/gallery/event_handling/poly_editor.html
# ---------------------------------------------------------------------------


class PolygonParams(BaseModel): ...


def dist_point_to_segment(p, s0, s1):
    """
    Get the distance from the point *p* to the segment (*s0*, *s1*), where
    *p*, *s0*, *s1* are ``[x, y]`` arrays.
    """
    s01 = s1 - s0
    s0p = p - s0
    if (s01 == 0).all():
        return np.hypot(*s0p)
    # Project onto segment, without going past segment ends.
    p1 = s0 + np.clip((s0p @ s01) / (s01 @ s01), 0, 1) * s01
    return np.hypot(*(p - p1))


class PolygonInteractor:
    """
    A polygon editor.

    Key-bindings

      't' toggle vertex markers on and off.  When vertex markers are on,
          you can move them, delete them

      'd' delete the vertex under point

      'i' insert a vertex at point.  You must be within epsilon of the
          line connecting two existing vertices

    """

    showverts = True
    epsilon = 5  # max pixel distance to count as a vertex hit

    def __init__(self, ax, poly):
        if poly.figure is None:
            raise RuntimeError(
                "You must first add the polygon to a figure "
                "or canvas before defining the interactor"
            )
        self.ax = ax
        canvas = poly.figure.canvas
        self.poly = poly

        x, y = zip(*self.poly.xy, strict=False)
        self.line = Line2D(x, y, marker="o", markerfacecolor="r", animated=True)
        self.ax.add_line(self.line)

        self.cid = self.poly.add_callback(self.poly_changed)
        self._ind = None  # the active vert

        canvas.mpl_connect("draw_event", self.on_draw)
        canvas.mpl_connect("button_press_event", self.on_button_press)
        canvas.mpl_connect("key_press_event", self.on_key_press)
        canvas.mpl_connect("button_release_event", self.on_button_release)
        canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas = canvas

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        # do not need to blit here, this will fire before the screen is
        # updated

    def poly_changed(self, poly):
        """This method is called whenever the pathpatch object is called."""
        # only copy the artist props to the line (except visibility)
        vis = self.line.get_visible()
        Artist.update_from(self.line, poly)
        self.line.set_visible(vis)  # don't use the poly visibility state

    def get_ind_under_point(self, event):
        """
        Return the index of the point closest to the event position or *None*
        if no point is within ``self.epsilon`` to the event position.
        """
        # display coords
        xy = np.asarray(self.poly.xy)
        xyt = self.poly.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]

        if d[ind] >= self.epsilon:
            ind = None

        return ind

    def on_button_press(self, event):
        """Callback for mouse button presses."""
        if not self.showverts:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self._ind = self.get_ind_under_point(event)

    def on_button_release(self, event):
        """Callback for mouse button releases."""
        if not self.showverts:
            return
        if event.button != 1:
            return
        self._ind = None

    def on_key_press(self, event):
        """Callback for key presses."""
        print(f"HERE {event.key} {event.inaxes=}")
        if not event.inaxes:
            return
        if event.key == "t":
            self.showverts = not self.showverts
            self.line.set_visible(self.showverts)
            if not self.showverts:
                self._ind = None
        elif event.key == "d":
            ind = self.get_ind_under_point(event)
            if ind is not None:
                self.poly.xy = np.delete(self.poly.xy, ind, axis=0)
                self.line.set_data(zip(*self.poly.xy, strict=False))
        elif event.key == "i":
            xys = self.poly.get_transform().transform(self.poly.xy)
            p = event.x, event.y  # display coords
            for i in range(len(xys) - 1):
                s0 = xys[i]
                s1 = xys[i + 1]
                d = dist_point_to_segment(p, s0, s1)
                if d <= self.epsilon:
                    self.poly.xy = np.insert(
                        self.poly.xy, i + 1, [event.xdata, event.ydata], axis=0
                    )
                    self.line.set_data(zip(*self.poly.xy, strict=False))
                    break
        print(f"{self.line.stale=}")
        if self.line.stale:
            self.canvas.draw_idle()

    def on_mouse_move(self, event):
        """Callback for mouse movements."""
        if not self.showverts:
            return
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        x, y = event.xdata, event.ydata

        self.poly.xy[self._ind] = x, y
        if self._ind == 0:
            self.poly.xy[-1] = x, y
        elif self._ind == len(self.poly.xy) - 1:
            self.poly.xy[0] = x, y
        self.line.set_data(zip(*self.poly.xy, strict=False))

        self.canvas.restore_region(self.background)
        self.ax.draw_artist(self.poly)
        self.ax.draw_artist(self.line)
        self.canvas.blit(self.ax.bbox)


def create_polygon_demo(fig: Figure, params: PolygonParams) -> PolygonInteractor:

    theta = np.arange(0, 2 * np.pi, 0.1)
    r = 1.5

    xs = r * np.cos(theta)
    ys = r * np.sin(theta)

    poly = Polygon(np.column_stack([xs, ys]), animated=True)

    ax = fig.subplots()
    ax.add_patch(poly)
    p = PolygonInteractor(ax, poly)

    ax.set_title("Click and drag a point to move it")
    ax.set_xlim((-2, 2))
    ax.set_ylim((-2, 2))
    return p


# ---------------------------------------------------------------------------
# Context-aware plot — customized per authenticated user
# ---------------------------------------------------------------------------


class WhoAmIParams(BaseModel):
    """Parameters for the context demo plot."""

    fontsize: int = Field(
        default=16, ge=8, le=48, description="Label font size in points"
    )


def create_whoami_plot(
    fig: Figure, params: WhoAmIParams, *, context: ConnectionInfo
) -> None:
    """Render the identity of the connected principal.

    Opts in to server-side context by declaring a keyword-only ``context``
    parameter.  ``context.principal`` is whatever the router's auth policy
    returned for this connection (``None`` under ``NoAuth``); a real
    AuthN/AuthZ policy would surface the authenticated user here, ready for
    on-behalf-of calls.
    """
    ax = fig.subplots()
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        f"principal: {context.principal!r}",
        ha="center",
        va="center",
        fontsize=params.fontsize,
    )
    ax.text(
        0.5,
        0.4,
        f"connection: {context.connection_id[:8]}…",
        ha="center",
        va="center",
        fontsize=params.fontsize * 0.6,
        color="gray",
    )
    context.logger.info("rendered whoami plot")


# ---------------------------------------------------------------------------
# Dataset Viewer (example with required parameters)
# ---------------------------------------------------------------------------


class DatasetParams(BaseModel):
    """Parameters for dataset visualization - demonstrates required parameters."""

    dataset_name: str = Field(description="Name of the dataset to visualize (required)")
    colormap: Literal["viridis", "magma", "plasma", "inferno"] = Field(
        default="viridis",
        description="Matplotlib colormap name",
    )
    grid_size: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Size of the data grid",
    )


def create_dataset_plot(fig: Figure, params: DatasetParams) -> None:
    """Render a synthetic dataset visualization.

    This plot demonstrates required parameters. The `dataset_name` field has
    no default value, so users must provide it via query string.  When started,
    this plot will show in the logs as:

        dataset_viewer       http://...?token=... (requires: dataset_name)

    Users must navigate to the plots list page or manually provide the parameter:

        http://...?token=...&dataset_name=my_data&colormap=plasma
    """
    import numpy as np

    ax = fig.subplots()

    # Generate synthetic data based on the dataset name (use name as seed)
    seed = sum(ord(c) for c in params.dataset_name) % 1000
    rng = np.random.default_rng(seed)
    data = rng.random((params.grid_size, params.grid_size))

    im = ax.imshow(data, cmap=params.colormap, aspect="auto", interpolation="nearest")
    ax.set_title(f"Dataset: {params.dataset_name}", fontsize=14, pad=10)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    # Add colorbar
    fig.colorbar(im, ax=ax, label="Value")

    # Add text annotation showing the parameters
    ax.text(
        0.02,
        0.98,
        f"Colormap: {params.colormap}\nGrid: {params.grid_size}×{params.grid_size}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )


# ---------------------------------------------------------------------------
# Registry — the only thing mpl_fastapi needs from this file
# ---------------------------------------------------------------------------

plots = {
    "sine": PlotConfig(
        description="Sine wave with configurable frequency, amplitude, and phase",
        init=InitConfig(function=create_sine_plot, params_model=SinePlotParams),
    ),
    "interactive_sine": PlotConfig(
        description="Interactive sine wave with live phase adjustment",
        init=InitConfig(function=create_interactive_sine, params_model=SinePlotParams),
        update=UpdateConfig(
            function=update_interactive_sine, params_model=SineUpdateParams
        ),
    ),
    "cosine": PlotConfig(
        description="Cosine wave with optional exponential damping",
        init=InitConfig(function=create_cosine_plot, params_model=CosinePlotParams),
    ),
    "lissajous": PlotConfig(
        description="Lissajous curves — parametric curves showing harmonic motion",
        init=InitConfig(function=create_lissajous_plot, params_model=LissajousParams),
    ),
    "polygon": PlotConfig(
        description="This is an example to show how to build cross-GUI applications using Matplotlib event handling to interact with objects on the canvas.",
        init=InitConfig(function=create_polygon_demo, params_model=PolygonParams),
    ),
    "whoami": PlotConfig(
        description="Context-aware plot showing the authenticated principal",
        init=InitConfig(function=create_whoami_plot, params_model=WhoAmIParams),
    ),
    "dataset_viewer": PlotConfig(
        description="Dataset viewer with required dataset_name parameter",
        init=InitConfig(function=create_dataset_plot, params_model=DatasetParams),
    ),
}
