"""
Demo application showing how to use mpl-fastapi with a simple sine wave plot.

This demonstrates the new router-based API where plots are generated on-demand
when users connect via WebSocket.

Run with:
    uvicorn demos.sine_wave:app --reload

Then visit:
    http://localhost:8000/plots - List all available plots
    http://localhost:8000/plot/sine?frequency=2.0&amplitude=1.5 - Sine wave with params
"""

import numpy as np
from fastapi import FastAPI
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import create_mpl_router


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
    """
    Generate a sine wave plot.

    Parameters
    ----------
    fig : Figure
        Matplotlib Figure to populate (no pyplot!)
    params : SinePlotParams
        Validated parameters for the plot
    """
    ax = fig.add_subplot(111)

    # Generate data
    x = np.linspace(0, 4 * np.pi, params.points)
    y = params.amplitude * np.sin(params.frequency * x + params.phase)

    # Plot
    ax.plot(x, y, linewidth=2, label=f"A={params.amplitude}, f={params.frequency}")
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        f"Sine Wave: y = {params.amplitude} × sin({params.frequency}x + {params.phase:.2f})"
    )
    ax.legend()


class CosinePlotParams(BaseModel):
    """Parameters for cosine wave visualization."""

    frequency: float = Field(default=1.0, ge=0.1, le=10.0)
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0)
    damping: float = Field(default=0.0, ge=0.0, le=1.0, description="Damping factor")


def create_cosine_plot(fig: Figure, params: CosinePlotParams) -> None:
    """Generate a potentially damped cosine wave plot."""
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
    title = f"Cosine Wave: y = {params.amplitude} × cos({params.frequency}x)"
    if params.damping > 0:
        title += f" × e^(-{params.damping}x)"
    ax.set_title(title)
    ax.legend()


class LissajousParams(BaseModel):
    """Parameters for Lissajous curve."""

    freq_x: float = Field(default=3.0, ge=1.0, le=10.0, description="X frequency")
    freq_y: float = Field(default=2.0, ge=1.0, le=10.0, description="Y frequency")
    delta: float = Field(default=1.57, ge=0.0, le=6.28, description="Phase difference")


def create_lissajous_plot(fig: Figure, params: LissajousParams) -> None:
    """Generate a Lissajous curve."""
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


# Create the matplotlib router
mpl = create_mpl_router(
    {
        "sine": (
            create_sine_plot,
            SinePlotParams,
            "Interactive sine wave with adjustable frequency, amplitude, and phase",
        ),
        "cosine": (
            create_cosine_plot,
            CosinePlotParams,
            "Cosine wave with optional exponential damping",
        ),
        "lissajous": (
            create_lissajous_plot,
            LissajousParams,
            "Lissajous curves - parametric curves showing harmonic motion",
        ),
    }
)

# Create FastAPI app and mount the router
app = FastAPI(
    title="Matplotlib FastAPI Demo",
    description="Interactive matplotlib plots served via FastAPI and WebSockets",
    version="1.0.0",
)

# Mount the matplotlib router
app.include_router(mpl.router, prefix="")

# Mount static files (required for matplotlib JavaScript and CSS)
app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
