"""
Sine wave demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/sine.py
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig


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


plots = {
    "sine": PlotConfig(
        description="Sine wave with configurable frequency, amplitude, and phase",
        init=InitConfig(function=create_sine_plot, params_model=SinePlotParams),
    ),
}
