"""
Cosine wave demo with optional damping for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/cosine.py
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig


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


plots = {
    "cosine": PlotConfig(
        description="Cosine wave with optional exponential damping",
        init=InitConfig(function=create_cosine_plot, params_model=CosinePlotParams),
    ),
}
