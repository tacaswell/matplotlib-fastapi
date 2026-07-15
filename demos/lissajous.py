"""
Lissajous curves demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/lissajous.py
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig


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


plots = {
    "lissajous": PlotConfig(
        description="Lissajous curves — parametric curves showing harmonic motion",
        init=InitConfig(function=create_lissajous_plot, params_model=LissajousParams),
    ),
}
