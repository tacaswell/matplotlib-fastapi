"""
Interactive sine wave demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/interactive_sine.py
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig, UpdateConfig


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


plots = {
    "interactive_sine": PlotConfig(
        description="Interactive sine wave with live phase adjustment",
        init=InitConfig(function=create_interactive_sine, params_model=SinePlotParams),
        update=UpdateConfig(
            function=update_interactive_sine, params_model=SineUpdateParams
        ),
    ),
}
