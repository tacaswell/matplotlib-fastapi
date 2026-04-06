"""
Demo plots for mpl-fastapi.

This file contains only plot definitions — no web-server boilerplate.
Run it with:

    python -m mpl_fastapi demos/plots.py

Then visit the URL printed to the console.
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig, UpdateConfig


# ---------------------------------------------------------------------------
# Sine wave
# ---------------------------------------------------------------------------

class SinePlotParams(BaseModel):
    """Parameters for sine wave visualization."""

    frequency: float = Field(default=1.0, ge=0.1, le=10.0, description="Frequency of the sine wave")
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0, description="Amplitude of the sine wave")
    phase: float = Field(default=0.0, ge=0.0, le=6.28, description="Phase shift in radians")
    points: int = Field(default=200, ge=50, le=1000, description="Number of points to plot")


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
        fr"Sine Wave: y = ${params.amplitude}\sin({params.frequency}x + {params.phase:.2f})$"
    )
    ax.legend()


# ---------------------------------------------------------------------------
# Interactive sine wave (with live updates)
# ---------------------------------------------------------------------------

class SineUpdateParams(BaseModel):
    """Update parameters — only phase can be changed live."""

    phase: float = Field(default=0.0, ge=0.0, le=6.28, description="Phase shift in radians")


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
        fr"Interactive Sine: $y = {params.amplitude}\sin({params.frequency}x)$"
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
        fr"Interactive Sine: $y = {state['amplitude']}\sin({state['frequency']}x + {params.phase:.2f})$"
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
    rhs = fr"{params.amplitude}\cos({params.frequency}x)"
    if params.damping > 0:
        rhs += fr"e^{{-{params.damping}x}}"
    ax.set_title(fr"Cosine Wave: y = ${rhs}$")
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
        update=UpdateConfig(function=update_interactive_sine, params_model=SineUpdateParams),
    ),
    "cosine": PlotConfig(
        description="Cosine wave with optional exponential damping",
        init=InitConfig(function=create_cosine_plot, params_model=CosinePlotParams),
    ),
    "lissajous": PlotConfig(
        description="Lissajous curves — parametric curves showing harmonic motion",
        init=InitConfig(function=create_lissajous_plot, params_model=LissajousParams),
    ),
}
