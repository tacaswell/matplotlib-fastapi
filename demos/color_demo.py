"""
Color customization demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/color_demo.py
"""

import numpy as np
from matplotlib.figure import Figure
from pydantic import BaseModel, Field
from pydantic_extra_types.color import Color

from mpl_fastapi import InitConfig, PlotConfig


class ColorDemoParams(BaseModel):
    """Parameters for color customization demo - demonstrates the Color type."""

    line_color: Color = Field(
        default="#1f77b4",
        description="Color for the main line",
    )
    fill_color: Color = Field(
        default="#ff7f0e",
        description="Color for the filled area",
    )
    background_color: Color = Field(
        default="#f0f0f0",
        description="Background color",
    )
    show_grid: bool = Field(
        default=True,
        description="Show grid lines",
    )
    line_width: float = Field(
        default=2.0,
        ge=0.5,
        le=5.0,
        description="Line width in points",
    )


def create_color_demo_plot(fig: Figure, params: ColorDemoParams) -> None:
    """Render a plot demonstrating color customization.

    This plot demonstrates the Color type which uses HTML5 color pickers.
    Colors can be specified as:
    - Named colors: "red", "blue", etc.
    - Hex: "#1f77b4"
    - RGB: "rgb(31, 119, 180)"
    """
    ax = fig.subplots()

    # Set background color
    bg_color = (
        params.background_color.as_hex()
        if hasattr(params.background_color, "as_hex")
        else str(params.background_color)
    )
    ax.set_facecolor(bg_color)

    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)

    # Convert colors
    line_color = (
        params.line_color.as_hex()
        if hasattr(params.line_color, "as_hex")
        else str(params.line_color)
    )
    fill_color = (
        params.fill_color.as_hex()
        if hasattr(params.fill_color, "as_hex")
        else str(params.fill_color)
    )

    # Plot
    ax.plot(x, y1, color=line_color, linewidth=params.line_width, label="sin(x)")
    ax.plot(x, y2, color=fill_color, linewidth=params.line_width, label="cos(x)")
    ax.fill_between(x, y1, y2, alpha=0.3, color=fill_color)

    # Styling
    if params.show_grid:
        ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Color Customization Demo")
    ax.legend()


plots = {
    "color_demo": PlotConfig(
        description="Color customization demo with HTML5 color pickers",
        init=InitConfig(function=create_color_demo_plot, params_model=ColorDemoParams),
    ),
}
