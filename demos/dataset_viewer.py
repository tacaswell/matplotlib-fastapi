"""
Dataset viewer demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/dataset_viewer.py
"""

import numpy as np
from typing import Literal
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig


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


plots = {
    "dataset_viewer": PlotConfig(
        description="Dataset viewer with required dataset_name parameter",
        init=InitConfig(function=create_dataset_plot, params_model=DatasetParams),
    ),
}
