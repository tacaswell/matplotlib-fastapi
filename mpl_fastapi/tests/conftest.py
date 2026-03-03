"""Pytest fixtures and configuration for mpl_fastapi tests."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig, UpdateConfig, create_mpl_router


class SimpleParams(BaseModel):
    """Simple test parameters."""

    value: float = Field(default=1.0, ge=0.1, le=10.0)


class UpdateParams(BaseModel):
    """Update parameters for testing."""

    phase: float = Field(default=0.0, ge=0.0, le=6.28)


def create_simple_plot(fig: Figure, params: SimpleParams) -> dict[str, object]:
    """Create a simple test plot.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to populate
    params : SimpleParams
        Plot parameters

    Returns
    -------
    dict
        State dictionary with plot elements
    """
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 100)
    y = params.value * np.sin(x)
    (line,) = ax.plot(x, y, label=f"value={params.value}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Simple Test Plot")
    ax.legend()
    ax.grid(True)
    return {"ax": ax, "line": line, "x": x, "params": params}


def update_simple_plot(
    state: dict[str, object], params: UpdateParams
) -> dict[str, object]:
    """Update the simple plot.

    Parameters
    ----------
    state : dict
        Current plot state
    params : UpdateParams
        Update parameters

    Returns
    -------
    dict
        Updated state dictionary
    """

    x = state["x"]
    line = state["line"]
    plot_params = state["params"]

    # Update the line data with phase shift
    y = plot_params.value * np.sin(x + params.phase)  # type: ignore[attr-defined, operator]
    line.set_ydata(y)  # type: ignore[attr-defined]

    # Update title
    ax = state["ax"]
    ax.set_title(f"Simple Test Plot (phase={params.phase:.2f})")  # type: ignore[attr-defined]

    return state


@pytest.fixture
def simple_plot_config() -> PlotConfig:
    """Simple plot configuration without update."""
    return PlotConfig(
        description="Simple test plot for basic functionality",
        init=InitConfig(
            function=create_simple_plot,
            params_model=SimpleParams,
        ),
    )


@pytest.fixture
def updatable_plot_config() -> PlotConfig:
    """Plot configuration with update support."""
    return PlotConfig(
        description="Updatable test plot",
        init=InitConfig(
            function=create_simple_plot,
            params_model=SimpleParams,
        ),
        update=UpdateConfig(
            function=update_simple_plot,
            params_model=UpdateParams,
        ),
    )


@pytest.fixture
def test_app(
    simple_plot_config: PlotConfig, updatable_plot_config: PlotConfig
) -> FastAPI:
    """FastAPI app with test routers."""
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "simple": simple_plot_config,
            "updatable": updatable_plot_config,
        }
    )
    app.include_router(mpl.router, prefix="/plots")
    app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
    return app


@pytest.fixture
def client(test_app: FastAPI) -> TestClient:
    """Test client for making requests."""
    return TestClient(test_app)


def create_interactive_plot(fig: Figure, params: SimpleParams) -> dict[str, object]:
    """Create a plot with interactive callbacks for testing.

    This plot registers callbacks that modify the plot when events occur,
    allowing us to verify that events are actually processed.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to populate
    params : SimpleParams
        Plot parameters

    Returns
    -------
    dict
        State dictionary with plot elements and callback tracking
    """
    ax = fig.add_subplot(111)
    x = np.linspace(0, 4 * np.pi, 100)
    y = params.value * np.sin(x)
    (line,) = ax.plot(x, y, "b-", label=f"value={params.value}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Interactive Test Plot")
    ax.legend()
    ax.grid(True)

    # Track events via a marker that we can modify
    (marker,) = ax.plot([], [], "ro", markersize=10, label="Last Click")

    # State to track callback invocations
    state = {
        "ax": ax,
        "line": line,
        "marker": marker,
        "x": x,
        "params": params,
        "click_count": 0,
        "last_click_pos": None,
    }

    # Callback for button press events
    def on_button_press(event):
        if event.inaxes == ax:
            state["click_count"] += 1
            state["last_click_pos"] = (event.xdata, event.ydata)
            # Update marker position to show the click
            marker.set_data([event.xdata], [event.ydata])
            # Update title to show click count
            ax.set_title(f"Interactive Test Plot (clicks: {state['click_count']})")
            fig.canvas.draw_idle()

    # Register the callback
    fig.canvas.mpl_connect("button_press_event", on_button_press)

    return state


@pytest.fixture
def interactive_plot_config() -> PlotConfig:
    """Plot configuration with interactive callbacks."""
    return PlotConfig(
        description="Interactive test plot with callbacks",
        init=InitConfig(
            function=create_interactive_plot,
            params_model=SimpleParams,
        ),
    )


@pytest.fixture
def test_app_with_interactive(
    simple_plot_config: PlotConfig,
    updatable_plot_config: PlotConfig,
    interactive_plot_config: PlotConfig,
) -> FastAPI:
    """FastAPI app with test routers including interactive plot."""
    app = FastAPI()
    mpl = create_mpl_router(
        {
            "simple": simple_plot_config,
            "updatable": updatable_plot_config,
            "interactive": interactive_plot_config,
        }
    )
    app.include_router(mpl.router, prefix="/plots")
    app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
    return app
