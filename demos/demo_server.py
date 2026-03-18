"""
Demo application showing how to use mpl-fastapi with a simple sine wave plot.

This demonstrates the new router-based API where plots are generated on-demand
when users connect via WebSocket. It also shows the update functionality where
plots can be dynamically updated without recreating the entire figure.

Run with:
    uvicorn demos.demo_server:app --reload

Then visit:
    http://localhost:8000/plots - List all available plots
    http://localhost:8000/plots/plot/sine?frequency=2.0&amplitude=1.5 - Sine wave with params
    http://localhost:8000/plots/plot/interactive_sine - Interactive sine with update controls
    http://localhost:8000/embeddable - Embeddable component demos
"""

import logging
import numpy as np
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import (
    InitConfig,
    PlotConfig,
    UpdateConfig,
    create_mpl_router,
    install_mpl_router,
)

# Configure logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure mpl_fastapi loggers are at DEBUG level
logging.getLogger('mpl_fastapi').setLevel(logging.DEBUG)
logging.getLogger('mpl_fastapi.router').setLevel(logging.DEBUG)
logging.getLogger('mpl_fastapi.mpl_backend').setLevel(logging.DEBUG)


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

    Returns
    -------
    None
        This version doesn't support updates, returns None
    """
    ax = fig.subplots()
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


# Interactive sine wave with update support
class InteractiveSineParams(BaseModel):
    """Initial parameters for interactive sine wave."""

    frequency: float = Field(default=1.0, ge=0.1, le=10.0)
    amplitude: float = Field(default=1.0, ge=0.1, le=5.0)
    points: int = Field(default=200, ge=50, le=1000)


class SineUpdateParams(BaseModel):
    """Update parameters for interactive sine wave - can modify phase."""

    phase: float = Field(
        default=0.0, ge=0.0, le=6.28, description="Phase shift in radians"
    )


def create_interactive_sine(fig: Figure, params: InteractiveSineParams) -> dict:
    """
    Generate an interactive sine wave plot with cached state.

    Returns a dict containing the figure state needed for updates.
    """
    ax = fig.add_subplot(111)

    # Generate data
    x = np.linspace(0, 4 * np.pi, params.points)
    y = params.amplitude * np.sin(params.frequency * x)

    # Create line
    (line,) = ax.plot(x, y, linewidth=2)
    ax.axhline(y=0, color="k", linestyle="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim(-params.amplitude * 1.2, params.amplitude * 1.2)
    title = ax.set_title(
        f"Interactive Sine: y = {params.amplitude} × sin({params.frequency}x)"
    )

    # Return state object that update function will use
    return {
        "line": line,
        "title": title,
        "x": x,
        "amplitude": params.amplitude,
        "frequency": params.frequency,
        "current_phase": 0.0,
    }


def update_interactive_sine(state: dict, params: SineUpdateParams) -> dict:
    """
    Update the interactive sine wave with new phase.

    This function receives the cached state and update parameters,
    modifies the plot, and returns the updated state.
    """
    # Update the line data with new phase
    y = state["amplitude"] * np.sin(state["frequency"] * state["x"] + params.phase)
    state["line"].set_ydata(y)

    # Update title to show current phase
    state["title"].set_text(
        f"Interactive Sine: y = {state['amplitude']} × sin({state['frequency']}x + {params.phase:.2f})"
    )

    # Update and return state
    state["current_phase"] = params.phase
    return state


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


# Create the matplotlib router with PlotConfig
mpl = create_mpl_router(
    {
        "sine": PlotConfig(
            description="sine wave with configure-on-init frequency, amplitude, and phase",
            init=InitConfig(
                function=create_sine_plot,
                params_model=SinePlotParams,
            ),
        ),
        "interactive_sine": PlotConfig(
            description="Interactive sine wave with live phase adjustment",
            init=InitConfig(
                function=create_interactive_sine,
                params_model=InteractiveSineParams,
            ),
            update=UpdateConfig(
                function=update_interactive_sine,
                params_model=SineUpdateParams,
            ),
        ),
        "cosine": PlotConfig(
            description="Cosine wave with optional exponential damping",
            init=InitConfig(
                function=create_cosine_plot,
                params_model=CosinePlotParams,
            ),
        ),
        "lissajous": PlotConfig(
            description="Lissajous curves - parametric curves showing harmonic motion",
            init=InitConfig(
                function=create_lissajous_plot,
                params_model=LissajousParams,
            ),
        ),
    }
)

# Create FastAPI app
app = FastAPI(
    title="Matplotlib FastAPI Demo",
    description="Interactive matplotlib plots served via FastAPI and WebSockets",
    version="1.0.0",
)

# Install the matplotlib router at /plots in a single call.
# This includes the router, mounts static files, and chains the
# shutdown lifespan automatically.
install_mpl_router(app, mpl, prefix="/plots")


# Add route to serve home page
@app.get("/")
async def home():
    """Serve the main menu page."""
    index_path = Path(__file__).parent / "index.html"
    return FileResponse(index_path, media_type="text/html")

# Add route to serve embeddable demo
@app.get("/embeddable")
async def embeddable_demo():
    """Serve the embeddable component demo page."""
    demo_path = Path(__file__).parent / "embeddable_demo.html"
    return FileResponse(demo_path, media_type="text/html")


# Serve React example if the build exists
react_build_path = Path(__file__).parent / "react-example" / "dist"
if react_build_path.exists():
    from starlette.staticfiles import StaticFiles

    # Mount the React build directory
    app.mount(
        "/react-app",
        StaticFiles(directory=react_build_path, html=True),
        name="react_app",
    )
else:
    # Fallback route if React isn't built yet
    @app.get("/react-app")
    @app.get("/react-app/{path:path}")
    async def react_not_built(path: str = ""):
        """Return instructions if React app hasn't been built."""
        from fastapi.responses import HTMLResponse
        return HTMLResponse(
            content="""
            <html>
            <head><title>React Example Not Built</title></head>
            <body style="font-family: sans-serif; padding: 40px;">
                <h1>React Example Not Built</h1>
                <p>The React example hasn't been built yet. To build it:</p>
                <pre style="background: #f4f4f4; padding: 15px; border-radius: 4px;">
# First, build the npm package (from repo root):
npm run build:npm

# Then build the React app:
cd demos/react-example
npm install
npm run build

# Restart the server and refresh this page
                </pre>
            </body>
            </html>
            """,
            status_code=200,
        )
