"""FastAPI router factory for matplotlib figure integration.

This module provides the router factory function and response models for
creating mountable FastAPI routers that serve interactive matplotlib plots.

The router supports:
- HTML listing of available plots with parameter forms
- JSON API for programmatic access
- Individual plot viewers with WebSocket communication
- Dynamic parameter passing via query strings
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.figure import Figure
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi.mpl_backend import FastAPICanvas, FastAPIManger

# Set up logging
logger = logging.getLogger(__name__)

# Type alias for plot generator functions
PlotGenerator = Callable[[Figure, BaseModel], None]


# Response models for type safety and documentation
class PlotInfo(BaseModel):
    """Information about a single plot."""

    description: str
    parameters: dict[str, Any]


class PlotsListResponse(BaseModel):
    """Response model for the plots list endpoint."""

    plots: dict[str, PlotInfo]


@dataclass
class MPLRouter:
    """Container for matplotlib router and its static assets."""

    router: APIRouter
    static_files: StaticFiles
    static_mount_path: str


def create_mpl_router(
    plot_generators: dict[str, tuple[PlotGenerator, type[BaseModel], str]],
    *,
    template_dir: Path | str | None = None,
    static_mount_path: str = "/mpl-static",
) -> MPLRouter:
    """
    Create a mountable router for matplotlib figures.

    Parameters
    ----------
    plot_generators : dict[str, tuple[PlotGenerator, type[BaseModel], str]]
        Mapping of plot names to (generator_func, params_model, description).
        - generator_func: Callable[[Figure, BaseModel], None] that populates the figure
        - params_model: Pydantic model for parameter validation
        - description: Human-readable description for the plot
    template_dir : Path | str, optional
        Custom template directory (defaults to package templates)
    static_mount_path : str, optional
        URL path for static assets (default: "/mpl-static")

    Returns
    -------
    MPLRouter
        Container with .router (APIRouter), .static_files (StaticFiles),
        and .static_mount_path (str)

    Example
    -------
    >>> class SinePlotParams(BaseModel):
    ...     frequency: float = 1.0
    ...     amplitude: float = 1.0
    >>>
    >>> def create_sine_plot(fig: Figure, params: SinePlotParams) -> None:
    ...     ax = fig.add_subplot(111)
    ...     x = np.linspace(0, 4*np.pi, 200)
    ...     y = params.amplitude * np.sin(params.frequency * x)
    ...     ax.plot(x, y)
    >>>
    >>> mpl = create_mpl_router({
    ...     "sine": (create_sine_plot, SinePlotParams, "Sine wave visualization"),
    ... })
    >>> app.include_router(mpl.router, prefix="/plots")
    >>> app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
    """
    router = APIRouter()

    # Setup templates
    if template_dir is None:
        template_dir = Path(__file__).parent / "templates"
    else:
        template_dir = Path(template_dir)
    templates = Jinja2Templates(directory=str(template_dir))

    # Setup static files
    static_dir = Path(__file__).parent / "static"
    static_files = StaticFiles(directory=str(static_dir))

    # Dependency for validating plot name
    def get_plot_config(plot_name: str) -> tuple[PlotGenerator, type[BaseModel], str]:
        """Dependency to validate and retrieve plot configuration."""
        if plot_name not in plot_generators:
            available = ", ".join(plot_generators.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Plot '{plot_name}' not found. Available plots: {available}",
            )
        return plot_generators[plot_name]

    # Route: HTML plots list (root)
    @router.get("/", response_class=HTMLResponse)
    async def plots_list_html(request: Request) -> HTMLResponse:
        """Render an HTML page listing all available plots."""
        plots_info = {}
        for name, (_, param_model, description) in plot_generators.items():
            plots_info[name] = {
                "description": description,
                "parameters": param_model.model_json_schema(),
            }
        return templates.TemplateResponse(
            "plots_list.html",
            {
                "request": request,
                "plots": plots_info,
            },
        )

    # Route: List all available plots (JSON API)
    @router.get("/plots", response_model=PlotsListResponse)
    async def list_plots() -> PlotsListResponse:
        """List all available plots with their parameter schemas."""
        plots_info = {}
        for name, (_, param_model, description) in plot_generators.items():
            plots_info[name] = PlotInfo(
                description=description,
                parameters=param_model.model_json_schema(),
            )
        return PlotsListResponse(plots=plots_info)

    # Route: View a specific plot
    @router.get("/plot/{plot_name}", response_class=HTMLResponse)
    async def view_plot(
        request: Request,
        plot_name: str,
        plot_config: tuple[PlotGenerator, type[BaseModel], str] = Depends(
            get_plot_config
        ),
    ) -> HTMLResponse:
        """Render the plot viewer HTML."""

        return templates.TemplateResponse(
            "figure.html",
            {
                "request": request,
                "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
                "fig_id": plot_name,
                "static_path": static_mount_path,
            },
        )

    # Route: WebSocket connection for interactive plotting
    @router.websocket("/ws/{plot_name}")
    async def websocket_endpoint(websocket: WebSocket, plot_name: str) -> None:
        """Handle WebSocket connection for a plot."""
        await websocket.accept()

        # Validate plot exists
        if plot_name not in plot_generators:
            logger.warning(
                f"WebSocket connection attempted for unknown plot: {plot_name}"
            )
            await websocket.close(code=1003, reason=f"Unknown plot: {plot_name}")
            return

        generator, param_model, _ = plot_generators[plot_name]

        # Parse parameters from query string
        try:
            params = param_model(**websocket.query_params)
        except ValidationError as e:
            logger.warning(f"Invalid parameters for plot {plot_name}: {e}")
            await websocket.close(code=1003, reason=f"Invalid params: {e}")
            return

        logger.info(f"WebSocket connected for plot '{plot_name}' with params: {params}")

        # Create figure and call generator to populate it
        # Note: generator is synchronous and may block for complex plots
        # For production use with heavy computation, consider:
        # - Using asyncio.to_thread() for CPU-bound operations
        # - Pre-generating figures with a background worker
        # - Adding timeouts for generator execution
        fig = Figure()
        try:
            generator(fig, params)
        except Exception as e:
            logger.error(f"Error generating plot '{plot_name}': {e}", exc_info=True)
            await websocket.close(code=1011, reason=f"Plot generation failed: {e}")
            return

        # Attach FastAPICanvas after figure is populated
        canvas = FastAPICanvas(fig)

        # Attach manager
        manager = FastAPIManger(canvas, 0)

        # Type narrowing for safety
        if not isinstance(canvas, FastAPICanvas):
            raise TypeError(f"Expected FastAPICanvas, got {type(canvas)}")
        if not isinstance(manager, FastAPIManger):
            raise TypeError(f"Expected FastAPIManger, got {type(manager)}")

        # Initial sync
        await websocket.send_json({"type": "image_mode", "mode": "full"})

        # Event loop
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for plot '{plot_name}'")
                    return
                except Exception as e:
                    logger.error(
                        f"Error receiving WebSocket message: {e}", exc_info=True
                    )
                    return

                try:
                    if data["type"] == "supports_binary":
                        manager.supports_binary = data["value"]
                    else:
                        e_type = data["type"]
                        handler = getattr(
                            canvas, f"handle_{e_type}", canvas.handle_unknown_event
                        )
                        await handler(data, websocket)
                    await canvas.drain_queue(websocket)
                except Exception as e:
                    logger.error(
                        f"Error handling event '{data.get('type', 'unknown')}': {e}",
                        exc_info=True,
                    )
                    # Continue processing other events
        finally:
            # Cleanup on disconnect
            logger.debug(f"Cleaning up resources for plot '{plot_name}'")
            try:
                manager.destroy()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)

    # Route: Serve matplotlib JavaScript
    @router.get("/js/mpl.js", response_class=PlainTextResponse)
    async def get_mpl_js() -> PlainTextResponse:
        """Serve the matplotlib JavaScript bundle."""
        js = FastAPIManger.get_javascript()
        return PlainTextResponse(js, headers={"Content-Type": "application/javascript"})

    return MPLRouter(
        router=router,
        static_files=static_files,
        static_mount_path=static_mount_path,
    )
