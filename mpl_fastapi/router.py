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

# Type aliases for plot generator and update functions
# Using Any for parameters to support subclasses of BaseModel
PlotGenerator = Callable[[Figure, Any], Any]
UpdateFunction = Callable[[Any, Any], Any]


@dataclass
class InitConfig:
    """Initial plot generation configuration."""

    function: PlotGenerator
    params_model: type[BaseModel]


@dataclass
class UpdateConfig:
    """Plot update configuration."""

    function: UpdateFunction
    params_model: type[BaseModel]


@dataclass
class PlotConfig:
    """Configuration for a plot with optional update capability."""

    description: str
    init: InitConfig
    update: UpdateConfig | None = None


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
    plot_generators: dict[str, PlotConfig],
    *,
    template_dir: Path | str | None = None,
    static_mount_path: str = "/mpl-static",
) -> MPLRouter:
    """
    Create a mountable router for matplotlib figures.

    Parameters
    ----------
    plot_generators : dict[str, PlotConfig]
        Mapping of plot names to PlotConfig objects.
        Each PlotConfig contains:
        - description: Human-readable description
        - init: InitConfig with generator function and params model
        - update: Optional UpdateConfig with update function and params model
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
    >>> class SineUpdateParams(BaseModel):
    ...     phase: float = 0.0
    >>>
    >>> def create_sine_plot(fig: Figure, params: SinePlotParams) -> dict[str, Any]:
    ...     ax = fig.add_subplot(111)
    ...     x = np.linspace(0, 4*np.pi, 200)
    ...     y = params.amplitude * np.sin(params.frequency * x)
    ...     line, = ax.plot(x, y)
    ...     return {"ax": ax, "line": line, "params": params}
    >>>
    >>> def update_sine_plot(state: dict[str, Any], params: SineUpdateParams) -> dict[str, Any]:
    ...     x = np.linspace(0, 4*np.pi, 200)
    ...     y = state["params"].amplitude * np.sin(
    ...         state["params"].frequency * x + params.phase
    ...     )
    ...     state["line"].set_ydata(y)
    ...     return state
    >>>
    >>> mpl = create_mpl_router({
    ...     "sine": PlotConfig(
    ...         description="Interactive sine wave",
    ...         init=InitConfig(create_sine_plot, SinePlotParams),
    ...         update=UpdateConfig(update_sine_plot, SineUpdateParams),
    ...     ),
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
    def validate_plot_name(plot_name: str) -> None:
        """Dependency to validate plot name exists."""
        if plot_name not in plot_generators:
            available = ", ".join(plot_generators.keys())
            raise HTTPException(
                status_code=404,
                detail=f"Plot '{plot_name}' not found. Available plots: {available}",
            )

    # Route: HTML plots list (root)
    @router.get("/", response_class=HTMLResponse)
    async def plots_list_html(request: Request) -> HTMLResponse:
        """Render an HTML page listing all available plots."""
        plots_info = {}
        for name, config in plot_generators.items():
            plots_info[name] = {
                "description": config.description,
                "parameters": config.init.params_model.model_json_schema(),
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
        for name, config in plot_generators.items():
            plots_info[name] = PlotInfo(
                description=config.description,
                parameters=config.init.params_model.model_json_schema(),
            )
        return PlotsListResponse(plots=plots_info)

    # Route: View a specific plot
    @router.get("/plot/{plot_name}", response_class=HTMLResponse)
    async def view_plot(
        request: Request,
        plot_name: str,
        _: None = Depends(validate_plot_name),
    ) -> HTMLResponse:
        """Render the plot viewer HTML."""
        config = plot_generators[plot_name]

        # Check if update functionality is available
        update_params_schema = None
        if config.update is not None:
            update_params_schema = config.update.params_model.model_json_schema()

        return templates.TemplateResponse(
            "figure.html",
            {
                "request": request,
                "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
                "fig_id": plot_name,
                "static_path": static_mount_path,
                "update_params_schema": update_params_schema,
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

        config = plot_generators[plot_name]

        # Parse parameters from query string
        try:
            params = config.init.params_model(**websocket.query_params)
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
            state = config.init.function(fig, params)
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
                    elif data["type"] == "update_params":
                        # Handle update request
                        if config.update is None:
                            logger.warning(
                                f"Update requested for plot '{plot_name}' "
                                "but no update function configured"
                            )
                        else:
                            try:
                                # Validate update parameters
                                update_params = config.update.params_model(
                                    **data["params"]
                                )
                                logger.info(
                                    f"Updating plot '{plot_name}' with params: {update_params}"
                                )

                                # Call update function and get new state
                                state = config.update.function(state, update_params)

                                # Trigger redraw
                                canvas.draw_idle()

                            except ValidationError as e:
                                logger.warning(
                                    f"Invalid update parameters for plot {plot_name}: {e}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Error updating plot '{plot_name}': {e}",
                                    exc_info=True,
                                )
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
