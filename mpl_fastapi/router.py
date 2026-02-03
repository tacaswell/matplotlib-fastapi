"""FastAPI router factory for matplotlib figure integration.

This module provides the router factory function and response models for
creating mountable FastAPI routers that serve interactive matplotlib plots.

The router supports:
- HTML listing of available plots with parameter forms
- JSON API for programmatic access
- Individual plot viewers with WebSocket communication
- Dynamic parameter passing via query strings
"""

import io
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.figure import Figure
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi.mpl_backend import FastAPICanvas, FastAPIManger

# Set up logging
logger = logging.getLogger(__name__)

# Module-level cache for active figures (connection-scoped)
# Maps connection_id -> (Figure, FastAPICanvas)
_active_figures: dict[str, tuple[Figure, FastAPICanvas]] = {}

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
    update_schema: dict[str, Any] | None = None


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

        # Extract base path from request URL
        base_path = request.url.path.rstrip('/')

        return templates.TemplateResponse(
            "plots_list.html",
            {
                "request": request,
                "plots": plots_info,
                "base_path": base_path,
            },
        )

    # Route: List all available plots (JSON API)
    @router.get("/plots", response_model=PlotsListResponse)
    async def list_plots() -> PlotsListResponse:
        """List all available plots with their parameter schemas."""
        plots_info = {}
        for name, config in plot_generators.items():
            update_schema = None
            if config.update is not None:
                update_schema = config.update.params_model.model_json_schema()

            plots_info[name] = PlotInfo(
                description=config.description,
                parameters=config.init.params_model.model_json_schema(),
                update_schema=update_schema,
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

        # Build WebSocket URI including the router prefix
        ws_uri = f"ws://{request.url.hostname}:{request.url.port}"
        # Extract the prefix from the request path
        # If request path is /plots/plot/name, we want /plots
        path_parts = request.url.path.rstrip('/').split('/')
        base_path = ""
        if len(path_parts) >= 2:
            # Get everything before /plot/name
            prefix_parts = []
            for part in path_parts[1:]:  # Skip empty string from leading /
                if part == 'plot':
                    break
                prefix_parts.append(part)
            if prefix_parts:
                base_path = '/' + '/'.join(prefix_parts)
                ws_uri += base_path

        return templates.TemplateResponse(
            "figure.html",
            {
                "request": request,
                "ws_uri": ws_uri,
                "base_path": base_path,
                "fig_id": plot_name,
                "static_path": static_mount_path,
                "update_params_schema": update_params_schema,
            },
        )

    # Route: Download plot as file
    @router.get("/download/{connection_id}")
    async def download_plot(
        connection_id: str,
        file_format: str = "png",
        dpi: int = 100,
        transparent: bool = False,
    ) -> StreamingResponse:
        """
        Download the current plot as a file in the specified format.

        This endpoint requires an active WebSocket connection to work,
        as it retrieves the figure from the connection-scoped cache.

        Parameters
        ----------
        connection_id : str
            The unique connection ID provided via WebSocket
        file_format : str
            Output format (png, pdf, svg, eps, etc.)
        dpi : int
            DPI for raster formats (default: 100)
        transparent : bool
            Whether to use transparent background (default: False)
        """
        # Validate connection exists
        if connection_id not in _active_figures:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Connection not found. The WebSocket connection may have been closed. "
                    "Please refresh the page and try again."
                ),
            )

        fig, _ = _active_figures[connection_id]

        # Validate format
        supported_formats = ["png", "pdf", "svg", "eps", "ps", "jpg", "jpeg", "tiff", "tif"]
        format_lower = file_format.lower()
        if format_lower not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format '{file_format}'. Supported: {', '.join(supported_formats)}",
            )

        # Determine MIME type
        mime_types = {
            "png": "image/png",
            "pdf": "application/pdf",
            "svg": "image/svg+xml",
            "eps": "application/postscript",
            "ps": "application/postscript",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "tiff": "image/tiff",
            "tif": "image/tiff",
        }
        mime_type = mime_types.get(format_lower, "application/octet-stream")

        # Save figure to BytesIO
        buf = io.BytesIO()
        try:
            # Set DPI for raster formats
            save_kwargs: dict[str, Any] = {"format": format_lower}
            if format_lower in ["png", "jpg", "jpeg", "tiff", "tif"]:
                save_kwargs["dpi"] = dpi
            if format_lower == "png":
                save_kwargs["transparent"] = transparent

            fig.savefig(buf, **save_kwargs)
            buf.seek(0)

            logger.info(f"Generated {format_lower} download for connection {connection_id}")
        except Exception as e:
            logger.error(f"Error generating plot download: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate plot: {str(e)}",
            ) from e

        # Return as streaming response with download headers
        return StreamingResponse(
            buf,
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename=plot.{format_lower}"
            },
        )

    # Route: WebSocket connection for interactive plotting
    @router.websocket("/ws/{plot_name}")
    async def websocket_endpoint(websocket: WebSocket, plot_name: str) -> None:
        """Handle WebSocket connection for a plot."""
        # Validate plot exists BEFORE accepting connection
        if plot_name not in plot_generators:
            logger.warning(
                f"WebSocket connection attempted for unknown plot: {plot_name}"
            )
            await websocket.close(code=1008, reason=f"Unknown plot: {plot_name}")
            return

        await websocket.accept()

        # Generate unique connection ID for this WebSocket session
        connection_id = str(uuid.uuid4())
        logger.debug(f"Generated connection ID: {connection_id}")

        config = plot_generators[plot_name]

        # Parse parameters from query string
        try:
            params = config.init.params_model(**websocket.query_params)
        except ValidationError as e:
            logger.warning(f"Invalid parameters for plot {plot_name}: {e}")
            await websocket.send_json({"type": "error", "message": f"Invalid parameters: {e}"})
            await websocket.close(code=1008, reason=f"Invalid params: {e}")
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

        # Store figure and canvas in cache for download endpoint
        _active_figures[connection_id] = (fig, canvas)
        logger.debug(f"Stored figure in cache with connection ID: {connection_id}")

        # Type narrowing for safety
        if not isinstance(canvas, FastAPICanvas):
            raise TypeError(f"Expected FastAPICanvas, got {type(canvas)}")
        if not isinstance(manager, FastAPIManger):
            raise TypeError(f"Expected FastAPIManger, got {type(manager)}")

        # Initial sync
        await websocket.send_json({"type": "image_mode", "mode": "full"})

        # Send connection ID to client for download functionality
        await websocket.send_json({"type": "connection_id", "id": connection_id})

        # Send toolbar configuration
        toolbar_config = FastAPIManger.get_toolbar_config()
        await websocket.send_json({
            "type": "toolbar_config",
            "items": toolbar_config["toolbar_items"]
        })
        await websocket.send_json({
            "type": "extensions",
            "extensions": toolbar_config["extensions"]
        })
        await websocket.send_json({
            "type": "default_extension",
            "extension": toolbar_config["default_extension"]
        })

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

                # Log all received messages
                logger.debug(f"Received message type='{data.get('type')}' for plot '{plot_name}'")

                try:
                    if data["type"] == "supports_binary":
                        manager.supports_binary = data["value"]
                        logger.debug(f"Set supports_binary={data['value']}")
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
                        # Skip logging for motion events to reduce noise
                        if e_type not in ('motion_notify', 'figure_enter', 'figure_leave'):
                            logger.debug(f"Calling handler for event type '{e_type}'")
                        await handler(data, websocket)

                    # Drain the message queue and send responses
                    queue_size = len(canvas._msg_queue)
                    if queue_size > 0:
                        logger.debug(f"Draining queue with {queue_size} messages")
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

            # Remove from active figures cache
            if connection_id in _active_figures:
                del _active_figures[connection_id]
                logger.debug(f"Removed figure from cache: {connection_id}")

            try:
                manager.destroy()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}", exc_info=True)

    # Route: Serve matplotlib JavaScript
    @router.get("/js/mpl.js", response_class=PlainTextResponse)
    async def get_mpl_js() -> PlainTextResponse:
        """Serve the matplotlib JavaScript bundle (TypeScript-compiled)."""
        js = FastAPIManger.get_javascript()
        return PlainTextResponse(js, headers={"Content-Type": "application/javascript"})

    # Route: Serve embeddable component bundle
    @router.get("/component.js", response_class=PlainTextResponse)
    async def get_component_js() -> PlainTextResponse:
        """Serve the embeddable matplotlib component JavaScript (TypeScript-compiled)."""
        # This now serves the TypeScript-compiled bundle from dist/
        js = FastAPIManger.get_javascript()
        return PlainTextResponse(js, headers={"Content-Type": "application/javascript"})

    # Route: Serve source map for debugging
    @router.get("/component.js.map", response_class=PlainTextResponse)
    async def get_component_js_map() -> PlainTextResponse:
        """Serve the source map for the TypeScript-compiled component."""
        map_path = Path(__file__).parent / "static/js/dist/component.js.map"
        if map_path.exists():
            return PlainTextResponse(
                map_path.read_text(encoding="utf-8"),
                headers={"Content-Type": "application/json"}
            )
        raise HTTPException(status_code=404, detail="Source map not found")

    # Route: Get schema for a specific plot
    @router.get("/api/plots/{plot_name}/schema")
    async def get_plot_schema(
        plot_name: str,
        _: None = Depends(validate_plot_name),
    ) -> dict[str, Any]:
        """Get the parameter schemas for a specific plot."""
        config = plot_generators[plot_name]

        response = {
            "plot_name": plot_name,
            "description": config.description,
            "init_schema": config.init.params_model.model_json_schema(),
            "update_schema": None,
        }

        if config.update is not None:
            response["update_schema"] = config.update.params_model.model_json_schema()

        return response

    return MPLRouter(
        router=router,
        static_files=static_files,
        static_mount_path=static_mount_path,
    )
