"""FastAPI router factory for matplotlib figure integration.

This module provides the router factory function and response models for
creating mountable FastAPI routers that serve interactive matplotlib plots.

The router supports:
- HTML listing of available plots with parameter forms
- JSON API for programmatic access
- Individual plot viewers with WebSocket communication
- Dynamic parameter passing via query strings
"""

import asyncio
import io
import logging
import uuid
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
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

# Thread pool for blocking figure operations
# Using a module-level executor to share threads across all routers
_figure_executor: ThreadPoolExecutor | None = None
_executor_max_workers = 4  # Configurable number of worker threads

# Configuration for saved file cache
SAVE_CACHE_TTL = timedelta(hours=1)  # Files expire after 1 hour
SAVE_CACHE_MAX_SIZE = 100  # Max files to keep in cache


@dataclass
class SavedFile:
    """Represents a saved figure file.

    Attributes
    ----------
    file_id : str
        Unique ID for this specific save
    connection_id : str
        WebSocket connection that created it
    data : bytes
        The actual file data
    format : str
        File format (png, pdf, etc.)
    filename : str
        Suggested download filename
    created_at : datetime
        Timestamp for TTL
    metadata : dict[str, Any]
        Optional metadata (DPI, dimensions, etc.)
    """

    file_id: str
    connection_id: str
    data: bytes
    format: str
    filename: str
    created_at: datetime
    metadata: dict[str, Any]


# Module-level cache for saved files (replaces _active_figures)
# Maps file_id -> SavedFile
_saved_files: dict[str, SavedFile] = {}
# Maps connection_id -> set of file_ids for cleanup
_connection_files: dict[str, set[str]] = defaultdict(set)

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


def _get_figure_executor() -> ThreadPoolExecutor:
    """Get or create the thread pool executor for figure operations.

    Returns
    -------
    ThreadPoolExecutor
        Shared thread pool for blocking figure operations
    """
    global _figure_executor
    if _figure_executor is None:
        _figure_executor = ThreadPoolExecutor(
            max_workers=_executor_max_workers, thread_name_prefix="mpl_worker"
        )
        logger.info(f"Created ThreadPoolExecutor with {_executor_max_workers} workers")
    return _figure_executor


def _sync_save_figure(
    fig: Figure, file_format: str, dpi: int, transparent: bool
) -> bytes:
    """Synchronous figure save operation for thread pool execution.

    This function runs in a background thread to avoid blocking the event loop.

    Parameters
    ----------
    fig : Figure
        The matplotlib Figure to save
    file_format : str
        Output format (png, pdf, svg, etc.)
    dpi : int
        DPI for raster formats
    transparent : bool
        Whether to use transparent background

    Returns
    -------
    bytes
        The saved figure data

    Raises
    ------
    ValueError
        If the format is unsupported
    Exception
        If savefig fails
    """
    buf = io.BytesIO()
    save_kwargs: dict[str, Any] = {"format": file_format}
    if file_format in ["png", "jpg", "jpeg", "tiff", "tif"]:
        save_kwargs["dpi"] = dpi
    if file_format == "png":
        save_kwargs["transparent"] = transparent

    fig.savefig(buf, **save_kwargs)
    return buf.getvalue()


def _sync_draw_figure(canvas: FastAPICanvas) -> bytes | None:
    """Synchronous figure draw operation for thread pool execution.

    This function runs in a background thread to avoid blocking the event loop.
    Performs the matplotlib draw operation and generates the diff image.

    Parameters
    ----------
    canvas : FastAPICanvas
        The canvas to draw

    Returns
    -------
    bytes | None
        PNG image data (diff or full) or None if no update needed
    """
    from io import BytesIO

    import numpy as np
    from PIL import Image

    # Perform the blocking draw operation
    canvas.draw()

    # Generate diff image (this is also potentially slow)
    renderer = canvas.get_renderer()

    # Buffer as uint32 for pixel comparison
    buff = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint32).reshape(
        (int(renderer.height), int(renderer.width))
    )

    # Check for transparency
    pixels = buff.view(dtype=np.uint8).reshape(buff.shape + (4,))

    if canvas._force_full or np.any(pixels[:, :, 3] != 255):
        # Full image mode
        canvas._current_image_mode = "full"
        output = buff
    else:
        # Diff mode
        canvas._current_image_mode = "diff"
        diff = buff != canvas._last_buff
        output = np.where(diff, buff, 0)

    # Store current buffer for next diff
    np.copyto(canvas._last_buff, buff)
    canvas._force_full = False

    # Encode as PNG
    data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
    png_buf = BytesIO()
    Image.fromarray(data).save(png_buf, format="png")
    return png_buf.getvalue()


def shutdown_figure_executor() -> None:
    """Shutdown the thread pool executor gracefully.

    This should be called when the application is shutting down.
    Can be registered as a FastAPI shutdown event handler.
    """
    global _figure_executor
    if _figure_executor is not None:
        logger.info("Shutting down ThreadPoolExecutor")
        _figure_executor.shutdown(wait=True)
        _figure_executor = None


def _clean_old_saved_files() -> None:
    """Remove expired files and enforce max size limit."""
    now = datetime.now()
    expired_ids = []

    # Find expired files
    for file_id, saved_file in _saved_files.items():
        if now - saved_file.created_at > SAVE_CACHE_TTL:
            expired_ids.append(file_id)

    # Remove expired
    for file_id in expired_ids:
        _remove_saved_file(file_id)

    # Enforce max size (remove oldest if over limit)
    if len(_saved_files) > SAVE_CACHE_MAX_SIZE:
        sorted_files = sorted(_saved_files.items(), key=lambda x: x[1].created_at)
        files_to_remove = len(_saved_files) - SAVE_CACHE_MAX_SIZE
        for file_id, _ in sorted_files[:files_to_remove]:
            _remove_saved_file(file_id)


def _remove_saved_file(file_id: str) -> None:
    """Remove a saved file from all caches."""
    if file_id in _saved_files:
        saved_file = _saved_files[file_id]
        del _saved_files[file_id]

        # Remove from connection index
        if saved_file.connection_id in _connection_files:
            _connection_files[saved_file.connection_id].discard(file_id)
            if not _connection_files[saved_file.connection_id]:
                del _connection_files[saved_file.connection_id]

        logger.debug(f"Removed saved file: {file_id}")


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
        base_path = request.url.path.rstrip("/")

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
        path_parts = request.url.path.rstrip("/").split("/")
        base_path = ""
        if len(path_parts) >= 2:
            # Get everything before /plot/name
            prefix_parts = []
            for part in path_parts[1:]:  # Skip empty string from leading /
                if part == "plot":
                    break
                prefix_parts.append(part)
            if prefix_parts:
                base_path = "/" + "/".join(prefix_parts)
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

    # Route: Download saved file
    @router.get("/download/{file_id}")
    async def download_saved_file(file_id: str) -> StreamingResponse:
        """
        Download a previously saved figure file.

        Files are cached temporarily after a save request via WebSocket.
        This endpoint does not require an active WebSocket connection.

        Parameters
        ----------
        file_id : str
            The unique file ID provided after a save operation
        """
        # Validate file exists
        if file_id not in _saved_files:
            raise HTTPException(
                status_code=404,
                detail="File not found. It may have expired or been deleted.",
            )

        saved_file = _saved_files[file_id]

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
        mime_type = mime_types.get(saved_file.format, "application/octet-stream")

        logger.info(f"Serving saved file {file_id} ({saved_file.format})")

        # Return as streaming response with download headers
        return StreamingResponse(
            io.BytesIO(saved_file.data),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={saved_file.filename}"
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
            await websocket.send_json(
                {"type": "error", "message": f"Invalid parameters: {e}"}
            )
            await websocket.close(code=1008, reason=f"Invalid params: {e}")
            return

        logger.info(f"WebSocket connected for plot '{plot_name}' with params: {params}")

        # Create figure and call generator to populate it in background thread
        fig = Figure()
        try:
            logger.debug(f"Initializing figure '{plot_name}' in background thread")
            loop = asyncio.get_event_loop()
            executor = _get_figure_executor()

            state = await loop.run_in_executor(
                executor,
                config.init.function,
                fig,
                params,
            )
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

        # Send connection ID to client for download functionality
        await websocket.send_json({"type": "connection_id", "id": connection_id})

        # Send toolbar configuration
        toolbar_config = FastAPIManger.get_toolbar_config()
        await websocket.send_json(
            {"type": "toolbar_config", "items": toolbar_config["toolbar_items"]}
        )
        await websocket.send_json(
            {"type": "save_formats", "formats": toolbar_config["save_formats"]}
        )
        await websocket.send_json(
            {
                "type": "default_save_format",
                "format": toolbar_config["default_save_format"],
            }
        )

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
                logger.debug(
                    f"Received message type='{data.get('type')}' for plot '{plot_name}'"
                )

                try:
                    if data["type"] == "supports_binary":
                        manager.supports_binary = data["value"]
                        logger.debug(f"Set supports_binary={data['value']}")
                    elif data["type"] == "save_figure":
                        # Handle save request
                        try:
                            file_format = data.get("format", "png")
                            dpi = data.get("dpi", 100)
                            transparent = data.get("transparent", False)

                            # Validate format
                            supported_formats = [
                                "png",
                                "pdf",
                                "svg",
                                "eps",
                                "ps",
                                "jpg",
                                "jpeg",
                                "tiff",
                                "tif",
                            ]
                            format_lower = file_format.lower()
                            if format_lower not in supported_formats:
                                raise ValueError(
                                    f"Unsupported format '{file_format}'. "
                                    f"Supported: {', '.join(supported_formats)}"
                                )

                            # Generate unique file ID
                            file_id = str(uuid.uuid4())

                            # Save figure in thread pool to avoid blocking event loop
                            logger.debug(
                                f"Saving figure '{plot_name}' to {format_lower} "
                                f"in background thread"
                            )
                            loop = asyncio.get_event_loop()
                            executor = _get_figure_executor()

                            file_data = await loop.run_in_executor(
                                executor,
                                _sync_save_figure,
                                fig,
                                format_lower,
                                dpi,
                                transparent,
                            )

                            # Create saved file entry
                            saved_file = SavedFile(
                                file_id=file_id,
                                connection_id=connection_id,
                                data=file_data,
                                format=format_lower,
                                filename=f"{plot_name}.{format_lower}",
                                created_at=datetime.now(),
                                metadata={"dpi": dpi, "transparent": transparent},
                            )

                            # Store in caches
                            _saved_files[file_id] = saved_file
                            _connection_files[connection_id].add(file_id)

                            # Clean old files
                            _clean_old_saved_files()

                            # Extract base path from WebSocket path
                            # WebSocket URL pattern: /prefix/ws/{plot_name}
                            # We want: /prefix/download/{file_id}
                            base_path = ""
                            ws_path = websocket.url.path
                            if "/ws/" in ws_path:
                                base_path = ws_path.split("/ws/")[0]

                            download_url = f"{base_path}/download/{file_id}"

                            # Send success response to client
                            await websocket.send_json(
                                {
                                    "type": "save_complete",
                                    "file_id": file_id,
                                    "download_url": download_url,
                                    "filename": saved_file.filename,
                                    "format": format_lower,
                                }
                            )

                            logger.info(
                                f"Saved figure '{plot_name}' as {format_lower}, "
                                f"file_id={file_id}"
                            )

                        except ValueError as e:
                            logger.warning(f"Invalid save request: {e}")
                            await websocket.send_json(
                                {"type": "save_error", "message": str(e)}
                            )
                        except Exception as e:
                            logger.error(
                                f"Error saving figure '{plot_name}': {e}", exc_info=True
                            )
                            await websocket.send_json(
                                {
                                    "type": "save_error",
                                    "message": f"Failed to save figure: {str(e)}",
                                }
                            )
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

                                # Call update function in background thread
                                logger.debug(
                                    f"Updating figure '{plot_name}' in background thread"
                                )
                                loop = asyncio.get_event_loop()
                                executor = _get_figure_executor()

                                state = await loop.run_in_executor(
                                    executor,
                                    config.update.function,
                                    state,
                                    update_params,
                                )

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
                        # Skip logging for motion events to reduce noise
                        if e_type not in (
                            "motion_notify",
                            "figure_enter",
                            "figure_leave",
                        ):
                            logger.debug(f"Calling handler for event type '{e_type}'")

                        # Special case for draw to use thread pool
                        if e_type == "draw":
                            # Run draw in background thread
                            logger.debug("Running draw() in background thread")
                            loop = asyncio.get_event_loop()

                            diff_image = await loop.run_in_executor(
                                executor,
                                _sync_draw_figure,
                                canvas,
                            )

                            # Send image mode if it changed
                            await websocket.send_json(
                                {
                                    "type": "image_mode",
                                    "mode": canvas._current_image_mode,
                                }
                            )

                            # Send the image data
                            if diff_image is not None:
                                await websocket.send_bytes(diff_image)
                        else:
                            handler = getattr(
                                canvas, f"handle_{e_type}", canvas.handle_unknown_event
                            )
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

            # Clean up all saved files for this connection
            if connection_id in _connection_files:
                file_ids = list(_connection_files[connection_id])
                for file_id in file_ids:
                    _remove_saved_file(file_id)
                logger.debug(
                    f"Removed {len(file_ids)} saved file(s) for connection {connection_id}"
                )

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
                headers={"Content-Type": "application/json"},
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
