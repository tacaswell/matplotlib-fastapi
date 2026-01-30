import json
import logging
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.backend_bases import (
    FigureManagerBase,
    LocationEvent,
    MouseEvent,
    NavigationToolbar2,
    _Backend,
)
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.figure import Figure
from PIL import Image
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi.registry import select_gui_toolkit

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


class FastAPICanvas(FigureCanvasAgg):
    # Attributes from this class
    _force_full: bool
    _current_image_mode: str
    _msg_queue: deque[dict[str, Any]]
    _png_is_old: bool
    _last_buff: npt.NDArray[np.uint32]
    _renderer: RendererAgg
    supports_binary: bool = True

    # Declare attributes from parent FigureCanvasBase that we use
    call_info: dict[str, Any]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._force_full = False
        self._current_image_mode = "full"
        self._msg_queue = deque()

    def start_event_loop(self, timeout: float = 0) -> None:
        self.call_info["start_event_loop"] = {"timeout": timeout}

    async def handle_unknown_event(
        self, ev: dict[str, Any], websocket: WebSocket
    ) -> None:
        logger.debug(f"Unknown event type: {ev['type']}, data: {ev}")
        return

    async def handle_ack(self, ev: dict[str, Any], websocket: WebSocket) -> None: ...

    async def handle_resize(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        w = int(ev["width"] * self.device_pixel_ratio)
        h = int(ev["height"] * self.device_pixel_ratio)
        fig = self.figure
        fig.set_size_inches(w / fig.dpi, h / fig.dpi, forward=False)
        px_w, px_h = fig.bbox.size
        await websocket.send_json(
            {
                "type": "resize",
                "size": (
                    round(px_w / self.device_pixel_ratio),
                    round(px_h / self.device_pixel_ratio),
                ),
                "forward": True,
            }
        )

    async def handle_set_device_pixel_ratio(
        self, ev: dict[str, Any], websocket: WebSocket
    ) -> None:
        device_pixel_ratio = ev["device_pixel_ratio"]
        if self._set_device_pixel_ratio(device_pixel_ratio):  # type: ignore[attr-defined]
            self._force_full = True
            await websocket.send_json({"type": "draw"})

    async def handle_send_image_mode(
        self, ev: dict[str, Any], websocket: WebSocket
    ) -> None:
        await websocket.send_json(
            {"type": "image_mode", "mode": self._current_image_mode}
        )

    async def handle_refresh(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        await websocket.send_json(
            {"type": "figure_label", "label": self.figure.get_label()}
        )
        self._force_full = True
        await websocket.send_json({"type": "draw"})

    async def handle_draw(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        self._png_is_old = True
        try:
            super().draw()
        finally:
            diff = await self.get_diff_image(websocket)
            if diff is not None:
                await websocket.send_bytes(diff)

    async def set_image_mode(
        self, mode: Literal["full", "diff"], websocket: WebSocket
    ) -> None:
        """
        Set the image mode for any subsequent images which will be sent
        to the clients. The modes may currently be either 'full' or 'diff'.

        Note: diff images may not contain transparency, therefore upon
        draw this mode may be changed if the resulting image has any
        transparent component.
        """
        # _api.check_in_list(["full", "diff"], mode=mode)
        if self._current_image_mode != mode:
            self._current_image_mode = mode
            await websocket.send_json(
                {"type": "image_mode", "mode": self._current_image_mode}
            )

    async def get_diff_image(self, websocket: WebSocket) -> bytes | None:
        if self._png_is_old:
            renderer = self.get_renderer()

            # The buffer is created as type uint32 so that entire
            # pixels can be compared in one numpy call, rather than
            # needing to compare each plane separately.
            buff = np.frombuffer(renderer.buffer_rgba(), dtype=np.uint32).reshape(
                (int(renderer.height), int(renderer.width))
            )

            # If any pixels have transparency, we need to force a full
            # draw as we cannot overlay new on top of old.
            pixels = buff.view(dtype=np.uint8).reshape(buff.shape + (4,))

            if self._force_full or np.any(pixels[:, :, 3] != 255):
                await self.set_image_mode("full", websocket)
                output = buff
            else:
                await self.set_image_mode("diff", websocket)
                diff = buff != self._last_buff
                output = np.where(diff, buff, 0)

            # Store the current buffer so we can compute the next diff.
            np.copyto(self._last_buff, buff)
            self._force_full = False
            self._png_is_old = False

            data = output.view(dtype=np.uint8).reshape((*output.shape, 4))
            with BytesIO() as png:
                Image.fromarray(data).save(png, format="png")
                return png.getvalue()
        return None

    def get_renderer(self, cleared: bool | None = None) -> RendererAgg:
        # Mirrors super.get_renderer, but caches the old one so that we can do
        # things such as produce a diff image in get_diff_image.
        w, h = np.asarray(self.figure.bbox.size).astype(int)
        key = w, h, self.figure.dpi
        try:
            _, _ = self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self._renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key  # type: ignore[assignment]
            self._last_buff = np.copy(
                np.frombuffer(self._renderer.buffer_rgba(), dtype=np.uint32).reshape(
                    (self._renderer.height, self._renderer.width)
                )
            )

        elif cleared:
            self._renderer.clear()

        return self._renderer

    async def _handle_mouse(self, event: dict[str, Any], _websocket: WebSocket) -> None:
        x = event["x"]
        y = event["y"]
        y = self.get_renderer().height - y

        # Javascript button numbers and matplotlib button numbers are
        # off by 1
        button = event["button"] + 1

        e_type = event["type"]
        gui_event = event.get("guiEvent")
        if e_type == "button_press":
            MouseEvent(
                "button_press_event", self, x, y, button, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "dblclick":
            MouseEvent(
                "button_press_event",
                self,
                x,
                y,
                button,
                dblclick=True,
                guiEvent=gui_event,
            )._process()  # type: ignore[attr-defined]
        elif e_type == "button_release":
            MouseEvent(
                "button_release_event", self, x, y, button, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "motion_notify":
            MouseEvent("motion_notify_event", self, x, y, guiEvent=gui_event)._process()  # type: ignore[attr-defined]
        elif e_type == "figure_enter":
            LocationEvent(
                "figure_enter_event", self, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "figure_leave":
            LocationEvent(
                "figure_leave_event", self, x, y, guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]
        elif e_type == "scroll":
            MouseEvent(
                "scroll_event", self, x, y, step=event["step"], guiEvent=gui_event
            )._process()  # type: ignore[attr-defined]

    handle_button_press = handle_button_release = handle_dblclick = (
        handle_figure_enter
    ) = handle_figure_leave = handle_motion_notify = handle_scroll = _handle_mouse

    async def handle_toolbar_button(
        self, event: dict[str, Any], _websocket: WebSocket
    ) -> None:
        # Call the toolbar method
        getattr(self.toolbar, event["name"])()
        # Queue a draw event for the client to request
        self.queue_event("draw")

    def queue_event(self, event_type: str, **kwargs: Any) -> None:
        self._msg_queue.append({"type": event_type, **kwargs})

    def draw_idle(self) -> None:
        """Queue a draw event to be sent to the client."""
        self.queue_event("draw")

    async def drain_queue(self, websocket: WebSocket) -> None:
        while len(self._msg_queue):
            payload = self._msg_queue.popleft()
            await websocket.send_json(payload)


_ALLOWED_TOOL_ITEMS: set[str | None] = {
    "home",
    "back",
    "forward",
    "pan",
    "zoom",
    "download",
    None,
}


class NavigationToolbar2FastAPI(NavigationToolbar2):
    # Use the standard toolbar items + download button
    toolitems = tuple(
        (text, tooltip_text, image_file, name_of_method)  # type: ignore[misc]
        for text, tooltip_text, image_file, name_of_method in (
            *NavigationToolbar2.toolitems,
            ("Download", "Download plot", "filesave", "download"),
        )
        if name_of_method in _ALLOWED_TOOL_ITEMS
    )

    message: str
    _cursor: None
    canvas: FastAPICanvas  # Override parent type to be more specific

    # Declare parent class attribute we use
    _nav_stack: Any  # NavigationToolbar2._NavStack

    def __init__(self, canvas: FastAPICanvas) -> None:
        self.message = ""
        self._cursor = None  # Remove with deprecation.
        super().__init__(canvas)

    def set_message(self, message: str) -> None:
        if message != self.message:
            self.canvas.queue_event("message", message=message)
        self.message = message

    def draw_rubberband(
        self, event: Any, x0: float, y0: float, x1: float, y1: float
    ) -> None:
        self.canvas.queue_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def remove_rubberband(self) -> None:
        self.canvas.queue_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    def save_figure(self, *args: Any) -> None:
        """Save the current figure"""
        self.canvas.queue_event("save")

    def pan(self) -> None:
        super().pan()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def zoom(self) -> None:
        super().zoom()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def set_history_buttons(self) -> None:
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        self.canvas.queue_event(
            "history_buttons", Back=can_backward, Forward=can_forward
        )


class FastAPIManger(FigureManagerBase):
    ToolbarCls = NavigationToolbar2FastAPI
    web_sockets: set[WebSocket]
    toolbar: NavigationToolbar2FastAPI
    supports_binary: bool = True

    def __init__(self, canvas: FastAPICanvas, num: int) -> None:
        self.web_sockets = set()
        super().__init__(canvas, num)
        self.toolbar = self.ToolbarCls(canvas)

    @classmethod
    def get_javascript(cls) -> str:
        output = StringIO()

        output.write(
            (Path(__file__).parent / "static/js/mpl.js").read_text(encoding="utf-8")
        )

        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(["", "", "", ""])
            else:
                toolitems.append([name, tooltip, image, method])  # type: ignore[list-item]
        output.write(f"mpl.toolbar_items = {json.dumps(toolitems)};\n\n")

        extensions = []
        for _filetype, ext in sorted(
            FastAPICanvas.get_supported_filetypes_grouped().items()
        ):
            extensions.append(ext[0])
        output.write(f"mpl.extensions = {json.dumps(extensions)};\n\n")

        output.write(
            f"mpl.default_extension = {json.dumps(FastAPICanvas.get_default_filetype())};"
        )

        return output.getvalue()


class FastAPIBackend(_Backend):
    FigureCanvas = FastAPICanvas
    FigureManager = FastAPIManger


# Configure matplotlib to use our backend
select_gui_toolkit(FastAPIBackend)


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
        plot_config: tuple[PlotGenerator, type[BaseModel], str] = Depends(get_plot_config),
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
            logger.warning(f"WebSocket connection attempted for unknown plot: {plot_name}")
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
                    logger.error(f"Error receiving WebSocket message: {e}", exc_info=True)
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
