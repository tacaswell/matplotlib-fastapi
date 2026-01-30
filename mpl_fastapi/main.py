from io import BytesIO, StringIO
from typing import Any, Literal
import json
from pathlib import Path
from collections import deque

from pydantic import BaseModel

from starlette.websockets import WebSocketDisconnect

from fastapi import FastAPI, WebSocket, Request, Depends, Form
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import numpy as np
import numpy.typing as npt
from PIL import Image

from matplotlib.backend_bases import (_Backend, FigureManagerBase,
                                      NavigationToolbar2, MouseEvent, LocationEvent)
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from mpl_fastapi.utils import get_base_url
from mpl_fastapi.registry import FigureRegistry, select_gui_toolkit


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

    async def handle_unknown_event(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        print(ev["type"], ev)
        return None

    async def handle_ack(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        ...

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

    async def handle_set_device_pixel_ratio(self, ev: dict[str, Any], websocket: WebSocket) -> None:
        device_pixel_ratio = ev["device_pixel_ratio"]
        if self._set_device_pixel_ratio(device_pixel_ratio):  # type: ignore[attr-defined]
            self._force_full = True
            await websocket.send_json({"type": "draw"})

    async def handle_send_image_mode(self, ev: dict[str, Any], websocket: WebSocket) -> None:
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

    async def set_image_mode(self, mode: Literal["full", "diff"], websocket: WebSocket) -> None:
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
            _,_ = self._lastKey, self._renderer
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

    async def _handle_mouse(self, event: dict[str, Any], websocket: WebSocket) -> None:
        x = event["x"]
        y = event["y"]
        y = self.get_renderer().height - y

        # Javascript button numbers and matplotlib button numbers are
        # off by 1
        button = event["button"] + 1

        e_type = event["type"]
        guiEvent = event.get("guiEvent", None)
        if e_type == "button_press":
            MouseEvent("button_press_event", self, x, y, button, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "dblclick":
            MouseEvent("button_press_event", self, x, y, button, dblclick=True, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "button_release":
            MouseEvent("button_release_event", self, x, y, button, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "motion_notify":
            MouseEvent("motion_notify_event", self, x, y, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "figure_enter":
            LocationEvent("figure_enter_event", self, x, y, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "figure_leave":
            LocationEvent("figure_leave_event", self, x, y, guiEvent=guiEvent)._process()  # type: ignore[attr-defined]
        elif e_type == "scroll":
            MouseEvent("scroll_event", self, x, y, step=event["step"], guiEvent=guiEvent)._process()  # type: ignore[attr-defined]

    handle_button_press = (
        handle_button_release
    ) = (
        handle_dblclick
    ) = (
        handle_figure_enter
    ) = handle_figure_leave = handle_motion_notify = handle_scroll = _handle_mouse

    async def handle_toolbar_button(self, event: dict[str, Any], websocket: WebSocket) -> None:
        # TODO: Be more suspicious of the input
        getattr(self.toolbar, event["name"])()

    def queue_event(self, event_type: str, **kwargs: Any) -> None:
        self._msg_queue.append({"type": event_type, **kwargs})

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

    def draw_rubberband(self, event: Any, x0: float, y0: float, x1: float, y1: float) -> None:
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
        output.write("mpl.toolbar_items = {0};\n\n".format(json.dumps(toolitems)))

        extensions = []
        for filetype, ext in sorted(
            FastAPICanvas.get_supported_filetypes_grouped().items()
        ):
            extensions.append(ext[0])
        output.write("mpl.extensions = {0};\n\n".format(json.dumps(extensions)))

        output.write(
            "mpl.default_extension = {0};".format(
                json.dumps(FastAPICanvas.get_default_filetype())
            )
        )

        return output.getvalue()


class FastAPIBackend(_Backend):
    FigureCanvas = FastAPICanvas
    FigureManager = FastAPIManger


select_gui_toolkit(FastAPIBackend)
fr = FigureRegistry()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "figure.html",
        {
            "request": request,
            "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
            "fig_id": "bob",
        },
    )


@app.get("/figure/view/{figname}", response_class=HTMLResponse)
async def view_figure(request: Request, figname: str) -> HTMLResponse:
    return templates.TemplateResponse(
        "figure.html",
        {
            "request": request,
            "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
            "fig_id": figname,
        },
    )


class PlotData(BaseModel):
    x: list[float]
    y: list[float]
    label: str | None = None


@app.post("/axes/plot/{figname}/{axes}")
async def plot_data(request: Request, figname: str, axes: str, payload: PlotData) -> None:
    fig = fr.by_label[figname]
    ax = fig.axd[axes]  # type: ignore[attr-defined]
    ax.plot(payload.x, payload.y, label=payload.label)


class MosaicFigure(BaseModel):
    name: str
    pattern: str
    width: float = 6.4
    height: float = 4.8


async def _create_figure(name: str, pattern: str) -> tuple[Figure, dict[str, Axes]]:
    fig, axd = fr.subplot_mosaic(pattern, label=name)
    # monkey patch the axes dictionary on....
    fig.axd = axd  # type: ignore[attr-defined]
    return fig, axd


@app.post("/figure/create")
async def create_figure(request: Request, figure: MosaicFigure) -> dict[str, str]:
    print(figure)
    base_url = get_base_url(request)
    fig, ax = await _create_figure(figure.name, figure.pattern)
    fig.set_size_inches(figure.width, figure.height, forward=True)
    return {
        "figure_url": f"{base_url}figure/view/{figure.name}",
        "fig_id": figure.name,
    }


@app.get("/figure/form")
async def figure_form_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("create_form.html", {"request": request})


@app.post("/figure/form")
async def figure_form_post(
    request: Request, name: str = Form(...), pattern: str = Form(...)
) -> HTMLResponse:
    fig, axd = await _create_figure(name, pattern)
    return templates.TemplateResponse(
        "figure.html",
        {
            "request": request,
            "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
            "fig_id": name,
        },
    )


# TODO add caching logic
@app.get("/js/mpl.js", response_class=PlainTextResponse)
async def get_mpl_js(request: Request) -> PlainTextResponse:
    js = FastAPIManger.get_javascript()
    return PlainTextResponse(js, headers={"Content-Type": "application/javascript"})


@app.websocket("/ws/{fignum}")
async def websocket_endpoint(websocket: WebSocket, fignum: str) -> None:
    await websocket.accept()
    fig = fr.by_label[fignum]
    canvas = fig.canvas
    manager = canvas.manager

    # Type narrowing: ensure we have the correct types
    if not isinstance(canvas, FastAPICanvas):
        raise TypeError(f"Expected FastAPICanvas, got {type(canvas)}")
    if not isinstance(manager, FastAPIManger):
        raise TypeError(f"Expected FastAPIManger, got {type(manager)}")

    await websocket.send_json({"type": "image_mode", "mode": "full"})
    while True:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        if data["type"] == "supports_binary":
            manager.supports_binary = data["value"]
        else:
            e_type = data["type"]
            handler = getattr(
                canvas, f"handle_{e_type}", canvas.handle_unknown_event
            )
            # TODO we need to pass a list of all websockets associated with this
            # figure, not just the one the message came in on so all views stay
            # in sync
            await handler(data, websocket)
        await canvas.drain_queue(websocket)
