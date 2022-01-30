from io import BytesIO, StringIO
from typing import List
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
from PIL import Image

from matplotlib.backend_bases import _Backend, FigureManagerBase, NavigationToolbar2

from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from mpl_fastapi.utils import get_base_url
import mpl_gui as mg


class FastAPICanvas(FigureCanvasAgg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._force_full = False
        self._current_image_mode = "full"
        self._msg_queue = deque()

    def start_event_loop(self, timeout=0):
        self.call_info["start_event_loop"] = {"timeout": timeout}

    async def handle_unknown_event(self, ev, websocket):
        print(ev["type"], ev)
        return None

    async def handle_ack(self, ev, websocket):
        ...

    async def handle_resize(self, ev, websocket):
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

    async def handle_set_device_pixel_ratio(self, ev, websocket):
        device_pixel_ratio = ev["device_pixel_ratio"]
        if self._set_device_pixel_ratio(device_pixel_ratio):
            self._force_full = True
            await websocket.send_json({"type": "draw"})

    async def handle_send_image_mode(self, ev, websocket):
        await websocket.send_json(
            {"type": "image_mode", "mode": self._current_image_mode}
        )

    async def handle_refresh(self, ev, websocket):
        await websocket.send_json(
            {"type": "figure_label", "label": self.figure.get_label()}
        )
        self._force_full = True
        await websocket.send_json({"type": "draw"})

    async def handle_draw(self, ev, websocket):
        self._png_is_old = True
        try:
            super().draw()
        finally:
            diff = await self.get_diff_image(websocket)
            if diff is not None:
                await websocket.send_bytes(diff)

    async def set_image_mode(self, mode, websocket):
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

    async def get_diff_image(self, websocket):
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

    def get_renderer(self, cleared=None):
        # Mirrors super.get_renderer, but caches the old one so that we can do
        # things such as produce a diff image in get_diff_image.
        w, h = self.figure.bbox.size.astype(int)
        key = w, h, self.figure.dpi
        try:
            self._lastKey, self._renderer
        except AttributeError:
            need_new_renderer = True
        else:
            need_new_renderer = self._lastKey != key

        if need_new_renderer:
            self._renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
            self._last_buff = np.copy(
                np.frombuffer(self._renderer.buffer_rgba(), dtype=np.uint32).reshape(
                    (self._renderer.height, self._renderer.width)
                )
            )

        elif cleared:
            self._renderer.clear()

        return self._renderer

    async def _handle_mouse(self, event, websocket):
        x = event["x"]
        y = event["y"]
        y = self.get_renderer().height - y

        # Javascript button numbers and matplotlib button numbers are
        # off by 1
        button = event["button"] + 1

        e_type = event["type"]
        guiEvent = event.get("guiEvent", None)
        if e_type == "button_press":
            self.button_press_event(x, y, button, guiEvent=guiEvent)
        elif e_type == "dblclick":
            self.button_press_event(x, y, button, dblclick=True, guiEvent=guiEvent)
        elif e_type == "button_release":
            self.button_release_event(x, y, button, guiEvent=guiEvent)
        elif e_type == "motion_notify":
            self.motion_notify_event(x, y, guiEvent=guiEvent)
        elif e_type == "figure_enter":
            self.enter_notify_event(xy=(x, y), guiEvent=guiEvent)
        elif e_type == "figure_leave":
            self.leave_notify_event()
        elif e_type == "scroll":
            self.scroll_event(x, y, event["step"], guiEvent=guiEvent)

    handle_button_press = (
        handle_button_release
    ) = (
        handle_dblclick
    ) = (
        handle_figure_enter
    ) = handle_figure_leave = handle_motion_notify = handle_scroll = _handle_mouse

    async def handle_toolbar_button(self, event, websocket):
        # TODO: Be more suspicious of the input
        getattr(self.toolbar, event["name"])()

    def queue_event(self, event_type, **kwargs):
        self._msg_queue.append({"type": event_type, **kwargs})

    async def drain_queue(self, websocket):
        while len(self._msg_queue):
            payload = self._msg_queue.popleft()
            await websocket.send_json(payload)


_ALLOWED_TOOL_ITEMS = {
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
    toolitems = [
        (text, tooltip_text, image_file, name_of_method)
        for text, tooltip_text, image_file, name_of_method in (
            *NavigationToolbar2.toolitems,
            ("Download", "Download plot", "filesave", "download"),
        )
        if name_of_method in _ALLOWED_TOOL_ITEMS
    ]

    def __init__(self, canvas):
        self.message = ""
        self._cursor = None  # Remove with deprecation.
        super().__init__(canvas)

    def set_message(self, message):
        if message != self.message:
            self.canvas.queue_event("message", message=message)
        self.message = message

    def draw_rubberband(self, event, x0, y0, x1, y1):
        self.canvas.queue_event("rubberband", x0=x0, y0=y0, x1=x1, y1=y1)

    def remove_rubberband(self):
        self.canvas.queue_event("rubberband", x0=-1, y0=-1, x1=-1, y1=-1)

    def save_figure(self, *args):
        """Save the current figure"""
        self.canvas.queue_event("save")

    def pan(self):
        super().pan()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def zoom(self):
        super().zoom()
        self.canvas.queue_event("navigate_mode", mode=self.mode.name)

    def set_history_buttons(self):
        can_backward = self._nav_stack._pos > 0
        can_forward = self._nav_stack._pos < len(self._nav_stack._elements) - 1
        self.canvas.queue_event(
            "history_buttons", Back=can_backward, Forward=can_forward
        )


class FastAPIManger(FigureManagerBase):
    ToolbarCls = NavigationToolbar2FastAPI

    def __init__(self, canvas, num):
        self.web_sockets = set()
        super().__init__(canvas, num)
        self.toolbar = self.ToolbarCls(canvas)

    @classmethod
    def get_javascript(cls):
        output = StringIO()

        output.write(
            (Path(__file__).parent / "static/js/mpl.js").read_text(encoding="utf-8")
        )

        toolitems = []
        for name, tooltip, image, method in cls.ToolbarCls.toolitems:
            if name is None:
                toolitems.append(["", "", "", ""])
            else:
                toolitems.append([name, tooltip, image, method])
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


mg.select_gui_toolkit(FastAPIBackend)
fr = mg.FigureRegistry()

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse(
        "figure.html",
        {
            "request": request,
            "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
            "fig_id": "bob",
        },
    )


@app.get("/figure/view/{figname}", response_class=HTMLResponse)
async def read_item(request: Request, figname: str):
    return templates.TemplateResponse(
        "figure.html",
        {
            "request": request,
            "ws_uri": f"ws://{request.url.hostname}:{request.url.port}",
            "fig_id": figname,
        },
    )


class PlotData(BaseModel):
    x: List[float]
    y: List[float]
    label: str | None = None


@app.post("/axes/plot/{figname}/{axes}")
async def read_item(request: Request, figname: str, axes: str, payload: PlotData):
    fig = fr.by_label[figname]
    ax = fig.axd[axes]
    ax.plot(payload.x, payload.y, label=payload.label)


class MosaicFigure(BaseModel):
    name: str
    pattern: str
    width: float = 6.4
    height: float = 4.8


async def _create_figure(name, pattern):
    fig, axd = fr.subplot_mosaic(pattern, label=name)
    mg.promote_figure(fig)
    # monkey patch the axes dictionary on....
    fig.axd = axd
    return fig, axd


@app.post("/figure/create")
async def create_figure(request: Request, figure: MosaicFigure):
    print(figure)
    base_url = get_base_url(request)
    fig, ax = await _create_figure(figure.name, figure.pattern)
    fig.set_size_inches(figure.width, figure.height, forward=True)
    return {
        "figure_url": f"{base_url}figure/view/{figure.name}",
        "fig_id": figure.name,
    }


@app.get("/figure/form")
async def figure_form(request: Request):
    return templates.TemplateResponse("create_form.html", {"request": request})


@app.post("/figure/form")
async def figure_form(
    request: Request, name: str = Form(...), pattern: str = Form(...)
):
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
async def get_mpl_js(request: Request):
    js = FastAPIManger.get_javascript()
    return PlainTextResponse(js, headers={"Content-Type": "application/javascript"})


@app.websocket("/ws/{fignum}")
async def websocket_endpoint(websocket: WebSocket, fignum: str):
    await websocket.accept()
    fig = fr.by_label[fignum]
    canvas = fig.canvas
    manager = canvas.manager

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
                canvas, "handle_{0}".format(e_type), canvas.handle_unknown_event
            )
            # TODO we need to pass a list of all websockets associated with this
            # figure, not just the one the message came in on so all views stay
            # in sync
            await handler(data, websocket)
        await canvas.drain_queue(websocket)
