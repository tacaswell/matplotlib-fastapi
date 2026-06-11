"""FastAPI router factory for matplotlib figure integration.

This module provides the router factory function and response models for
creating mountable FastAPI routers that serve interactive matplotlib plots.

The router supports:
- HTML listing of available plots with parameter forms
- JSON API for programmatic access
- Individual plot viewers with WebSocket communication
- Dynamic parameter passing via query strings

Protocol v0 uses a simplified message flow:
- Client sends consolidated `init` message as first message
- Server responds with consolidated `config` message
- Binary image messages have 8-byte headers with sequence numbers
"""

import asyncio
import hashlib
import io
import json
import logging
import os
import struct
import uuid
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import (
    HTMLResponse,
    PlainTextResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from matplotlib.figure import Figure
from pydantic import BaseModel, ValidationError
from starlette.websockets import WebSocketDisconnect

from mpl_fastapi import __version__
from mpl_fastapi.auth import AuthPolicy, NoAuth
from mpl_fastapi.mpl_backend import FastAPICanvas, FastAPIManger

# Protocol constants
PROTOCOL_VERSION = 0

# Reserved query-string prefix for update parameters.
# Keys like ``_update.phase=1.57`` are split out and validated against
# the UpdateConfig.params_model.  All other keys are treated as init params.
_UPDATE_PREFIX = "_update."

# Query-string keys that are consumed by the framework (auth, etc.)
# and must not be forwarded to plot init/update parameter models.
_RESERVED_QUERY_KEYS = frozenset({"token"})


def _split_query_params(
    raw: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """Separate init params from ``_update.*`` params.

    Parameters
    ----------
    raw : dict
        Raw query-string key/value pairs from the request.

    Returns
    -------
    init_raw : dict
        Keys that do **not** start with ``_update.`` and are not reserved.
    update_raw : dict
        Keys that start with ``_update.``, with the prefix stripped.
    """
    init_raw: dict[str, str] = {}
    update_raw: dict[str, str] = {}
    for key, value in raw.items():
        if key.startswith(_UPDATE_PREFIX):
            update_raw[key[len(_UPDATE_PREFIX) :]] = value
        elif key not in _RESERVED_QUERY_KEYS:
            init_raw[key] = value
    return init_raw, update_raw


# JS caching control: set MPL_NO_CACHE_JS=1 (or any truthy value) to relax
# the aggressive Cache-Control headers on content-hashed bundles.  This is
# useful during development so that rebuilding the JS bundle is immediately
# reflected in the browser without a hard refresh.
# This only affects static JS serving — it does not change logging levels
# or any other server behaviour.
_NO_CACHE_JS: bool = os.environ.get("MPL_NO_CACHE_JS", "").strip() not in ("", "0")

# Cache headers for content-hashed JS bundles.
# These URLs contain the content hash, so the resource at a given URL
# truly never changes — ``immutable`` is semantically correct.
# In dev-mode we use ``no-cache`` so the browser always revalidates.
_JS_IMMUTABLE_HEADERS: dict[str, str] = (
    {"Cache-Control": "no-cache"}
    if _NO_CACHE_JS
    else {"Cache-Control": "public, max-age=31536000, immutable"}
)

# Cache headers for stable (hash-free) redirect URLs.
# Browsers must revalidate on every visit so they pick up the new hash
# after a deployment.  The redirect itself is tiny.
_JS_REDIRECT_HEADERS: dict[str, str] = {"Cache-Control": "no-cache"}


def _read_static_file(filename: str) -> tuple[str, str] | None:
    """Read a static JS file from disk, returning ``(content, etag)``.

    The file is read fresh on every call so a rebuilt bundle is picked up
    without restarting the server.  The ETag is derived from a content
    hash, which also drives the content-hashed URL scheme.
    """
    path = Path(__file__).parent / "static/js/dist" / filename
    if path.exists():
        content = path.read_text(encoding="utf-8")
        etag = hashlib.sha256(content.encode()).hexdigest()[:16]
        return content, f'"{etag}"'
    return None


def _get_js_hash(filename: str) -> str | None:
    """Return the short content hash for a JS file, or None if missing."""
    result = _read_static_file(filename)
    if result is None:
        return None
    # ETag is '"<hex>"'; strip the quotes to get the bare hash.
    return result[1].strip('"')


def _prefix_before(path: str, marker: str) -> str:
    """Return the router-mount prefix that precedes *marker* in *path*.

    The router exposes its routes under a mount prefix (e.g. ``/plots``),
    so a request path looks like ``/plots/plot/sine`` or
    ``/plots/ws/v0/sine``.  Given the marker segment that begins the
    router-internal portion of the path (``/plot/``, ``/ws/``, ``/plots``),
    return everything before it (``/plots``), or ``""`` when the router is
    mounted at the application root.

    This is the single source of truth for deriving the mount prefix from
    a request URL so that the HTML, JSON, and save handlers all agree.
    """
    idx = path.find(marker)
    if idx == -1:
        return path.rstrip("/")
    return path[:idx].rstrip("/")


def _http_scheme(request: Request) -> str:
    """Return the effective HTTP scheme, honoring a reverse proxy."""
    return request.headers.get("x-forwarded-proto", request.url.scheme)


def _request_host(request: Request) -> str:
    """Return the effective host:port, honoring a reverse proxy."""
    return request.headers.get("x-forwarded-host", request.headers.get("host", ""))


def _ws_scheme(request: Request) -> str:
    """Return ``wss`` when the page is served over HTTPS, else ``ws``."""
    return "wss" if _http_scheme(request) == "https" else "ws"


def _normalize_origin(origin: str) -> str:
    """Normalize an HTTP ``Origin`` for case/slash-insensitive comparison.

    An Origin is ``scheme://host[:port]`` whose scheme and host are
    case-insensitive (RFC 6454).  Lowercase the value and strip any
    trailing slash so that, e.g., ``https://Example.com/`` and
    ``https://example.com`` compare equal.
    """
    return origin.strip().rstrip("/").lower()


class ImageTypeMode(IntEnum):
    """Combined image type and mode byte values."""

    FULL = 0x00  # Full image
    DIFF = 0x01  # Differential image


class ImageFormat(IntEnum):
    """Image format byte values."""

    PNG = 0x01
    JPEG = 0x02
    WEBP = 0x03


def _build_image_header(
    type_mode: ImageTypeMode,
    image_format: ImageFormat,
    seq_num: int,
    base_seq: int,
    flags: int = 0,
) -> bytes:
    """Build the 8-byte binary image header.

    Header structure:
        Byte 0:    type_mode   - 0x00=full_image, 0x01=diff_image
        Byte 1:    format      - 0x01=PNG, 0x02=JPEG, 0x03=WebP
        Bytes 2-3: seq_num     - uint16 BE, sequence number of this image
        Bytes 4-5: base_seq    - uint16 BE, base sequence (0 for full images)
        Bytes 6-7: flags       - reserved (0x0000)

    Parameters
    ----------
    type_mode : ImageTypeMode
        Image type and mode (full or diff)
    image_format : ImageFormat
        Image encoding format
    seq_num : int
        Sequence number of this image (1-65535, wraps)
    base_seq : int
        Sequence number this diff is based on (0 for full images)
    flags : int
        Reserved flags (default 0)

    Returns
    -------
    bytes
        8-byte header
    """
    return struct.pack(
        ">BBHHH",
        type_mode,
        image_format,
        seq_num & 0xFFFF,
        base_seq & 0xFFFF,
        flags & 0xFFFF,
    )


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


class RouterState:
    """Holds mutable state for a router instance.

    This class encapsulates all per-router state including:
    - Active WebSocket connections tracking
    - Cached saved files

    Using a class instance instead of module-level globals allows
    multiple router instances to coexist without shared state.
    """

    def __init__(self) -> None:
        # Saved files cache
        # Maps file_id -> SavedFile
        self.saved_files: dict[str, SavedFile] = {}
        # Maps connection_id -> set of file_ids for cleanup
        self.connection_files: dict[str, set[str]] = defaultdict(set)

        # Connection tracking
        # Maps plot_name -> count of active connections
        self.active_connections: dict[str, int] = defaultdict(int)
        self.total_connections: int = 0

    def connect(self, plot_name: str) -> None:
        """Record a new WebSocket connection."""
        self.active_connections[plot_name] += 1
        self.total_connections += 1

    def disconnect(self, plot_name: str) -> None:
        """Record a WebSocket disconnection."""
        self.active_connections[plot_name] -= 1
        self.total_connections -= 1
        if self.active_connections[plot_name] <= 0:
            del self.active_connections[plot_name]

    def get_health_status(self) -> dict[str, Any]:
        """Get the minimal, public health status.

        This payload is intentionally free of operational details so it can
        be exposed without authentication (e.g. to load balancers).

        Returns
        -------
        dict
            ``{"status": "ok"}``
        """
        return {"status": "ok"}

    def get_health_details(self) -> dict[str, Any]:
        """Get detailed health statistics (authenticated callers only).

        Returns
        -------
        dict
            Health status with connection counts and cached file count.
        """
        return {
            "status": "ok",
            "connections": self.total_connections,
            "connections_by_plot": dict(self.active_connections),
            "cached_files": len(self.saved_files),
        }


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
    ws_url: str | None = None
    view_url: str | None = None


class PlotsListResponse(BaseModel):
    """Response model for the plots list endpoint."""

    plots: dict[str, PlotInfo]


@dataclass
class MPLRouter:
    """Container for matplotlib router and its static assets."""

    router: APIRouter
    static_files: StaticFiles
    static_mount_path: str
    state: RouterState


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
        logger.info("Created ThreadPoolExecutor with %d workers", _executor_max_workers)
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


@dataclass
class ImageSequenceState:
    """Tracks image sequence numbers for a connection."""

    seq_num: int = 0  # Current sequence number
    last_full_seq: int = 0  # Sequence number of last full image

    def next_full(self) -> tuple[int, int]:
        """Get next sequence number for a full image.

        Returns
        -------
        tuple[int, int]
            (seq_num, base_seq) where base_seq is 0 for full images
        """
        self.seq_num = (self.seq_num % 65535) + 1  # 1-65535, skip 0
        self.last_full_seq = self.seq_num
        return self.seq_num, 0

    def next_diff(self) -> tuple[int, int]:
        """Get next sequence number for a diff image.

        Returns
        -------
        tuple[int, int]
            (seq_num, base_seq) where base_seq is the previous seq_num
        """
        base_seq = self.seq_num
        self.seq_num = (self.seq_num % 65535) + 1
        return self.seq_num, base_seq


async def _send_image_frame(
    websocket: WebSocket,
    image_data: bytes,
    is_diff: bool,
    seq_state: ImageSequenceState,
) -> None:
    """Build the 8-byte header and send a single binary image frame.

    This is the single place where header construction and ``send_bytes`` live.
    Both the render/refresh path and the blit drain path call this function.

    Parameters
    ----------
    websocket : WebSocket
        Active WebSocket connection.
    image_data : bytes
        PNG-encoded image data.
    is_diff : bool
        True if *image_data* is a diff frame; False for a full frame.
    seq_state : ImageSequenceState
        Sequence number tracker for this connection.
    """
    if is_diff:
        seq_num, base_seq = seq_state.next_diff()
        type_mode = ImageTypeMode.DIFF
    else:
        seq_num, base_seq = seq_state.next_full()
        type_mode = ImageTypeMode.FULL

    header = _build_image_header(
        type_mode=type_mode,
        image_format=ImageFormat.PNG,
        seq_num=seq_num,
        base_seq=base_seq,
    )
    await websocket.send_bytes(header + image_data)
    logger.debug(
        "Sent image: type=%s, seq=%s, base=%s, size=%d bytes",
        type_mode.name,
        seq_num,
        base_seq,
        len(image_data),
    )


async def _send_render_response(
    websocket: WebSocket,
    canvas: FastAPICanvas,
    executor: ThreadPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    seq_state: ImageSequenceState,
) -> None:
    """Render the figure in a background thread and send the binary frame.

    Parameters
    ----------
    websocket : WebSocket
        Active WebSocket connection
    canvas : FastAPICanvas
        Canvas to render
    executor : ThreadPoolExecutor
        Thread pool for blocking operations
    loop : asyncio.AbstractEventLoop
        Event loop for running in executor
    seq_state : ImageSequenceState
        Sequence number tracker for this connection
    """
    logger.debug("Running draw_and_get_diff() in background thread")
    image_data, is_diff = await loop.run_in_executor(
        executor,
        canvas.draw_and_get_diff,
    )
    await _send_image_frame(websocket, image_data, is_diff, seq_state)


async def _drain_canvas(
    canvas: FastAPICanvas,
    websocket: WebSocket,
    seq_state: ImageSequenceState,
) -> None:
    """Drain the canvas JSON message queue and any binary frames queued by blit().

    Parameters
    ----------
    canvas : FastAPICanvas
        The canvas whose queues should be flushed.
    websocket : WebSocket
        Active WebSocket connection to send messages on.
    seq_state : ImageSequenceState
        Sequence number tracker for this connection.
    """
    # JSON messages first (invalidate, message, rubberband, etc.)
    await canvas.drain_queue(websocket)

    # Binary images queued by blit()
    while canvas._binary_queue:
        image_data, is_diff = canvas._binary_queue.popleft()
        await _send_image_frame(websocket, image_data, is_diff, seq_state)


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


def _clean_old_saved_files(state: RouterState) -> None:
    """Remove expired files and enforce max size limit."""
    now = datetime.now()
    expired_ids = []

    # Find expired files
    for file_id, saved_file in state.saved_files.items():
        if now - saved_file.created_at > SAVE_CACHE_TTL:
            expired_ids.append(file_id)

    # Remove expired
    for file_id in expired_ids:
        _remove_saved_file(state, file_id)

    # Enforce max size (remove oldest if over limit)
    if len(state.saved_files) > SAVE_CACHE_MAX_SIZE:
        sorted_files = sorted(state.saved_files.items(), key=lambda x: x[1].created_at)
        files_to_remove = len(state.saved_files) - SAVE_CACHE_MAX_SIZE
        for file_id, _ in sorted_files[:files_to_remove]:
            _remove_saved_file(state, file_id)


def _remove_saved_file(state: RouterState, file_id: str) -> None:
    """Remove a saved file from all caches."""
    if file_id in state.saved_files:
        saved_file = state.saved_files[file_id]
        del state.saved_files[file_id]

        # Remove from connection index
        if saved_file.connection_id in state.connection_files:
            state.connection_files[saved_file.connection_id].discard(file_id)
            if not state.connection_files[saved_file.connection_id]:
                del state.connection_files[saved_file.connection_id]

        logger.debug("Removed saved file: %s", file_id)


# Type alias for the lifespan callable that FastAPI expects.
# Using Any for the context manager return to be compatible with
# Starlette's Lifespan which can return state mappings.
Lifespan = Callable[..., AbstractAsyncContextManager[Any]]


def compose_lifespans(
    *lifespans: Lifespan,
) -> Lifespan:
    """Compose multiple lifespan context managers into one.

    FastAPI only accepts a single ``lifespan`` async context manager.
    This helper chains several together so that startup hooks run in
    order and shutdown hooks run in reverse order (like nested ``with``
    blocks).

    Parameters
    ----------
    *lifespans : Callable[[FastAPI], AbstractAsyncContextManager[None]]
        Lifespan context managers to compose.  Each must be decorated
        with ``@asynccontextmanager`` (or be an async generator function
        that yields exactly once).

    Returns
    -------
    Callable[[FastAPI], AbstractAsyncContextManager[None]]
        A single lifespan suitable for passing to ``FastAPI(lifespan=...)``.

    Examples
    --------
    >>> from contextlib import asynccontextmanager
    >>> @asynccontextmanager
    ... async def db_lifespan(app):
    ...     print("db startup")
    ...     yield
    ...     print("db shutdown")
    >>> @asynccontextmanager
    ... async def cache_lifespan(app):
    ...     print("cache startup")
    ...     yield
    ...     print("cache shutdown")
    >>> app = FastAPI(lifespan=compose_lifespans(db_lifespan, cache_lifespan))
    """

    @asynccontextmanager
    async def composed(app: FastAPI) -> AsyncIterator[None]:
        match lifespans:
            case ():
                yield
            case (only,):
                async with only(app):
                    yield
            case (first, *rest):
                async with first(app), compose_lifespans(*rest)(app):
                    yield

    return composed  # type: ignore[return-value]


def _mpl_lifespan() -> Lifespan:
    """Create a lifespan context manager that cleans up the figure executor.

    Returns
    -------
    Callable
        An ``@asynccontextmanager``-decorated async generator suitable for
        use with :func:`compose_lifespans` or ``FastAPI(lifespan=...)``.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
        yield
        shutdown_figure_executor()

    return lifespan  # type: ignore[return-value]


def install_mpl_router(
    app: FastAPI,
    mpl: "MPLRouter",
    *,
    prefix: str = "",
) -> None:
    """Install an :class:`MPLRouter` onto a FastAPI application.

    This performs all the required wiring in a single call:

    1. ``app.include_router(mpl.router, prefix=prefix)``
    2. ``app.mount(mpl.static_mount_path, mpl.static_files, ...)``
    3. Compose the mpl shutdown lifespan with any existing app lifespan.

    The function is idempotent with respect to the lifespan — calling it
    multiple times (e.g. to mount several ``MPLRouter`` instances) will
    compose all their lifespans correctly without overwriting the
    application's own lifespan.

    Parameters
    ----------
    app : FastAPI
        The application to install the router onto.
    mpl : MPLRouter
        The router container returned by :func:`create_mpl_router`.
    prefix : str, optional
        URL prefix for the router (default ``""``).  For example,
        ``prefix="/plots"`` would make the list endpoint available at
        ``/plots/``.

    Examples
    --------
    >>> mpl = create_mpl_router({"sine": sine_config})
    >>> app = FastAPI()
    >>> install_mpl_router(app, mpl, prefix="/plots")
    >>>
    >>> # Multiple routers work too:
    >>> mpl2 = create_mpl_router({"cosine": cosine_config})
    >>> install_mpl_router(app, mpl2, prefix="/plots2")
    """
    # 1. Include routes
    app.include_router(mpl.router, prefix=prefix)

    # 2. Mount static files
    # Use a unique name to allow multiple mounts
    static_name = f"mpl_static_{prefix.strip('/')}" if prefix else "mpl_static"
    app.mount(mpl.static_mount_path, mpl.static_files, name=static_name)

    # 3. Chain the mpl lifespan with the existing app lifespan
    existing_lifespan = app.router.lifespan_context
    app.router.lifespan_context = compose_lifespans(existing_lifespan, _mpl_lifespan())


def create_mpl_router(
    plot_generators: dict[str, PlotConfig],
    *,
    template_dir: Path | str | None = None,
    static_mount_path: str = "/mpl-static",
    auth: AuthPolicy | None = None,
    allowed_origins: list[str] | None = None,
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
    auth : AuthPolicy or None, optional
        Authentication policy for the router.  Pass ``None`` or omit
        for open access (:class:`~mpl_fastapi.auth.NoAuth`).  Use
        :class:`~mpl_fastapi.auth.SingleUserToken` for bearer-token
        protection, or supply any object satisfying the
        :class:`~mpl_fastapi.auth.AuthPolicy` protocol.
    allowed_origins : list[str] or None, optional
        Explicit list of origins permitted to open WebSocket connections,
        e.g. ``["https://example.com", "http://localhost:3000"]``.
        When ``None`` (the default) the ``BACKEND_CORS_ORIGINS``
        environment variable is read and split on commas.  An empty list
        means *any* origin is allowed (permissive, suitable for local
        development or fully token-protected deployments).  Note: browsers
        do not enforce CORS on WebSocket upgrades; this check is
        server-side enforcement of the same policy.

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
    >>> app = FastAPI()
    >>> install_mpl_router(app, mpl, prefix="/plots")

    Or, for manual control over each step:

    >>> app.include_router(mpl.router, prefix="/plots")
    >>> app.mount(mpl.static_mount_path, mpl.static_files, name="mpl_static")
    """
    router = APIRouter()

    # Resolve auth policy
    if auth is None:
        auth = NoAuth()
    _http_auth = auth.http_dependency()
    _ws_auth = auth.ws_dependency()

    # Resolve WebSocket origin allow-list.
    # None / [] → no restriction (any origin accepted).
    # Non-empty list → only those origins may open connections.
    #
    # An Origin is ``scheme://host[:port]`` with no path or trailing slash.
    # The scheme and host are case-insensitive (RFC 6454), so normalize both
    # the configured list and the incoming header to a lowercase form without
    # a trailing slash to avoid surprising case/slash mismatches.
    _ws_allowed_origins: frozenset[str] = frozenset(
        _normalize_origin(o) for o in (allowed_origins or [])
    )

    async def _check_ws_origin(websocket: WebSocket) -> None:
        """Dependency: reject WebSocket upgrades from disallowed origins."""
        if not _ws_allowed_origins:
            return  # no restriction configured
        origin = _normalize_origin(websocket.headers.get("origin", ""))
        if origin not in _ws_allowed_origins:
            await websocket.close(code=1008, reason="Origin not allowed")
            from fastapi import WebSocketException

            raise WebSocketException(code=1008, reason="Origin not allowed")

    # Create state instance for this router
    router_state = RouterState()

    # Precompute the JSON schemas for every plot once at construction time.
    # Pydantic schemas are static for a given model, so generating them per
    # request (in the list/view/schema endpoints and the WebSocket config
    # handshake) is wasted work under load.  ``_update_schemas[name]`` is
    # ``None`` for plots without an update function.
    _init_schemas: dict[str, dict[str, Any]] = {
        name: config.init.params_model.model_json_schema()
        for name, config in plot_generators.items()
    }
    _update_schemas: dict[str, dict[str, Any] | None] = {
        name: (
            config.update.params_model.model_json_schema()
            if config.update is not None
            else None
        )
        for name, config in plot_generators.items()
    }

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
    @router.get("/", response_class=HTMLResponse, dependencies=[Depends(_http_auth)])
    async def plots_list_html(request: Request) -> HTMLResponse:
        """Render an HTML page listing all available plots."""
        plots_info = {}
        for name, config in plot_generators.items():
            plots_info[name] = {
                "description": config.description,
                "parameters": _init_schemas[name],
            }

        # Extract base path from request URL
        base_path = request.url.path.rstrip("/")

        return templates.TemplateResponse(
            request,
            "plots_list.html",
            {
                "plots": plots_info,
                "base_path": base_path,
                "watermark": await watermark(),
            },
        )

    # Route: List all available plots (JSON API)
    @router.get(
        "/plots", response_model=PlotsListResponse, dependencies=[Depends(_http_auth)]
    )
    async def list_plots(request: Request) -> PlotsListResponse:
        """List all available plots with their parameter schemas."""
        # Derive base URLs from the request so clients get absolute,
        # ready-to-use WebSocket and HTTP addresses for each plot.
        http_scheme = _http_scheme(request)
        host = _request_host(request)
        ws_scheme = _ws_scheme(request)

        # Extract the router prefix from the request path.
        # request.url.path for this endpoint is e.g. "/plots/plots".
        prefix = _prefix_before(request.url.path, "/plots")

        plots_info = {}
        for name, config in plot_generators.items():
            plots_info[name] = PlotInfo(
                description=config.description,
                parameters=_init_schemas[name],
                update_schema=_update_schemas[name],
                ws_url=f"{ws_scheme}://{host}{prefix}/ws/v0/{name}",
                view_url=f"{http_scheme}://{host}{prefix}/plot/{name}",
            )
        return PlotsListResponse(plots=plots_info)

    # Route: Public health check endpoint (unauthenticated).
    # Deliberately returns only a bare status so it leaks no operational
    # detail (connection counts, plot names) to unauthenticated callers.
    @router.get("/health")
    async def health_check() -> dict[str, Any]:
        """Liveness probe returning a bare ``{"status": "ok"}``.

        This endpoint is intentionally unauthenticated so it can be used by
        load balancers and uptime checks.  Detailed statistics live behind
        the authenticated ``/health/details`` route.
        """
        return router_state.get_health_status()

    # Route: Detailed health check (authenticated).
    @router.get("/health/details", dependencies=[Depends(_http_auth)])
    async def health_details() -> dict[str, Any]:
        """Health check with connection statistics (requires auth).

        Returns
        -------
        dict
            Health status with:
            - status: "ok" if healthy
            - connections: Total active WebSocket connections
            - connections_by_plot: Breakdown of connections per plot
            - cached_files: Number of cached save files
        """
        return router_state.get_health_details()

    @router.get("/watermark", dependencies=[Depends(_http_auth)])
    async def watermark() -> dict[str, str]:
        """Version watermark reporting key dependency versions."""
        import importlib.metadata
        import platform

        import fastapi
        import matplotlib
        import starlette

        versions: dict[str, str] = {
            "fastapi": fastapi.__version__,
            "Matplotlib": matplotlib.__version__,
            "mpl_fastapi": __version__,
            "pydantic": importlib.metadata.version("pydantic"),
            "Python": platform.python_version(),
            "starlette": starlette.__version__,
            "uvicorn": importlib.metadata.version("uvicorn"),
        }
        return versions

    # Route: View a specific plot
    @router.get(
        "/plot/{plot_name}",
        response_class=HTMLResponse,
        dependencies=[Depends(_http_auth)],
    )
    async def view_plot(
        request: Request,
        plot_name: str,
        _: None = Depends(validate_plot_name),
    ) -> HTMLResponse:
        """Render the plot viewer HTML."""
        # Check if update functionality is available (precomputed schema).
        update_params_schema = _update_schemas[plot_name]

        # Derive the router mount prefix (e.g. "/plots") from the request
        # path "/plots/plot/{name}".
        base_path = _prefix_before(request.url.path, "/plot/")

        # Build the WebSocket origin honoring TLS and any reverse proxy.
        # Using the forwarded scheme/host fixes mixed-content failures when
        # the page itself is served over HTTPS (ws:// would be blocked).
        ws_uri = f"{_ws_scheme(request)}://{_request_host(request)}{base_path}"

        # Extract _update.* values from the page URL so the form can
        # be pre-populated with them instead of schema defaults.
        _init_values, update_values = _split_query_params(dict(request.query_params))
        del _init_values

        # Reference the stable JS URL; the redirect route resolves it to the
        # current content-hashed bundle (single source of truth for the hash).
        js_url = f"{base_path}/component.js"

        return templates.TemplateResponse(
            request,
            "figure.html",
            {
                "ws_uri": ws_uri,
                "base_path": base_path,
                "fig_id": plot_name,
                "static_path": static_mount_path,
                "js_url": js_url,
                "update_params_schema": update_params_schema,
                "update_values": update_values,
                "watermark": await watermark(),
            },
        )

    # Route: Download saved file
    @router.get("/download/{file_id}", dependencies=[Depends(_http_auth)])
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
        if file_id not in router_state.saved_files:
            raise HTTPException(
                status_code=404,
                detail="File not found. It may have expired or been deleted.",
            )

        saved_file = router_state.saved_files[file_id]

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

        logger.info("Serving saved file %s (%s)", file_id, saved_file.format)

        # Return as streaming response with download headers
        return StreamingResponse(
            io.BytesIO(saved_file.data),
            media_type=mime_type,
            headers={
                "Content-Disposition": f"attachment; filename={saved_file.filename}"
            },
        )

    # Route: WebSocket connection for interactive plotting (v0 protocol)
    @router.websocket(
        "/ws/v0/{plot_name}",
        dependencies=[Depends(_ws_auth), Depends(_check_ws_origin)],
    )
    async def websocket_endpoint_v0(websocket: WebSocket, plot_name: str) -> None:
        """Handle WebSocket connection for a plot using v0 protocol.

        Protocol v0 Flow:
        1. Server accepts connection
        2. Client sends `init` message (REQUIRED first message)
        3. Server validates and sends `config` message
        4. Client sends `refresh` to get first image
        5. Server sends binary image with 8-byte header
        6. Interactive session begins
        """
        # Validate plot exists BEFORE accepting connection
        if plot_name not in plot_generators:
            logger.warning(
                "WebSocket connection attempted for unknown plot: %s", plot_name
            )
            await websocket.close(code=1008, reason=f"Unknown plot: {plot_name}")
            return

        await websocket.accept()
        logger.debug("WebSocket accepted for plot '%s'", plot_name)

        # Track active connection
        router_state.connect(plot_name)

        # Generate unique connection ID for this WebSocket session
        connection_id = str(uuid.uuid4())
        logger.debug("Generated connection ID: %s", connection_id)

        config = plot_generators[plot_name]

        # Split query string into init params and _update.* params
        init_raw, update_raw = _split_query_params(dict(websocket.query_params))

        # Parse and validate init parameters
        try:
            params = config.init.params_model(**init_raw)
        except ValidationError as e:
            logger.warning("Invalid parameters for plot %s: %s", plot_name, e)
            await websocket.send_json(
                {"type": "error", "message": f"Invalid parameters: {e}"}
            )
            await websocket.close(code=1008, reason=f"Invalid params: {e}")
            return

        # Parse and validate _update.* parameters (if any)
        initial_update_params = None
        if update_raw:
            if config.update is None:
                logger.warning(
                    "Ignoring _update.* params for plot '%s' (no update configured)",
                    plot_name,
                )
            else:
                try:
                    initial_update_params = config.update.params_model(**update_raw)
                except ValidationError as e:
                    logger.warning(
                        "Invalid _update.* parameters for plot %s: %s", plot_name, e
                    )
                    await websocket.send_json(
                        {"type": "error", "message": f"Invalid update parameters: {e}"}
                    )
                    await websocket.close(
                        code=1008, reason=f"Invalid update params: {e}"
                    )
                    return

        # Wait for client `init` message as FIRST message (REQUIRED)
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected before init for plot '%s'", plot_name)
            return
        except Exception as e:
            logger.exception("Error receiving init message: %s", e)
            return

        # Validate this is the init message
        if data.get("type") != "init":
            logger.error("Expected 'init' as first message, got '%s'", data.get("type"))
            await websocket.send_json(
                {
                    "type": "error",
                    "message": "init must be the first client message",
                }
            )
            await websocket.close(code=1008, reason="init must be first message")
            return

        # Validate protocol version (REQUIRED field in init)
        client_version = data.get("protocol_version")
        if client_version is None:
            logger.error("Protocol version missing from init message")
            await websocket.send_json(
                {"type": "error", "message": "protocol_version is required in init"}
            )
            await websocket.close(code=1008, reason="Protocol version missing")
            return
        if client_version != PROTOCOL_VERSION:
            logger.error(
                "Incompatible protocol: server=%s, client=%s",
                PROTOCOL_VERSION,
                client_version,
            )
            await websocket.send_json(
                {
                    "type": "error",
                    "message": f"Incompatible protocol version. Server: {PROTOCOL_VERSION}, client: {client_version}",
                }
            )
            await websocket.close(
                code=1008,
                reason=f"Protocol mismatch: expected {PROTOCOL_VERSION}, got {client_version}",
            )
            return

        # Extract client configuration from init message
        device_pixel_ratio = data.get("device_pixel_ratio", 1.0)
        supports_binary = data.get("supports_binary", True)

        logger.info(
            "WebSocket initialized for plot '%s': params=%s, dpr=%s, binary=%s",
            plot_name,
            params,
            device_pixel_ratio,
            supports_binary,
        )

        loop = asyncio.get_running_loop()
        executor = _get_figure_executor()

        # Create figure and call generator to populate it in background thread
        fig = Figure()

        # Attach FastAPICanvas before passing into user code
        canvas = FastAPICanvas(fig)

        try:
            logger.debug("Initializing figure '%s' in background thread", plot_name)
            state = await loop.run_in_executor(
                executor,
                config.init.function,
                fig,
                params,
            )
        except Exception as e:
            logger.exception("Error generating plot '%s': %s", plot_name, e)
            await websocket.send_json(
                {"type": "error", "message": f"Plot generation failed: {e}"}
            )
            await websocket.close(code=1011, reason=f"Plot generation failed: {e}")
            return

        # If _update.* params were provided, apply them now
        if initial_update_params is not None and config.update is not None:
            try:
                logger.debug(
                    "Applying initial update params for '%s': %s",
                    plot_name,
                    initial_update_params,
                )
                state = await loop.run_in_executor(
                    executor,
                    config.update.function,
                    state,
                    initial_update_params,
                )
            except Exception as e:
                logger.exception(
                    "Error applying initial update for '%s': %s", plot_name, e
                )
                await websocket.send_json(
                    {"type": "error", "message": f"Initial update failed: {e}"}
                )
                await websocket.close(code=1011, reason=f"Initial update failed: {e}")
                return

        # Apply device pixel ratio
        if device_pixel_ratio != 1.0:
            if canvas._set_device_pixel_ratio(device_pixel_ratio):  # type: ignore[attr-defined]
                canvas._force_full = True
            logger.debug("Set device pixel ratio: %s", device_pixel_ratio)

        # Attach manager
        manager = FastAPIManger(canvas, 0)
        manager.supports_binary = supports_binary

        # Initialize image sequence tracker
        seq_state = ImageSequenceState()

        # Get toolbar configuration
        toolbar_config = FastAPIManger.get_toolbar_config()

        # Calculate figure size in CSS pixels
        width_inches, height_inches = fig.get_size_inches()
        width_px = round(width_inches * fig.dpi / canvas.device_pixel_ratio)
        height_px = round(height_inches * fig.dpi / canvas.device_pixel_ratio)

        # Build and send consolidated config message
        config_msg = {
            "type": "config",
            "protocol_version": PROTOCOL_VERSION,
            "connection_id": connection_id,
            "figure": {
                "size": [width_px, height_px],
                "dpi": fig.dpi,
                "label": fig.get_label(),
            },
            "toolbar": {
                "items": toolbar_config["toolbar_items"],
                "history": {"back": False, "forward": False},
            },
            "save": {
                "formats": toolbar_config["save_formats"],
                "default_format": toolbar_config["default_save_format"],
            },
            "image": {
                "format": "png",  # Currently only PNG supported
            },
            "update_schema": None,
            "init_params": params.model_dump(mode="json"),
            "update_params": (
                initial_update_params.model_dump(mode="json")
                if initial_update_params is not None
                else None
            ),
        }

        # Include update schema if available (precomputed at construction).
        config_msg["update_schema"] = _update_schemas[plot_name]

        await websocket.send_json(config_msg)
        logger.debug("Sent consolidated config message")

        # Track current update params for this connection
        current_update_params = initial_update_params

        # Event loop
        try:
            while True:
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected for plot '%s'", plot_name)
                    return
                except Exception as e:
                    logger.exception("Error receiving WebSocket message: %s", e)
                    return

                e_type = data.get("type")

                # Skip logging for high-frequency events
                if e_type not in ("motion_notify", "figure_enter", "figure_leave"):
                    logger.debug(
                        "Received message type='%s' for plot '%s'", e_type, plot_name
                    )

                try:
                    if e_type == "save_figure":
                        # Handle save request
                        try:
                            file_format = data.get("format", "png")
                            dpi = data.get("dpi", 100)
                            transparent = data.get("transparent", False)

                            # Validate format
                            supported_formats = toolbar_config["save_formats"]
                            format_lower = file_format.lower()
                            if format_lower not in supported_formats:
                                raise ValueError(
                                    f"Unsupported format '{file_format}'. "
                                    f"Supported: {', '.join(supported_formats)}"
                                )

                            # Generate unique file ID
                            file_id = str(uuid.uuid4())

                            # Save figure in thread pool
                            logger.debug(
                                "Saving figure '%s' to %s", plot_name, format_lower
                            )

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
                            router_state.saved_files[file_id] = saved_file
                            router_state.connection_files[connection_id].add(file_id)

                            # Clean old files
                            _clean_old_saved_files(router_state)

                            # Extract the router mount prefix from the
                            # WebSocket path "/{prefix}/ws/v0/{name}".
                            base_path = _prefix_before(websocket.url.path, "/ws/")

                            download_url = f"{base_path}/download/{file_id}"

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
                                "Saved figure '%s' as %s", plot_name, format_lower
                            )

                        except ValueError as e:
                            logger.warning("Invalid save request: %s", e)
                            await websocket.send_json(
                                {"type": "save_error", "message": str(e)}
                            )
                        except Exception as e:
                            logger.exception(
                                "Error saving figure '%s': %s", plot_name, e
                            )
                            await websocket.send_json(
                                {
                                    "type": "save_error",
                                    "message": f"Failed to save figure: {e!s}",
                                }
                            )

                    elif e_type == "update_params":
                        # Handle update request
                        if config.update is None:
                            logger.warning(
                                "Update requested for plot '%s' "
                                "but no update function configured",
                                plot_name,
                            )
                            continue
                        try:
                            update_params = config.update.params_model(**data["params"])
                        except ValidationError as e:
                            logger.warning(
                                "Invalid update parameters for plot %s: %s",
                                plot_name,
                                e,
                            )
                            continue
                        try:
                            logger.info(
                                "Updating plot '%s' with params: %s",
                                plot_name,
                                update_params,
                            )

                            state = await loop.run_in_executor(
                                executor,
                                config.update.function,
                                state,
                                update_params,
                            )

                            # Track current update params
                            current_update_params = update_params

                            # Only queue an invalidate if the update function
                            # did NOT already push a frame via blit().  When
                            # blit() was called, the image is already in
                            # _binary_queue and will be sent by _drain_canvas()
                            # below — sending an invalidate on top would cause
                            # the client to request a redundant render.
                            if not canvas._binary_queue:
                                canvas.draw_idle()

                        except Exception as e:
                            logger.exception(
                                "Error updating plot '%s': %s", plot_name, e
                            )

                    elif e_type == "render":
                        # Client requests render with binary image response.
                        # force_full=true means the client detected a gap and
                        # needs a self-contained FULL frame to resync from.
                        if data.get("force_full"):
                            canvas._force_full = True
                        await _send_render_response(
                            websocket, canvas, executor, loop, seq_state
                        )
                        continue  # Don't drain queue - response already sent

                    elif e_type == "refresh":
                        # Client requests full refresh
                        canvas._force_full = True
                        await _send_render_response(
                            websocket, canvas, executor, loop, seq_state
                        )
                        continue  # Don't drain queue - response already sent

                    else:
                        # Explicit dispatch table for canvas event handlers.
                        # Every allowable client event type is listed here;
                        # unknown types fall through to handle_unknown_event.
                        _canvas_handlers = {
                            "ack": canvas.handle_ack,
                            "resize": canvas.handle_resize,
                            "set_device_pixel_ratio": canvas.handle_set_device_pixel_ratio,
                            "send_image_mode": canvas.handle_send_image_mode,
                            "button_press": canvas.handle_button_press,
                            "button_release": canvas.handle_button_release,
                            "dblclick": canvas.handle_dblclick,
                            "figure_enter": canvas.handle_figure_enter,
                            "figure_leave": canvas.handle_figure_leave,
                            "motion_notify": canvas.handle_motion_notify,
                            "scroll": canvas.handle_scroll,
                            "toolbar_button": canvas.handle_toolbar_button,
                            "key_press": canvas.handle_key_press,
                            "key_release": canvas.handle_key_release,
                        }
                        handler = _canvas_handlers.get(
                            e_type, canvas.handle_unknown_event
                        )
                        await handler(data, websocket)

                    # Drain JSON queue and any binary frames queued by blit()
                    await _drain_canvas(canvas, websocket, seq_state)

                except Exception as e:
                    logger.exception("Error handling event '%s': %s", e_type, e)
        finally:
            # Cleanup on disconnect
            logger.debug("Cleaning up resources for plot '%s'", plot_name)

            # Decrement connection counter
            router_state.disconnect(plot_name)

            # Clean up saved files for this connection
            if connection_id in router_state.connection_files:
                file_ids = list(router_state.connection_files[connection_id])
                for file_id in file_ids:
                    _remove_saved_file(router_state, file_id)
                logger.debug(
                    "Removed %d saved file(s) for connection %s",
                    len(file_ids),
                    connection_id,
                )

            try:
                manager.destroy()
            except Exception as e:
                logger.exception("Error during cleanup: %s", e)

    # ── Content-hashed JS routes ─────────────────────────────────────────
    # These routes embed the content hash in the URL so `immutable` caching
    # is semantically correct.  Templates use these URLs directly.

    def _serve_hashed_js(
        request: Request,
        filename: str,
        url_hash: str,
        content_type: str = "application/javascript",
    ) -> Response:
        """Serve a JS file only if *url_hash* matches the current content."""
        result = _read_static_file(filename)
        if result is None:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        content, etag = result
        actual_hash = etag.strip('"')
        if url_hash != actual_hash:
            raise HTTPException(
                status_code=404,
                detail="Hash mismatch — bundle may have been redeployed",
            )
        if request.headers.get("if-none-match") == etag:
            return Response(
                status_code=304, headers={**_JS_IMMUTABLE_HEADERS, "ETag": etag}
            )
        return PlainTextResponse(
            content,
            headers={
                "Content-Type": content_type,
                **_JS_IMMUTABLE_HEADERS,
                "ETag": etag,
            },
        )

    @router.get("/component.{hash}.js")
    async def get_component_js_hashed(request: Request, hash: str) -> Response:
        """Serve the IIFE component bundle at a content-hashed URL."""
        return _serve_hashed_js(request, "component.js", hash)

    @router.get("/component.{hash}.esm.js")
    async def get_component_esm_js_hashed(request: Request, hash: str) -> Response:
        """Serve the ESM component bundle at a content-hashed URL."""
        return _serve_hashed_js(request, "component.esm.js", hash)

    @router.get("/component.{hash}.js.map")
    async def get_component_js_map_hashed(request: Request, hash: str) -> Response:
        """Serve the IIFE source map at a content-hashed URL."""
        return _serve_hashed_js(
            request, "component.js.map", hash, content_type="application/json"
        )

    @router.get("/component.{hash}.esm.js.map")
    async def get_component_esm_js_map_hashed(request: Request, hash: str) -> Response:
        """Serve the ESM source map at a content-hashed URL."""
        return _serve_hashed_js(
            request, "component.esm.js.map", hash, content_type="application/json"
        )

    # ── Stable redirect routes ──────────────────────────────────────────
    # Hash-free URLs redirect to the current content-hashed URL.
    # External embedders and docs can reference these stable paths.

    def _redirect_to_hashed(filename: str, url_template: str) -> RedirectResponse:
        """Build a redirect from a stable URL to the content-hashed variant."""
        file_hash = _get_js_hash(filename)
        if file_hash is None:
            raise HTTPException(status_code=404, detail=f"{filename} not found")
        return RedirectResponse(
            url=url_template.format(hash=file_hash),
            status_code=302,
            headers=_JS_REDIRECT_HEADERS,
        )

    def _register_redirect(route: str, filename: str, url_template: str) -> None:
        """Register a stable hash-free URL that 302s to the hashed bundle."""

        async def _redirect() -> RedirectResponse:
            return _redirect_to_hashed(filename, url_template)

        _redirect.__doc__ = f"Redirect {route} to the content-hashed bundle."
        router.add_api_route(route, _redirect, methods=["GET"])

    # (stable route, source file, hashed URL template).  The legacy
    # ``/js/mpl.js`` path is kept as an alias for the IIFE bundle.
    _redirect_routes: list[tuple[str, str, str]] = [
        ("/js/mpl.js", "component.js", "component.{hash}.js"),
        ("/component.js", "component.js", "component.{hash}.js"),
        ("/component.esm.js", "component.esm.js", "component.{hash}.esm.js"),
        ("/component.js.map", "component.js.map", "component.{hash}.js.map"),
        (
            "/component.esm.js.map",
            "component.esm.js.map",
            "component.{hash}.esm.js.map",
        ),
    ]
    for _route, _filename, _template in _redirect_routes:
        _register_redirect(_route, _filename, _template)

    # Route: Get schema for a specific plot
    @router.get("/api/plots/{plot_name}/schema", dependencies=[Depends(_http_auth)])
    async def get_plot_schema(
        plot_name: str,
        _: None = Depends(validate_plot_name),
    ) -> dict[str, Any]:
        """Get the parameter schemas for a specific plot."""
        config = plot_generators[plot_name]

        return {
            "plot_name": plot_name,
            "description": config.description,
            "init_schema": _init_schemas[plot_name],
            "update_schema": _update_schemas[plot_name],
        }

    return MPLRouter(
        router=router,
        static_files=static_files,
        static_mount_path=static_mount_path,
        state=router_state,
    )
