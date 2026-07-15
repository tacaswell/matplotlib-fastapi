"""
Demo application showing how to use mpl-fastapi.

Run with:
    uvicorn demos.demo_server:app --reload

Then visit the URL printed to the console.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse

from mpl_fastapi import build_app
from mpl_fastapi.__main__ import _load_plots

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logging.getLogger("mpl_fastapi").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

plots = _load_plots("demos.plots:plots")

_react_build_path = Path(__file__).parent / "react-example" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Handle startup and shutdown events."""
    # Startup: mount React app
    from starlette.staticfiles import StaticFiles

    if _react_build_path.exists():
        app.mount(
            "/react-app",
            StaticFiles(directory=_react_build_path, html=True),
            name="react_app",
        )
        logger.info("React app mounted from %s", _react_build_path)
    else:
        logger.warning(
            "React build not found at %s; serving build instructions", _react_build_path
        )

        @app.get("/react-app")
        @app.get("/react-app/{path:path}")
        async def react_not_built(path: str = "") -> HTMLResponse:  # noqa: ARG001
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

    yield  # App is running

    # Shutdown (if needed in the future)


app = build_app(plots, lifespan=lifespan)


# ---------------------------------------------------------------------------
# Extra demo routes (home page, client demo)
# ---------------------------------------------------------------------------


@app.get("/")
async def home() -> FileResponse:
    """Serve the main menu page."""
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/client")
async def client_demo() -> FileResponse:
    """Serve the JavaScript client demo page."""
    return FileResponse(
        Path(__file__).parent / "client_demo.html", media_type="text/html"
    )
