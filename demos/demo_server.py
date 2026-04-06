"""
Demo application showing how to use mpl-fastapi.

Run with:
    uvicorn demos.demo_server:app --reload

Then visit the URL printed to the console.
"""

import logging
from pathlib import Path

from mpl_fastapi import build_app
from mpl_fastapi.__main__ import _load_plots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logging.getLogger('mpl_fastapi').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

plots = _load_plots("demos.plots:plots")
app = build_app(plots)


# ---------------------------------------------------------------------------
# Extra demo routes (home page, embeddable demo, React example)
# ---------------------------------------------------------------------------

@app.get("/")
async def home():
    """Serve the main menu page."""
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


@app.get("/embeddable")
async def embeddable_demo():
    """Serve the embeddable component demo page."""
    from fastapi.responses import FileResponse
    return FileResponse(Path(__file__).parent / "embeddable_demo.html", media_type="text/html")


_react_build_path = Path(__file__).parent / "react-example" / "dist"


def _mount_react_app() -> None:
    from fastapi.responses import HTMLResponse
    from starlette.staticfiles import StaticFiles

    if _react_build_path.exists():
        app.mount(
            "/react-app",
            StaticFiles(directory=_react_build_path, html=True),
            name="react_app",
        )
        logger.info("React app mounted from %s", _react_build_path)
    else:
        logger.warning("React build not found at %s; serving build instructions", _react_build_path)

        @app.get("/react-app")
        @app.get("/react-app/{path:path}")
        async def react_not_built(path: str = ""):  # noqa: ARG001
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


app.add_event_handler("startup", _mount_react_app)
