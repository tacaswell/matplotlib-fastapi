"""Export the OpenAPI schema from the mpl-fastapi router to a JSON file.

Run this script before building the Sphinx docs:
    python docs/export_openapi.py
"""

import json
import pathlib

from fastapi import FastAPI

from mpl_fastapi.router import create_mpl_router

# Minimal app wrapping the router so FastAPI can generate the schema.
app = FastAPI(
    title="mpl-fastapi",
    description="FastAPI router for serving interactive Matplotlib figures via WebSocket.",
    version="0.0.1",
)

mpl_router = create_mpl_router(plot_generators={})
app.include_router(mpl_router.router)

schema = app.openapi()

output = pathlib.Path(__file__).parent / "source" / "_static" / "openapi.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(schema, indent=2))
print(f"Wrote OpenAPI schema to {output}")
