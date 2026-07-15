"""CLI entry-point for mpl-fastapi.

Accepts a target in uvicorn's ``module:attribute`` form:

    python -m mpl_fastapi package.module:attribute [uvicorn options…]

Both parts are required.  The attribute must be a ``dict[str, PlotConfig]``.
Everything else (auth, middleware, static files, lifespan) is handled
automatically.

Examples::

    python -m mpl_fastapi demos.plots:plots --host 0.0.0.0 --port 8080 --reload
    python -m mpl_fastapi mypackage.figures:my_registry --port 8080
    python -m mpl_fastapi demos.plots:plots --no-auth --reload

Flags
-----
``--no-auth``
    Disable authentication (serve without the bearer token requirement).

Environment variables
---------------------
``MPL_FASTAPI_TOKEN``
    Pre-set the authentication token instead of generating a random one.
    Set to an empty string (``MPL_FASTAPI_TOKEN=""``) to disable authentication.
``HOST`` / ``PORT``
    Override the host/port shown in the startup log (defaults: 127.0.0.1 / 8000).
``BACKEND_CORS_ORIGINS``
    Comma-separated list of origins permitted to make cross-origin requests::

        BACKEND_CORS_ORIGINS=https://app.example.com,http://localhost:3000 \\
            python -m mpl_fastapi demos.plots:plots --reload

    Defaults to empty (deny all cross-origin requests).
"""

from __future__ import annotations

import importlib
import sys

from mpl_fastapi.router import PlotConfig


def _load_plots(target: str) -> dict[str, PlotConfig]:
    """Resolve ``module:attr`` and return the validated plots dict."""
    if ":" not in target:
        print(
            f"error: target must be in 'module:attribute' form, got '{target}'",
            file=sys.stderr,
        )
        sys.exit(1)

    module_path, attr = target.rsplit(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        print(f"error: cannot import '{module_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    plots = getattr(module, attr, None)
    if plots is None:
        print(f"error: '{module_path}' has no attribute '{attr}'", file=sys.stderr)
        sys.exit(1)

    if not isinstance(plots, dict):
        print(
            f"error: '{target}' must be a dict, got {type(plots).__name__}",
            file=sys.stderr,
        )
        sys.exit(1)

    bad = {
        k: type(v).__name__ for k, v in plots.items() if not isinstance(v, PlotConfig)
    }
    if bad:
        print(
            f"error: all values in '{target}' must be PlotConfig instances.\n"
            f"  offending keys: {', '.join(f'{k!r} ({t})' for k, t in bad.items())}",
            file=sys.stderr,
        )
        sys.exit(1)

    return plots


def main() -> None:
    import uvicorn

    args = sys.argv[1:]

    if not args or args[0].startswith("-"):
        print(
            "usage: python -m mpl_fastapi <module:attribute> [options…]\n\n"
            "  examples:\n"
            "    python -m mpl_fastapi demos.plots:plots\n"
            "    python -m mpl_fastapi mypackage.figures:registry --host 0.0.0.0 --port 8080\n"
            "    python -m mpl_fastapi demos.plots:plots --no-auth --reload\n\n"
            "  options:\n"
            "    --no-auth                   Disable bearer token authentication\n"
            "    [other uvicorn options]\n\n"
            "  CORS: set BACKEND_CORS_ORIGINS=https://a.com,http://localhost:3000\n"
            "  Token: set MPL_FASTAPI_TOKEN='my-token' or MPL_FASTAPI_TOKEN='' (empty = no auth)",
            file=sys.stderr,
        )
        sys.exit(1)

    target, *uvicorn_args = args

    # Validate the target before handing off to uvicorn.
    _load_plots(target)

    import os

    os.environ["_MPL_FASTAPI_TARGET"] = target

    # Extract --no-auth flag if present (before passing to uvicorn)
    if "--no-auth" in uvicorn_args:
        os.environ["_MPL_FASTAPI_NO_AUTH"] = "1"
        uvicorn_args.remove("--no-auth")

    # Extract --host and --port if present to set environment variables
    # so the startup log displays the correct URLs
    for i, arg in enumerate(uvicorn_args):
        if arg == "--host" and i + 1 < len(uvicorn_args):
            os.environ["HOST"] = uvicorn_args[i + 1]
        elif arg == "--port" and i + 1 < len(uvicorn_args):
            os.environ["PORT"] = uvicorn_args[i + 1]
        elif arg.startswith("--host="):
            os.environ["HOST"] = arg.split("=", 1)[1]
        elif arg.startswith("--port="):
            os.environ["PORT"] = arg.split("=", 1)[1]

    # Pass args directly to Click so sys.argv is never mutated.
    uvicorn.main(args=["mpl_fastapi._app:app", *uvicorn_args], standalone_mode=True)


if __name__ == "__main__":
    main()
