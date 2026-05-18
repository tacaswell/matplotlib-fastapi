"""Shared CLI helpers for the mpl_fastapi remote launcher entry points.

Both :mod:`~mpl_fastapi.remote.backend_tkremote` and
:mod:`~mpl_fastapi.remote.backend_qtremote` can be run as modules (``python
-m mpl_fastapi.remote.backend_tkremote``) and are registered as console
scripts.  This module provides the common argument-parsing and token
resolution logic so neither backend duplicates it.

Authentication token
--------------------
The authentication token is resolved in this order:

1. The ``--token`` CLI argument.
2. The ``MPL_FASTAPI_TOKEN`` environment variable.

If neither is set the client connects without a token, which works when
the server has authentication disabled.
"""

from __future__ import annotations

import argparse
import os


ENV_TOKEN = "MPL_FASTAPI_TOKEN"
"""Environment variable used to supply the authentication token.

Set this variable to avoid passing ``--token`` on every invocation::

    export MPL_FASTAPI_TOKEN=my-secret-token
    python -m mpl_fastapi.remote.backend_tkremote
"""

DEFAULT_URL = "ws://localhost:8000/plots"


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    """Return an :class:`argparse.ArgumentParser` for a launcher entry point.

    Parameters
    ----------
    description : str
        Short description shown in ``--help`` output.

    Returns
    -------
    argparse.ArgumentParser
        Parser with ``--url``, ``--plot``, ``--token``, ``params``, and
        ``--update`` arguments all pre-configured.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=(
            "Base URL of the mpl-fastapi router mount point "
            f"(default: {DEFAULT_URL}).  Must include the router prefix, "
            "e.g. ws://host:8000/plots or http://host:8000/plots."
        ),
    )
    parser.add_argument(
        "--plot",
        default=None,
        help=(
            "Plot name to open directly.  The launcher window is "
            "always shown; this pre-opens the named plot as well."
        ),
    )
    parser.add_argument(
        "params",
        nargs="*",
        metavar="KEY=VALUE",
        help="Init parameters as key=value pairs (e.g. frequency=3.0)",
    )
    parser.add_argument(
        "--update",
        nargs="*",
        metavar="KEY=VALUE",
        default=[],
        help="Update parameters as key=value pairs (e.g. phase=1.57)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Authentication token. "
            f"Defaults to the {ENV_TOKEN!r} environment variable if set."
        ),
    )
    return parser


def resolve_token(cli_token: str | None) -> str | None:
    """Return the effective authentication token.

    Resolution order:

    1. *cli_token* (the value of ``--token``).
    2. The :data:`MPL_FASTAPI_TOKEN` environment variable.
    3. ``None`` (unauthenticated).

    Parameters
    ----------
    cli_token : str or None
        Token supplied on the command line, or ``None`` if omitted.

    Returns
    -------
    str or None
        Token to use, or ``None`` for unauthenticated access.
    """
    if cli_token is not None:
        return cli_token
    return os.environ.get(ENV_TOKEN, "").strip() or None


def parse_kv_pairs(
    pairs: list[str],
    parser: argparse.ArgumentParser,
    label: str = "Parameters",
) -> dict[str, str]:
    """Parse a list of ``KEY=VALUE`` strings into a dictionary.

    Parameters
    ----------
    pairs : list[str]
        Raw strings from the argument parser.
    parser : argparse.ArgumentParser
        Used to emit a clean error message on malformed input.
    label : str
        Human-readable label used in the error message.

    Returns
    -------
    dict[str, str]
        Mapping of keys to (string) values.
    """
    result: dict[str, str] = {}
    for kv in pairs:
        if "=" not in kv:
            parser.error(f"{label} must be KEY=VALUE, got: {kv!r}")
        key, value = kv.split("=", 1)
        result[key] = value
    return result
