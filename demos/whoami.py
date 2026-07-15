"""
Context-aware plot demo for mpl-fastapi.

Run it with:
    python -m mpl_fastapi demos/whoami.py
"""

from matplotlib.figure import Figure
from pydantic import BaseModel, Field

from mpl_fastapi import InitConfig, PlotConfig
from mpl_fastapi.router import ConnectionInfo


class WhoAmIParams(BaseModel):
    """Parameters for the context demo plot."""

    fontsize: int = Field(
        default=16, ge=8, le=48, description="Label font size in points"
    )


def create_whoami_plot(
    fig: Figure, params: WhoAmIParams, *, context: ConnectionInfo
) -> None:
    """Render the identity of the connected principal.

    Opts in to server-side context by declaring a keyword-only ``context``
    parameter.  ``context.principal`` is whatever the router's auth policy
    returned for this connection (``None`` under ``NoAuth``); a real
    AuthN/AuthZ policy would surface the authenticated user here, ready for
    on-behalf-of calls.
    """
    ax = fig.subplots()
    ax.axis("off")
    ax.text(
        0.5,
        0.6,
        f"principal: {context.principal!r}",
        ha="center",
        va="center",
        fontsize=params.fontsize,
    )
    ax.text(
        0.5,
        0.4,
        f"connection: {context.connection_id[:8]}…",
        ha="center",
        va="center",
        fontsize=params.fontsize * 0.6,
        color="gray",
    )
    context.logger.info("rendered whoami plot")


plots = {
    "whoami": PlotConfig(
        description="Context-aware plot showing the authenticated principal",
        init=InitConfig(function=create_whoami_plot, params_model=WhoAmIParams),
    ),
}
