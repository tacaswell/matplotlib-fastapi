try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .router import (
    InitConfig,
    Lifespan,
    MPLRouter,
    PlotConfig,
    PlotGenerator,
    PlotInfo,
    PlotsListResponse,
    UpdateConfig,
    UpdateFunction,
    compose_lifespans,
    create_mpl_router,
    install_mpl_router,
    shutdown_figure_executor,
)

__all__ = [
    "InitConfig",
    "Lifespan",
    "MPLRouter",
    "PlotConfig",
    "PlotGenerator",
    "PlotInfo",
    "PlotsListResponse",
    "UpdateConfig",
    "UpdateFunction",
    "__version__",
    "compose_lifespans",
    "create_mpl_router",
    "install_mpl_router",
    "shutdown_figure_executor",
]
