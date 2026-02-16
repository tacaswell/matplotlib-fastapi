try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .router import (
    InitConfig,
    MPLRouter,
    PlotConfig,
    PlotGenerator,
    PlotInfo,
    PlotsListResponse,
    UpdateConfig,
    UpdateFunction,
    create_mpl_router,
    shutdown_figure_executor,
)

__all__ = [
    "__version__",
    "create_mpl_router",
    "shutdown_figure_executor",
    "InitConfig",
    "MPLRouter",
    "PlotConfig",
    "PlotGenerator",
    "PlotInfo",
    "PlotsListResponse",
    "UpdateConfig",
    "UpdateFunction",
]
