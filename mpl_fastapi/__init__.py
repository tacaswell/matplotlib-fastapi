try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .router import (
    MPLRouter,
    PlotGenerator,
    PlotInfo,
    PlotsListResponse,
    create_mpl_router,
)

__all__ = [
    "__version__",
    "create_mpl_router",
    "MPLRouter",
    "PlotGenerator",
    "PlotInfo",
    "PlotsListResponse",
]
