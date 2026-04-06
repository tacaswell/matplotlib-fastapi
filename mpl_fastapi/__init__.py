try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .auth import AuthPolicy, NoAuth, SingleUserToken
from ._server import build_app
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
    "AuthPolicy",
    "build_app",
    "InitConfig",
    "Lifespan",
    "MPLRouter",
    "NoAuth",
    "PlotConfig",
    "PlotGenerator",
    "PlotInfo",
    "PlotsListResponse",
    "SingleUserToken",
    "UpdateConfig",
    "UpdateFunction",
    "__version__",
    "compose_lifespans",
    "create_mpl_router",
    "install_mpl_router",
    "shutdown_figure_executor",
]
