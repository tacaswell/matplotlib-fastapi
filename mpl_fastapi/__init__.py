try:
    from ._version import __version__
except ImportError:
    __version__ = "unknown"

from .main import create_mpl_router, MPLRouter, PlotGenerator

__all__ = [
    "__version__",
    "create_mpl_router",
    "MPLRouter",
    "PlotGenerator",
]
