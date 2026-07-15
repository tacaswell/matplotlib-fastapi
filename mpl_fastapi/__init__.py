def _get_version() -> str:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # Only call setuptools_scm when inside the real mpl-fastapi git checkout.
    # The sentinel file (.mpl-fastapi-repo) is present in the git repo and in
    # sdists, but .git is not included in sdists. This combination prevents
    # setuptools_scm from accidentally reading a parent git repo when the
    # sdist is unpacked inside one. Shallow clones (used by many CI systems)
    # also trigger a warning from setuptools_scm, so skip those too.
    if (
        (root / ".mpl-fastapi-repo").exists()
        and (root / ".git").exists()
        and not (root / ".git/shallow").exists()
    ):
        try:
            import setuptools_scm

            return setuptools_scm.get_version(root=root)
        except Exception:
            pass
    # Fall back to the version written into _version.py at build time.
    try:
        from . import _version

        return _version.version
    except ImportError:
        return "unknown"


__version__ = _get_version()

from ._server import build_app  # noqa: E402  (after __version__ computation)
from .auth import (  # noqa: E402
    AuthPolicy,
    NoAuth,
    Principal,
    SingleUserToken,
    TokenPrincipal,
)
from .router import (  # noqa: E402
    ConnectionInfo,
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
    "ConnectionInfo",
    "InitConfig",
    "Lifespan",
    "MPLRouter",
    "NoAuth",
    "PlotConfig",
    "PlotGenerator",
    "PlotInfo",
    "PlotsListResponse",
    "Principal",
    "SingleUserToken",
    "TokenPrincipal",
    "UpdateConfig",
    "UpdateFunction",
    "__version__",
    "build_app",
    "compose_lifespans",
    "create_mpl_router",
    "install_mpl_router",
    "shutdown_figure_executor",
]
