"""Optional OpenTelemetry integration for mpl-fastapi.

When the ``opentelemetry`` packages are installed, this module provides:

- Automatic FastAPI request/response span creation via ``FastAPIInstrumentor``
- Helpers to obtain a tracer and meter (real or no-op)
- A composable lifespan for setup/teardown
- Log correlation (``trace_id`` / ``span_id`` injected into stdlib log records)

All exporter configuration is handled by the OpenTelemetry SDK through
standard ``OTEL_*`` environment variables — no code-level exporter wiring
is needed.  When the packages are not installed every helper is a silent
no-op, so the rest of the codebase can call these unconditionally.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Import guards
# ---------------------------------------------------------------------------
try:
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.metrics import get_meter as _otel_get_meter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.trace import StatusCode

    HAS_OTEL = True
except ImportError:  # pragma: no cover
    HAS_OTEL = False


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_tracer(name: str) -> Any:
    """Return an OpenTelemetry tracer, or a no-op stand-in.

    Parameters
    ----------
    name : str
        Instrumentation scope name, e.g. ``"mpl_fastapi.router"``.
    """
    if HAS_OTEL:
        return trace.get_tracer(name)
    return _NoOpTracer()


def get_meter(name: str) -> Any:
    """Return an OpenTelemetry meter, or a no-op stand-in.

    Parameters
    ----------
    name : str
        Instrumentation scope name, e.g. ``"mpl_fastapi.router"``.
    """
    if HAS_OTEL:
        return _otel_get_meter(name)
    return _NoOpMeter()


def capture_context() -> Any:
    """Snapshot the current OTel context for cross-thread propagation.

    Call this on the async side, then pass the token to the sync wrapper
    running in :func:`~concurrent.futures.ThreadPoolExecutor`.
    """
    if HAS_OTEL:
        return otel_context.get_current()
    return None


def attach_context(ctx: Any) -> Any:
    """Attach a previously captured context in the current thread.

    Returns a token that **must** be passed to :func:`detach_context`
    when the work is done (use a ``try/finally``).
    """
    if HAS_OTEL and ctx is not None:
        return otel_context.attach(ctx)
    return None


def detach_context(token: Any) -> None:
    """Detach a context previously attached via :func:`attach_context`."""
    if HAS_OTEL and token is not None:
        otel_context.detach(token)


def set_span_error(span: Any, exception: BaseException) -> None:
    """Record an exception on a span if OTel is available."""
    if HAS_OTEL and span is not None:
        span.set_status(StatusCode.ERROR, str(exception))
        span.record_exception(exception)


# ---------------------------------------------------------------------------
# Setup / teardown
# ---------------------------------------------------------------------------

def setup_telemetry(app: FastAPI, *, service_name: str = "mpl-fastapi") -> None:
    """Instrument a FastAPI application with OpenTelemetry.

    This is a no-op when the ``opentelemetry`` packages are not installed
    or when ``OTEL_SDK_DISABLED=true`` is set in the environment.

    Parameters
    ----------
    app : FastAPI
        The application to instrument.
    service_name : str
        Fallback service name — overridden by ``OTEL_SERVICE_NAME`` env var.
    """
    if not HAS_OTEL:
        logger.debug("OpenTelemetry packages not installed — skipping instrumentation")
        return

    import os

    if os.environ.get("OTEL_SDK_DISABLED", "").lower() == "true":
        logger.debug("OTEL_SDK_DISABLED=true — skipping instrumentation")
        return

    # Set a default service name if the env var is absent.
    os.environ.setdefault("OTEL_SERVICE_NAME", service_name)

    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry instrumentation enabled for %s", service_name)


def teardown_telemetry(app: FastAPI) -> None:
    """Remove OpenTelemetry instrumentation and flush providers.

    Parameters
    ----------
    app : FastAPI
        Previously instrumented application.
    """
    if not HAS_OTEL:
        return

    try:
        FastAPIInstrumentor.uninstrument_app(app)
    except Exception:
        logger.debug("FastAPIInstrumentor.uninstrument_app failed", exc_info=True)

    # Flush any buffered spans / metrics so nothing is lost on shutdown.
    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        try:
            provider.force_flush(timeout_millis=5000)
        except Exception:
            logger.debug("Tracer provider flush failed", exc_info=True)


# ---------------------------------------------------------------------------
# Composable lifespan
# ---------------------------------------------------------------------------

def otel_lifespan():
    """Return a lifespan context manager for OpenTelemetry setup/teardown.

    Designed to be composed with other lifespans via
    :func:`~mpl_fastapi.compose_lifespans`::

        from mpl_fastapi._otel import otel_lifespan
        from mpl_fastapi.router import compose_lifespans

        app = FastAPI(lifespan=compose_lifespans(otel_lifespan(), other_lifespan))
    """

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_telemetry(app)
        try:
            yield
        finally:
            teardown_telemetry(app)

    return _lifespan


# ---------------------------------------------------------------------------
# No-op stubs (used when opentelemetry is not installed)
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Minimal stand-in for ``opentelemetry.trace.Span``."""

    def set_attribute(self, key: str, value: Any) -> None: ...
    def set_status(self, status: Any, description: str | None = None) -> None: ...
    def record_exception(self, exception: BaseException) -> None: ...
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None: ...
    def end(self) -> None: ...
    def __enter__(self) -> _NoOpSpan:
        return self
    def __exit__(self, *args: object) -> None: ...


class _NoOpTracer:
    """Minimal stand-in for ``opentelemetry.trace.Tracer``."""

    def start_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()

    def start_as_current_span(self, name: str, **kwargs: Any) -> _NoOpSpan:
        return _NoOpSpan()


class _NoOpCounter:
    def add(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None: ...


class _NoOpHistogram:
    def record(self, value: int | float, attributes: dict[str, Any] | None = None) -> None: ...


class _NoOpUpDownCounter:
    def add(self, amount: int | float, attributes: dict[str, Any] | None = None) -> None: ...


class _NoOpMeter:
    """Minimal stand-in for ``opentelemetry.metrics.Meter``."""

    def create_counter(self, name: str, **kwargs: Any) -> _NoOpCounter:
        return _NoOpCounter()

    def create_histogram(self, name: str, **kwargs: Any) -> _NoOpHistogram:
        return _NoOpHistogram()

    def create_up_down_counter(self, name: str, **kwargs: Any) -> _NoOpUpDownCounter:
        return _NoOpUpDownCounter()
