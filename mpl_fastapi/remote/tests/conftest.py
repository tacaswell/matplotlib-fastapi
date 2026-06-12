"""Pytest configuration for the Qt/Tk remote-backend tests.

The PySide6 / pytest-qt tests construct a ``QApplication``.  Without a windowing
system Qt's default ``xcb`` platform plugin cannot connect to a display server
and calls ``qFatal()`` -> ``abort()``, which kills the whole pytest process
rather than failing a single test.  In headless environments (CI, containers,
sandboxes) there is no display, so default ``QT_QPA_PLATFORM`` to the headless
``offscreen`` plugin.

A developer running on a real desktop can still exercise the genuine windowing
backend by exporting ``QT_QPA_PLATFORM`` (or any of ``DISPLAY`` /
``WAYLAND_DISPLAY``) before invoking pytest; this only sets a default when none
of those are already present.
"""

import os

# Only override when the user has not made an explicit choice and no display is
# advertised.  This must happen at import time, before any ``QApplication`` is
# instantiated by the pytest-qt ``qapp`` fixture.
if (
    "QT_QPA_PLATFORM" not in os.environ
    and not os.environ.get("DISPLAY")
    and not os.environ.get("WAYLAND_DISPLAY")
):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
