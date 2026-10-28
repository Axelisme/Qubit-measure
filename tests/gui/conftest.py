"""Shared pytest fixtures for GUI tests."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# Force a deterministic headless-safe backend for GUI tests.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication with a headless-safe platform."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(scope="module", autouse=True)
def plotting_state_policy(request: pytest.FixtureRequest) -> Iterator[None]:
    """Identify modules that leak shutdown state or live figure attachments."""
    from zcu_tools.gui.plotting import dump_plot_state, is_shutting_down

    before = (is_shutting_down(), dump_plot_state())
    yield
    after = (is_shutting_down(), dump_plot_state())
    assert after == before, (
        f"plotting state polluter: {request.node.path}; "
        f"before={before!r}; after={after!r}"
    )


@pytest.fixture(params=["white", "#202020"], ids=["light", "dark"])
def window_palette(qapp, request):
    """A synthetic Window palette without changing application-wide colours."""
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ColorRole.Window, QColor(request.param))
    return palette


@pytest.fixture(autouse=True, name="_drain_qt_events")
def fixture_drain_qt_events(qapp):
    """Drain pending Qt events before and after every test.

    Each test process re-uses its session ``QApplication`` across many tests.
    A ``BackgroundRunner`` task delivers its outcome to the main thread via a
    *queued* cross-thread signal (a ``QMetaCallEvent``), and a dedicated
    ``_OpWorker`` posts a ``deleteLater`` (a ``DeferredDelete`` event); both sit
    in the main-thread queue after the logical work has finished. If the runner
    (and the QObjects it owns) are GC'd while one of those events is still
    queued, a later test's event pump (a ``QEventLoop.exec`` or ``processEvents``)
    dispatches it onto a freed C++ object and segfaults.

    The DETERMINISTIC guard against that is NOT this drain: every fixture that
    spawns a real worker must call ``BackgroundRunner.quiesce`` (join the worker
    threads, then flush their queued deliveries) BEFORE its owning object goes
    out of scope — see ``tests/gui/test_controller.py`` (the ``cf`` fixture) and
    ``tests/gui/session/services/test_device*.py`` (the ``_LIVE_BG`` quiesce fixtures).
    This per-test drain is only a cheap best-effort flush of any stray queued
    event so it lands while the QApplication is healthy; it does NOT join worker
    threads and is not sufficient on its own.
    """
    qapp.processEvents()
    qapp.processEvents()
    yield
    qapp.processEvents()
    qapp.processEvents()
