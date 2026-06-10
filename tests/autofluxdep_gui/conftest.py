"""Shared fixtures for autofluxdep-gui tests (ui and non-ui).

The session core is async and Qt-backed: a controller composes the shared session
services (ConnectionService / DeviceService / BackgroundService — all QObjects)
at construction, and establishing a mock SoC goes through ConnectionService's
``QTimer.singleShot`` settle. So a ``QApplication`` must exist *before* any
``build_core()`` (a QObject created with no app gets its C++ side torn down) and
to pump the connect loop. The ``qapp`` fixture is therefore ``autouse`` — it is
created once at session start, ahead of every test. This is the same ``qapp`` +
``QEventLoop`` pattern measure-gui's tests use (``tests/gui/conftest.py`` /
``tests/gui/services/test_connection.py``).
"""

from __future__ import annotations

import os

import pytest

# Force a deterministic headless-safe Qt platform before any QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def qapp():
    """A single offscreen QApplication for the test session (created before any
    test body, so controller QObjects are constructed against a live app)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=True)
def _drain_qt_events(qapp):
    """Drain pending Qt events before and after every test (xdist segfault prevention).

    See tests/gui/conftest.py for the full rationale.
    """
    qapp.processEvents()
    qapp.processEvents()
    yield
    qapp.processEvents()
    qapp.processEvents()
