"""Explicit headless Qt fixtures for the moved shared plotter cases."""

from __future__ import annotations

import os

import pytest
from qtpy.QtWidgets import QApplication

# Force a deterministic headless-safe Qt platform before any QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"


@pytest.fixture(scope="session", autouse=False)
def qapp():
    """A single offscreen QApplication for the test session (created before any
    test body, so controller QObjects are constructed against a live app)."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=False)
def drain_qt_events(qapp):
    """Drain pending Qt events before and after every test (xdist segfault prevention).

    See tests/gui/conftest.py for the full rationale.
    """
    qapp.processEvents()
    qapp.processEvents()
    yield
    qapp.processEvents()
    qapp.processEvents()
