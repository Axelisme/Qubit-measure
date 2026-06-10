"""Headless Qt fixtures for dispersive-fit-gui UI tests."""

from __future__ import annotations

import os

import pytest

# Force a deterministic headless-safe Qt platform before any QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A single offscreen QApplication for the test session."""
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
