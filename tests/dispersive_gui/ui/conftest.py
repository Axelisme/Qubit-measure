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
