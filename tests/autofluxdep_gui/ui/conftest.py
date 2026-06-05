"""Headless Qt fixtures for autofluxdep-gui UI tests."""

from __future__ import annotations

import os

import pytest

# Force a deterministic headless-safe Qt platform before any QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from qtpy.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A single offscreen QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
