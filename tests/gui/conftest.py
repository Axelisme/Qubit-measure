"""Shared pytest fixtures for GUI tests."""

from __future__ import annotations

import os

import pytest

# Force a deterministic headless-safe backend for GUI tests.
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from qtpy.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    """Provide a single QApplication with a headless-safe platform."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
