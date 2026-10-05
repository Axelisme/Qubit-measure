"""Optional headless Qt fixtures for this Autofluxdep test owner.

Qt-backed controller cases opt in with explicit ``usefixtures("qapp",
"drain_qt_events")``. ``qapp`` has session scope; neither fixture is autouse.
The remaining pure InfoTracker cases do not request either fixture and do not
create a QApplication. Shared plotter cases have their own explicit fixtures
under ``tests/zcu_lab/v2/_support/autofluxdep``.
"""

from __future__ import annotations

import os

import pytest
from qtpy.QtWidgets import QApplication

# Force a deterministic headless-safe Qt platform before any QApplication.
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["QT_QPA_PLATFORMTHEME"] = "generic"


@pytest.fixture(scope="session", autouse=False)
def qapp():
    """Create one offscreen QApplication before an explicitly opted-in case."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture(autouse=False)
def drain_qt_events(qapp):
    """Drain pending Qt events before and after an explicitly opted-in case.

    See tests/gui/conftest.py for the full rationale.
    """
    qapp.processEvents()
    qapp.processEvents()
    yield
    qapp.processEvents()
    qapp.processEvents()
