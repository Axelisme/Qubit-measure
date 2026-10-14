"""Shared fixtures for MCP measure tests.

Every test runs without process-global MCP call-log files. Tests that drive a
real GUI seam request ``qapp`` for one headless QApplication per session.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

from ._support import MeasureClient


@pytest.fixture(autouse=True)
def disable_call_log(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZCU_MCP_CALL_LOG", "0")


@pytest.fixture
def clients(tmp_path: Path) -> Iterator[list[MeasureClient]]:
    created: list[MeasureClient] = []
    yield created
    for client in created:
        client.context.session.close()


@pytest.fixture(scope="session")
def qapp() -> Iterator[QApplication]:
    """Keep a headless Qt owner alive for tests that drive a real GUI seam."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("QT_QPA_PLATFORM", "offscreen")
        patch.setenv("QT_QPA_PLATFORMTHEME", "generic")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        assert isinstance(app, QApplication)
        yield app
