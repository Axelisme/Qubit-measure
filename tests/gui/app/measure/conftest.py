"""Fixtures shared by the Run and artifact owners."""

from collections.abc import Generator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication

from tests.gui.app.measure._native_save_support import (
    NativeSaveSession,
    native_save_session,
)


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    with native_save_session(tmp_path, qapp) as session:
        yield session
