"""Fixtures shared by the Run and artifact owners."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from qtpy.QtWidgets import QApplication

if TYPE_CHECKING:
    from tests.gui.app.measure._native_save_support import NativeSaveSession


@pytest.fixture
def native_session(tmp_path: Path, qapp: QApplication) -> Generator[NativeSaveSession]:
    from tests.gui.app.measure._native_save_support import native_save_session

    with native_save_session(tmp_path, qapp) as session:
        yield session
