"""Function-scoped Controller progress recording and synthetic Window sessions."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication
from zcu_tools.gui.session.services.progress import (
    BoundProgressFactory,
    ProgressService,
)
from zcu_tools.resources.entry import component_registry

from tests.resources.entry.fakes import Driver, registry_state, restore_registry

from ._window_support import QUBIT_KIND, WindowSession


@pytest.fixture
def progress_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[int]]:
    """Observe the run-lifetime factory/discard contract using the real service."""
    created: list[int] = []
    discarded: list[int] = []
    make_factory = ProgressService.make_factory
    discard_operation = ProgressService.discard_operation

    def make(service: ProgressService, token: int, owner: str) -> BoundProgressFactory:
        created.append(token)
        return make_factory(service, token, owner)

    def discard(service: ProgressService, token: int) -> None:
        discarded.append(token)
        discard_operation(service, token)

    monkeypatch.setattr(ProgressService, "make_factory", make)
    monkeypatch.setattr(ProgressService, "discard_operation", discard)
    return created, discarded


@pytest.fixture
def window_session(qapp: QApplication, tmp_path: Path) -> Iterator[WindowSession]:
    """Build a Window over synthetic workflows; restore the component registry."""
    before = registry_state()
    component_registry.register(QUBIT_KIND, Driver)
    installed = registry_state()
    session = None
    try:
        session = WindowSession(tmp_path)
        yield session
    finally:
        try:
            if session is not None:
                session.close(qapp)
            assert registry_state() == installed, "window test polluted registries"
        finally:
            restore_registry(before)
