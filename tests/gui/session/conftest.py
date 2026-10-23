"""Synthetic selected-point owner for external file synchronization tests."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


@pytest.fixture
def selected(
    tmp_path: Path,
) -> Generator[tuple[ContextService, BaseEventBus, ResultEntry]]:
    before = registry_state()
    register_fakes()
    try:
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "database",
        )
        entry.setup.add_component("Q1", kind="fake/drive/a", rate=4300.0, sense="R1")
        entry.new_point("first")
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        context.use_point(PointUseRequest("first"))
        yield context, bus, entry
    finally:
        restore_registry(before)
