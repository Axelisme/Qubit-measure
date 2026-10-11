"""Function-scoped session lifecycle shared by writeback contract tests."""

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, component_registry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import registry_state, restore_registry
from tests.zcu_lab._writeback_support import WritebackSession
from zcu_lab.components import QubitSchema, ResonatorSchema


@pytest.fixture
def session(tmp_path: Path) -> Generator[WritebackSession]:
    """Build isolated stores under tmp_path; guard and restore registry on teardown.

    Each case gets fresh session owners. Setup errors propagate, and teardown
    asserts registration was preserved before restoring the incoming registry.
    """
    before = registry_state()
    try:
        for kind, model in [
            ("qubit/transmon", QubitSchema),
            ("resonator", ResonatorSchema),
        ]:
            if kind not in component_registry.kinds():
                component_registry.register(kind, model)
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "results",
            database_root=tmp_path / "Database",
        )
        entry.new_point("run-point")
        entry.new_point("later-point")
        bus = BaseEventBus()
        context = ContextService(
            SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
            bus,
            bind_assets=ArbWaveformDatabase.info_at,
        )
        context.install_entry(entry)
        control = ContextControlFacet(context=context)
        for point in ("run-point", "later-point"):
            control.use_point(PointUseRequest(point))
            for index in (1, 2):
                control.add_component(f"R{index}", "resonator", {})
                control.add_component(
                    f"Q{index}",
                    "qubit/transmon",
                    {"resonator": f"R{index}", "ext": {"readout_cal": {}}},
                )
        control.use_point(PointUseRequest("run-point"))
        control.set_alias_rows(
            (AliasRow("qubit", "Q1"), AliasRow("resonator", "$qubit.resonator"))
        )
        editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
        service = WritebackService(editor, control)
        yield WritebackSession(control, context, service, entry)
        assert registry_state() == installed
    finally:
        restore_registry(before)
