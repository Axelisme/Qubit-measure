"""Template factories and library writes share a fixed public alias capture."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import PulseReadoutSpec, PulseSpec
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, component_registry

from tests.gui.app.measure._components import register_all
from tests.gui.app.measure._templates import TEMPLATES
from tests.resources.entry.fakes import registry_state, restore_registry


@dataclass
class TemplateSession:
    controller: Controller
    entry: ResultEntry
    state: State


@pytest.fixture
def session(qapp, tmp_path: Path) -> Generator[TemplateSession]:
    before = registry_state()
    try:
        register_all(component_registry)
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        entry.setup.add_component(
            "R1", kind="resonator", freq=6123.0, wiring={"ch": 1, "ro_ch": 2}
        )
        entry.setup.add_component(
            "R2", kind="resonator", freq=7200.0, wiring={"ch": 3, "ro_ch": 4}
        )
        entry.setup.add_component("Empty", kind="resonator")
        entry.new_point("first")
        entry.new_point("second")
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        ctrl = Controller(
            state,
            Registry(templates=TEMPLATES),
            None,
            BaseEventBus(),
            project_root=str(tmp_path),
        )
        ctrl.setup_control.open_entry(EntryRequest("entry"))
        ctrl.context_control.use_point(PointUseRequest("first"))
        ctrl.context_control.set_alias_rows((AliasRow("resonator", "R1"),))
        try:
            yield TemplateSession(ctrl, entry, state)
            assert registry_state() == installed, (
                "template capture tests polluted registries"
            )
        finally:
            ctrl.close()
            qapp.processEvents()
    finally:
        restore_registry(before)


def test_bound_resonator_creates_original_probe_from_public_template(
    session: TemplateSession,
) -> None:
    ctrl = session.controller
    original = ctrl.read_context()
    ledger = session.entry.ledger.events()
    try:
        ctrl.create_from_template("module", "res_probe", "R1.pulses.probe")
    except InvalidInputError as error:
        raise AssertionError(
            "A bound calibrated resonator must create the original template"
        ) from error
    current = ctrl.read_context()
    probe = current.library.module_spec("R1.pulses.probe")
    assert isinstance(probe, PulseSpec)
    assert probe.freq == 6123.0
    assert probe.ch == 1
    assert original.library.module_keys() == ()
    source = current.library.meta("R1.pulses.probe.freq")
    assert source is not None and source.source == "manual"
    assert session.entry.ledger.events() == ledger


def test_bound_resonator_creates_original_readout_with_adc_channel(
    session: TemplateSession,
) -> None:
    ctrl = session.controller
    ctrl.context_control.set_parameter(
        ctrl.read_context().target, "R1.ext.time_of_fly", 0.5
    )
    try:
        ctrl.create_from_template("module", "readout", "R1.readout.rf")
    except InvalidInputError as error:
        raise AssertionError(
            "A bound resonator must create the original readout template"
        ) from error
    readout = ctrl.read_context().library.module_spec("R1.readout.rf")
    assert isinstance(readout, PulseReadoutSpec)
    assert isinstance(readout.pulse_cfg, PulseSpec)
    assert readout.pulse_cfg.freq == 6123.0
    assert readout.pulse_cfg.ch == 1
    assert readout.ro_cfg.ro_ch == 2


def test_bound_empty_resonator_reports_first_missing_leaf_without_write(
    session: TemplateSession,
) -> None:
    ctrl = session.controller
    ctrl.context_control.set_alias_rows((AliasRow("resonator", "Empty"),))
    before = ctrl.read_context()
    version = session.state.version.get("context")
    ledger = session.entry.ledger.events()
    with pytest.raises(InvalidInputError) as error:
        ctrl.create_from_template("module", "res_probe", "Empty.readout.probe")
    assert "ch" in str(error.value)
    assert "Field required" in str(error.value)
    assert ctrl.read_context().library.document() == before.library.document()
    assert session.state.version.get("context") == version
    assert session.entry.ledger.events() == ledger
