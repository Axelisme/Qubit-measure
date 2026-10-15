"""Template factories and library writes share a fixed public alias capture."""

from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path

import pytest
from zcu_tools.cfg_model import PulseReadoutSpec, PulseSpec
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.specs import make_pulse_spec
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog, TemplateEntry
from zcu_tools.gui.cfg import (
    EvalValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import EntryRequest, PointUseRequest
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry, component_registry

from tests.gui.app.measure._components import register_all
from tests.gui.app.measure._templates import register_all_templates
from tests.resources.entry.fakes import registry_state, restore_registry


@dataclass
class TemplateSession:
    controller: Controller
    entry: ResultEntry
    catalog: TemplateCatalog
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
        catalog = TemplateCatalog()
        register_all_templates(catalog)
        state = State(SessionEnv(parameters=None, soc=None, soccfg=None))
        ctrl = Controller(
            state,
            Registry(),
            None,
            BaseEventBus(),
            template_catalog=catalog,
            project_root=str(tmp_path),
        )
        ctrl.setup_control.open_entry(EntryRequest("entry"))
        ctrl.context_control.use_point(PointUseRequest("first"))
        ctrl.context_control.set_alias_rows((AliasRow("resonator", "R1"),))
        try:
            yield TemplateSession(ctrl, entry, catalog, state)
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
    assert "$resonator.wiring.ch" in str(error.value)
    assert "missing parameter" in str(error.value)
    assert ctrl.read_context().library.document() == before.library.document()
    assert session.state.version.get("context") == version
    assert session.entry.ledger.events() == ledger


def test_factory_and_apply_keep_same_capture_across_source_and_selection_changes(
    session: TemplateSession,
) -> None:
    ctrl = session.controller
    origin = ctrl.read_context().target
    seen: list[object] = []

    def make_value(environment: SessionEnv) -> ReferenceValue:
        parameters = environment.parameters
        assert parameters is not None
        assert parameters.aliases == (AliasRow("resonator", "R1"),)
        seen.append(parameters.value("$resonator.freq"))
        ctrl.context_control.set_parameter(origin, "R1.freq", 7000.0)
        ctrl.context_control.set_alias_rows((AliasRow("resonator", "R2"),))
        ctrl.context_control.use_point(PointUseRequest("second"))
        value = make_default_value(make_pulse_spec())
        value.fields["freq"] = EvalValue("$resonator.freq")
        return ReferenceValue(make_custom_reference_key("pulse"), value)

    session.catalog.register(
        TemplateEntry(
            "fixed_frequency", "Fixed frequency", "module", make_pulse_spec, make_value
        )
    )
    ctrl.create_from_template("module", "fixed_frequency", "R1.pulses.fixed")
    assert seen == [6123.0]
    assert ctrl.read_context().target != origin
    assert ctrl.read_context().library.module_keys() == ()
    written = ctrl.context_control.read_target(origin).library.module_spec(
        "R1.pulses.fixed"
    )
    assert isinstance(written, PulseSpec)
    assert written.freq == 6123.0
    assert ctrl.context_control.read_target(origin).value("R1.freq") == 7000.0
