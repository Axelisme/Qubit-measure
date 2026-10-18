"""Complete previews read current values at each draft's fixed destination."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.writeback import (
    ModuleWriteback,
    ParameterWriteback,
    WritebackService,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.library_entry import LibraryEntryInvalid
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ComponentSchema, ResultEntry, RoleSpec
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry
from tests.resources.entry.view_support import registered_model


@pytest.fixture
def component_kinds():
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        yield
        assert registry_state() == installed
    finally:
        restore_registry(before)


@pytest.fixture
def session(tmp_path: Path, component_kinds):
    entry = ResultEntry.create(
        "entry", result_root=tmp_path / "Result", database_root=tmp_path / "Database"
    )
    entry.new_point("a")
    bus = BaseEventBus()
    context = ContextService(
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        bus,
        bind_assets=ArbWaveformDatabase.info_at,
    )
    context.install_entry(entry)
    control = ContextControlFacet(context=context)
    control.use_point(PointUseRequest("a"))
    control.add_component("Q1", "fake/drive/a", {})
    service = WritebackService(control, arb_names=lambda: ())
    return entry, control, service


def test_static_unit_facts_follow_captured_target_and_retarget_without_conversion(
    session,
):
    entry, control, service = session

    class Measured(ComponentSchema):
        frequency: float = Field(default=19.8, json_schema_extra={"unit": "MHz"})
        duration: float | None = Field(default=None, json_schema_extra={"unit": "us"})

    with registered_model("fake/unit-preview", Measured) as kind:
        control.add_component("U1", kind, {})
        target = control.read_context().target
        captured = control.read_context().bind_roles(
            {"driver": RoleSpec(kind)}, focus="U1", explicit={}
        )
        draft = service.create_draft(
            [
                ParameterWriteback(
                    "driver.frequency", "Frequency", proposed_value=20.09, stderr=0.18
                )
            ],
            parameters=captured,
            experiment_name="test/writeback",
        )
        try:
            initial = service.preview_values(draft, session_id="param-1")["param-1"]
            assert (initial.actual_path, initial.unit) == ("U1.frequency", "MHz")
            assert (initial.current, initial.proposed) == (19.8, 20.09)
            control.set_parameter(target, "U1.frequency", 21.5)
            entry.new_point("b")
            control.use_point(PointUseRequest("b"))
            control.add_component("Q2", "fake/drive/a", {})
            control.set_alias_rows((AliasRow("driver", "Q2"),))
            refreshed = service.preview_values(draft)["param-1"]
            assert (refreshed.actual_path, refreshed.unit) == ("U1.frequency", "MHz")
            assert (refreshed.current, refreshed.proposed) == (21.5, 20.09)
            draft.edit("param-1", target_name="driver.duration")
            unset = service.preview_values(draft)["param-1"]
            assert (unset.actual_path, unset.unit) == ("U1.duration", "us")
            assert (unset.current, unset.proposed) == (None, 20.09)
            draft.edit("param-1", target_name="general.ext.value")
            ext = service.preview_values(draft)["param-1"]
            assert (ext.actual_path, ext.unit) == ("general.ext.value", None)
            assert (ext.current, ext.proposed) == (None, 20.09)
            assert draft.items[0].stderr == 0.18
            assert control.read_context().target != target
        finally:
            draft.teardown()


def test_concrete_destination_with_same_named_alias_reads_fixed_target(session):
    entry, control, service = session

    class Coupler(ComponentSchema):
        module: dict[str, str]

    with registered_model("fake/preview-coupler", Coupler) as kind:
        control.add_component(
            "coupler", kind, {"module": {"cz_f2": "coupler.pulses.original"}}
        )
        control.set_alias_rows((AliasRow("coupler", "coupler"),))
        captured = control.read_context()
        draft = service.create_draft(
            [
                ParameterWriteback(
                    "coupler.module.cz_f2",
                    "Calibrated pulse slot",
                    proposed_value="coupler.pulses.calibrated",
                )
            ],
            parameters=captured,
            experiment_name="test/writeback",
        )
        try:
            initial = service.preview_values(draft, session_id="param-1")["param-1"]
            assert initial.actual_path == "coupler.module.cz_f2"
            assert initial.current == "coupler.pulses.original"
            assert initial.proposed == "coupler.pulses.calibrated"
            control.set_parameter(
                captured.target, "coupler.module.cz_f2", "coupler.pulses.updated"
            )
            entry.new_point("b")
            control.use_point(PointUseRequest("b"))
            control.add_component(
                "other_coupler", kind, {"module": {"cz_f2": "other.pulses.decoy"}}
            )
            control.set_alias_rows((AliasRow("coupler", "other_coupler"),))
            refreshed = service.preview_values(draft)["param-1"]
            assert refreshed.actual_path == "coupler.module.cz_f2"
            assert refreshed.current == "coupler.pulses.updated"
            assert refreshed.proposed == "coupler.pulses.calibrated"
            assert captured.value("coupler.module.cz_f2") == "coupler.pulses.original"
            assert control.read_context().target != captured.target
        finally:
            draft.teardown()


def test_values_follow_fixed_target_retarget_edits_and_actual_current_values(session):
    entry, control, service = session
    target = control.read_context().target
    control.set_parameter(target, "general.ext.value", [1, 2])
    capture = control.read_context()
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "d", proposed_value=[3, 4])],
        parameters=capture,
        experiment_name="test/writeback",
    )
    first = service.preview_values(draft)["param-1"]
    assert first.actual_path == "general.ext.value"
    assert first.current == [1, 2]
    assert first.proposed == [3, 4]
    control.set_parameter(target, "general.ext.value", [1, 2, 5])
    assert first.current == [1, 2]
    assert isinstance(first.proposed, list)
    first.proposed.append(99)
    assert service.preview_values(draft)["param-1"].proposed == [3, 4]
    assert service.preview_values(draft)["param-1"].current == [1, 2, 5]
    draft.edit("param-1", target_name="general.ext.other")
    assert service.preview_values(draft)["param-1"].current is None
    control.set_parameter(target, "general.ext.other", 7)
    assert service.preview_values(draft)["param-1"].current == 7
    entry.new_point("b")
    control.use_point(PointUseRequest("b"))
    control.set_parameter(control.read_context().target, "general.ext.other", 8)
    retargeted = service.preview_values(draft)["param-1"]
    assert retargeted.actual_path == "general.ext.other"
    assert retargeted.current == 7
    assert service.get_all_applied(draft) == {"param-1": False}
    assert draft.items[0].selected is True


def test_module_preview_contains_complete_current_and_live_form_values(session):
    _entry, control, service = session
    name = "Q1.pulses.target"
    control.set_module(
        control.read_context().target,
        name,
        PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
        ),
    )
    capture = control.read_context()
    proposal = ModuleWriteback(name, "d", capture.library.module_spec(name))
    draft = service.create_draft(
        [proposal], parameters=capture, experiment_name="test/writeback"
    )
    preview = service.preview_values(draft)["ml-1"]
    assert preview.actual_path == name
    assert preview.unit is None
    assert preview.current == capture.library.entry_value(name, kind="module")
    assert preview.proposed == preview.current
    service.edit_draft(draft, "ml-1", edits={("freq",): 250.0})
    live = service.preview_values(draft)["ml-1"]
    assert isinstance(live.proposed, dict)
    assert live.proposed["freq"] == 250.0
    assert live.current == preview.current
    draft.edit("ml-1", target_name=name.replace("target", "new"))
    missing = service.preview_values(draft)["ml-1"]
    assert missing.actual_path == name.replace("target", "new")
    assert missing.unit is None
    assert missing.current is None
    assert missing.proposed == live.proposed
    draft.teardown()
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)


def test_item_preview_excludes_invalid_siblings_but_bulk_remains_strict(session):
    _entry, control, service = session
    proposals = []
    for suffix in ("a", "b"):
        name = f"Q1.pulses.{suffix}"
        control.set_module(
            control.read_context().target,
            name,
            PulseSpec(
                ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
            ),
        )
        proposals.append(
            ModuleWriteback(
                name, name, control.read_context().library.module_spec(name)
            )
        )
    draft = service.create_draft(
        proposals, parameters=control.read_context(), experiment_name="test/writeback"
    )
    try:
        first, second = draft.items
        second.selected = False
        form = service.get_item_form(draft, second.session_id)
        form.state.write({("freq",): ""}, form.env)
        selected = service.preview_values(draft, session_id=first.session_id)
        assert list(selected) == [first.session_id]
        fact = selected[first.session_id]
        assert fact.actual_path == first.target_name
        assert isinstance(fact.current, dict) and isinstance(fact.proposed, dict)
        assert fact.current["freq"] == fact.proposed["freq"] == 100.0
        with pytest.raises(LibraryEntryInvalid):
            service.preview_values(draft)
        with pytest.raises(LibraryEntryInvalid):
            service.preview_values(draft, session_id=second.session_id)
        with pytest.raises(KeyError):
            service.preview_values(draft, session_id="missing")
        form.state.write({("freq",): 100.0}, form.env)
        assert len(service.preview_values(draft)) == 2
        assert second.selected is False
    finally:
        draft.teardown()
    empty = service.create_draft([], parameters=None, experiment_name="test/writeback")
    try:
        assert service.preview_values(empty) == {}
        with pytest.raises(KeyError):
            service.preview_values(empty, session_id="missing")
    finally:
        empty.teardown()


def test_preview_without_context_fails_instead_of_claiming_missing_targets():
    service = WritebackService(MagicMock(), arb_names=lambda: ())
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "d", proposed_value=1)],
        parameters=None,
        experiment_name="test/writeback",
    )
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)
