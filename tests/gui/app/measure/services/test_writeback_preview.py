"""Complete previews read current values at each draft's fixed destination."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import Field
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    ModuleWriteback,
    ParameterWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.cfg.binding.fields import ScalarField
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
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
    editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
    service = WritebackService(editor, control)
    return entry, control, editor, service


def test_static_unit_facts_follow_captured_target_and_retarget_without_conversion(
    session,
):
    entry, control, _editor, service = session

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


def test_values_follow_fixed_target_retarget_edits_and_actual_current_values(session):
    entry, control, _editor, service = session
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


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_cfg_preview_contains_complete_current_and_live_draft_values(
    session, kind, monkeypatch
):
    _entry, control, editor, service = session
    target = control.read_context().target
    if kind == "module":
        cfg = PulseSpec(
            ch=0, nqz=1, freq=100.0, gain=0.5, waveform=ConstWaveformCfg(length=0.1)
        )
        name = "Q1.pulses.target"
        control.set_module(target, name, cfg)
        capture = control.read_context()
        spec, value = module_cfg_to_value(capture.library.get_module(name))
        proposal = ModuleWriteback(name, "d", edit_schema=CfgSchema(spec, value))
        item_id, path, edited = "ml-1", "freq", 250.0
    else:
        name = "Q1.waveforms.target"
        control.set_waveform(target, name, ConstWaveformCfg(length=0.1))
        capture = control.read_context()
        spec, value = waveform_cfg_to_value(capture.library.get_waveform(name))
        proposal = WaveformWriteback(name, "d", edit_schema=CfgSchema(spec, value))
        item_id, path, edited = "wf-1", "length", 0.2
    draft = service.create_draft(
        [proposal], parameters=capture, experiment_name="test/writeback"
    )
    preview = service.preview_values(draft)[item_id]
    assert preview.actual_path == name
    assert preview.unit is None
    assert preview.current == capture.library.entry_value(name, kind=kind)
    if kind == "module":
        assert PulseSpec.model_validate(
            preview.proposed
        ) == capture.library.module_spec(name)
    else:
        assert preview.proposed == preview.current
    service.edit_draft(draft, item_id, edits=[{"path": path, "value": edited}])
    live = service.preview_values(draft)[item_id]
    assert isinstance(live.proposed, dict)
    assert live.proposed[path] == edited
    assert live.current == preview.current
    draft.edit(item_id, target_name=name.replace("target", "new"))
    missing = service.preview_values(draft)[item_id]
    assert missing.actual_path == name.replace("target", "new")
    assert missing.unit is None
    assert missing.current is None
    assert missing.proposed == live.proposed
    error = FailedPreconditionError("draft unavailable")

    def fail_read(_editor_id):
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(editor, "get_draft", fail_read)
        with pytest.raises(FailedPreconditionError) as caught:
            service.preview_values(draft)
        assert caught.value is error
    draft.teardown()
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)


@pytest.mark.parametrize("kind", ["module", "waveform"])
def test_item_preview_excludes_invalid_siblings_but_bulk_remains_strict(session, kind):
    _entry, control, _editor, service = session
    proposals = []
    for suffix in ("a", "b"):
        if kind == "module":
            module_cfg = PulseSpec(
                ch=0,
                nqz=1,
                freq=100.0,
                gain=0.5,
                waveform=ConstWaveformCfg(length=0.1),
            )
            name = f"Q1.pulses.{suffix}"
            control.set_module(control.read_context().target, name, module_cfg)
            spec, value = module_cfg_to_value(
                control.read_context().library.get_module(name)
            )
            proposals.append(
                ModuleWriteback(name, name, edit_schema=CfgSchema(spec, value))
            )
            field_name, expected = "freq", 100.0
        else:
            waveform_cfg = ConstWaveformCfg(length=0.1)
            name = f"Q1.waveforms.{suffix}"
            control.set_waveform(control.read_context().target, name, waveform_cfg)
            spec, value = waveform_cfg_to_value(waveform_cfg)
            proposals.append(
                WaveformWriteback(name, name, edit_schema=CfgSchema(spec, value))
            )
            field_name, expected = "length", 0.1
    draft = service.create_draft(
        proposals, parameters=control.read_context(), experiment_name="test/writeback"
    )
    try:
        first, second = draft.items
        second.selected = False
        field = service.get_item_draft(draft, second.session_id).root.fields[field_name]
        assert isinstance(field, ScalarField)
        field.set_text("")
        selected = service.preview_values(draft, session_id=first.session_id)
        assert list(selected) == [first.session_id]
        fact = selected[first.session_id]
        assert fact.actual_path == first.target_name
        assert fact.current[field_name] == fact.proposed[field_name] == expected
        with pytest.raises(RuntimeError):
            service.preview_values(draft)
        with pytest.raises(RuntimeError):
            service.preview_values(draft, session_id=second.session_id)
        with pytest.raises(KeyError):
            service.preview_values(draft, session_id="missing")
        field.set_text(str(expected))
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
    service = WritebackService(MagicMock(), MagicMock())
    draft = service.create_draft(
        [ParameterWriteback("general.ext.value", "d", proposed_value=1)],
        parameters=None,
        experiment_name="test/writeback",
    )
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)
