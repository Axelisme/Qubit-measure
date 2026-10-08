"""Complete previews read current values at each draft's fixed destination."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import (
    module_cfg_to_value,
    waveform_cfg_to_value,
)
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.writeback import WritebackService
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


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
    control.use_point(PointUseRequest(entry.entry_id, "a"))
    control.add_component("Q1", "fake/drive/a", {})
    editor = CfgEditorService(MagicMock(), control, MagicMock(), MagicMock(), bus)
    service = WritebackService(editor, control, entry_ledger=context.entry_ledger)
    return entry, control, editor, service


def test_values_follow_fixed_target_retarget_edits_and_actual_current_values(session):
    entry, control, _editor, service = session
    target = control.read_context().target
    control.set_parameter(target, "general.ext.value", [1, 2])
    capture = control.read_context()
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.value", "d", proposed_value=[3, 4])],
        parameters=capture,
    )
    first = service.preview_values(draft)["md-1"]
    assert first.current == [1, 2]
    assert first.proposed == [3, 4]
    control.set_parameter(target, "general.ext.value", [1, 2, 5])
    assert first.current == [1, 2]
    assert isinstance(first.proposed, list)
    first.proposed.append(99)
    assert service.preview_values(draft)["md-1"].proposed == [3, 4]
    assert service.preview_values(draft)["md-1"].current == [1, 2, 5]
    draft.edit("md-1", target_name="general.ext.other")
    assert service.preview_values(draft)["md-1"].current is None
    control.set_parameter(target, "general.ext.other", 7)
    assert service.preview_values(draft)["md-1"].current == 7
    entry.new_point("b")
    control.use_point(PointUseRequest(entry.entry_id, "b"))
    control.set_parameter(control.read_context().target, "general.ext.other", 8)
    assert service.preview_values(draft)["md-1"].current == 7
    assert service.get_all_applied(draft) == {"md-1": False}
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
    draft = service.create_draft([proposal], parameters=capture)
    preview = service.preview_values(draft)[item_id]
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


def test_preview_without_context_fails_instead_of_claiming_missing_targets():
    service = WritebackService(MagicMock(), MagicMock(), entry_ledger=MagicMock())
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.value", "d", proposed_value=1)],
        parameters=None,
    )
    with pytest.raises(FailedPreconditionError):
        service.preview_values(draft)
