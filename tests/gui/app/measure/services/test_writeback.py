"""Persistent writeback drafts keep editor ownership and fixed destinations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
from zcu_tools.gui.app.measure.services.ports import CfgEdit, CfgEditResult
from zcu_tools.gui.app.measure.services.writeback import (
    WritebackEdit,
    WritebackService,
    WritebackWritten,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlFacet
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.gui.session.services.context import ContextService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import ResultEntry
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

from tests.resources.entry.fakes import register_fakes, registry_state, restore_registry


def editor_service(editor: MagicMock) -> WritebackService:
    """Isolate draft/editor lifecycle from context persistence."""
    return WritebackService(editor, MagicMock(), entry_ledger=MagicMock())


def pulse_schema() -> CfgSchema:
    spec, value = module_cfg_to_value(
        PulseSpec(
            ch=0, nqz=1, freq=5000.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
        ).model_dump(mode="python")
    )
    return CfgSchema(spec, value)


@pytest.fixture
def session(tmp_path: Path):
    before = registry_state()
    try:
        register_fakes()
        installed = registry_state()
        entry = ResultEntry.create(
            "entry",
            result_root=tmp_path / "Result",
            database_root=tmp_path / "Database",
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
        yield entry, control, editor, service
        assert registry_state() == installed
    finally:
        restore_registry(before)


def test_create_draft_keeps_two_owners_independent_and_hides_editor_id():
    editor = MagicMock()
    editor.open_seeded.side_effect = [("editor-a", ()), ("editor-b", ())]
    service = editor_service(editor)
    draft_a = service.create_draft(
        [ModuleWriteback("Q1.pulses.a", "a", edit_schema=MagicMock())], parameters=None
    )
    draft_b = service.create_draft(
        [ModuleWriteback("Q1.pulses.b", "b", edit_schema=MagicMock())], parameters=None
    )
    assert draft_a is not draft_b
    assert [item.session_id for item in draft_a.preview()] == ["ml-1"]
    assert [item.session_id for item in draft_b.preview()] == ["ml-1"]
    assert not hasattr(draft_a.preview()[0], "editor_id")
    draft_a.edit("ml-1", target_name="Q1.pulses.tuned")
    assert draft_a.preview()[0].target_name == "Q1.pulses.tuned"
    assert draft_b.preview()[0].target_name == "Q1.pulses.b"
    assert editor.open_seeded.call_count == 2
    assert (
        editor.open_seeded.call_args_list[0].kwargs["owner_key"]
        != editor.open_seeded.call_args_list[1].kwargs["owner_key"]
    )


def test_create_draft_cleans_all_opened_sessions_when_a_later_item_fails():
    editor = MagicMock()
    editor.open_seeded.side_effect = [("editor-1", ()), RuntimeError("open failed")]
    proposals = [
        ModuleWriteback("Q1.pulses.a", "a", edit_schema=MagicMock()),
        WaveformWriteback("Q1.waveforms.b", "b", edit_schema=MagicMock()),
    ]
    with pytest.raises(RuntimeError, match="open failed"):
        editor_service(editor).create_draft(proposals, parameters=None)
    editor.teardown.assert_called_once_with("editor-1")
    assert not hasattr(proposals[0], "editor_id")


def test_create_draft_rejects_proposal_with_dynamic_editor_identity():
    @dataclass
    class EditorOwnedProposal(ModuleWriteback):
        editor_id: str = "legacy-editor"

    proposal = EditorOwnedProposal("Q1.pulses.a", "a", edit_schema=MagicMock())
    with pytest.raises(InvalidInputError, match="must not expose editor_id"):
        editor_service(MagicMock()).create_draft([proposal], parameters=None)


def test_draft_teardown_is_idempotent_even_when_cleanup_raises():
    editor = MagicMock()
    editor.open_seeded.return_value = ("editor-1", ())
    editor.teardown.side_effect = RuntimeError("close failed")
    draft = editor_service(editor).create_draft(
        [ModuleWriteback("Q1.pulses.a", "a", edit_schema=MagicMock())], parameters=None
    )
    draft.teardown()
    draft.teardown()
    assert draft.is_active is False
    editor.teardown.assert_called_once_with("editor-1")


def test_draft_cfg_edits_use_owner_editor_session():
    editor = MagicMock()
    editor.open_seeded.return_value = ("editor-1", ())
    editor.set_fields.return_value = CfgEditResult(valid=True)
    draft = editor_service(editor).create_draft(
        [ModuleWriteback("Q1.pulses.a", "a", edit_schema=MagicMock())], parameters=None
    )
    result = draft.edit("ml-1", edits=[{"path": "freq", "value": 5000.0}])
    assert result == {"valid": True, "removed": [], "added": []}
    editor.set_fields.assert_called_once_with(
        "editor-1", [CfgEdit("freq", 5000.0)], agent_edit=True
    )


def test_agent_writeback_edits_delegate_to_shared_aggregate_grammar():
    editor = MagicMock()
    editor.open_seeded.return_value = ("editor-1", ())
    editor.set_fields.return_value = CfgEditResult(valid=True)
    service = editor_service(editor)
    draft = service.create_draft(
        [ModuleWriteback("Q1.pulses.a", "d", edit_schema=MagicMock())], parameters=None
    )
    sweep = {"start": 1.0, "stop": 2.0, "expts": 3}
    result = service.edit_draft(draft, "ml-1", edits=[{"path": "freq", "value": sweep}])
    assert result["valid"] is True
    editor.set_fields.assert_called_once_with(
        "editor-1", [CfgEdit("freq", sweep)], agent_edit=True
    )


def test_non_scalar_current_and_proposed_summaries_use_the_same_bounded_shape(session):
    _entry, control, _editor, service = session
    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    control.set_parameter(
        control.read_context().target, "general.ext.confusion_matrix", matrix
    )
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.confusion_matrix", "d", proposed_value=matrix)],
        parameters=control.read_context(),
    )
    assert service.get_summaries(draft, "md-1") == ("3 × 3 matrix", "3 × 3 matrix")


def test_applied_state_follows_successful_writes_and_content_edits(session):
    _entry, control, _editor, service = session
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.r_f", "d", proposed_value=1.0),
            MetaDictWriteback("general.ext.skip", "d", proposed_value=2.0),
        ],
        parameters=control.read_context(),
    )
    assert service.get_all_applied(draft) == {"md-1": False, "md-2": False}
    draft.edit("md-2", selected=False)
    draft.apply()
    assert service.get_all_applied(draft) == {"md-1": True, "md-2": False}
    draft.edit("md-1", selected=False)
    assert service.get_all_applied(draft)["md-1"] is True
    draft.edit("md-1", target_name="general.ext.r_f", proposed_value=1.0)
    assert service.get_all_applied(draft)["md-1"] is True
    draft.edit("md-1", proposed_value=3.0)
    assert service.get_all_applied(draft)["md-1"] is False


def test_direct_cfg_draft_change_marks_applied_item_unapplied(session):
    _entry, control, _editor, service = session
    draft = service.create_draft(
        [ModuleWriteback("Q1.pulses.readout", "d", edit_schema=pulse_schema())],
        parameters=control.read_context(),
    )
    draft.apply()
    assert service.get_all_applied(draft) == {"ml-1": True}
    service.get_item_draft(draft, "ml-1").resolve_target("gain").set_value(0.4)
    assert service.get_all_applied(draft) == {"ml-1": False}


def test_duplicate_selected_destination_is_rejected_before_write(session):
    entry, control, _editor, service = session
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.same", "first", proposed_value=1.0),
            MetaDictWriteback("general.ext.same", "second", proposed_value=2.0),
        ],
        parameters=control.read_context(),
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError, match="duplicate writeback destination"):
        draft.apply()
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.same").present
    assert service.get_all_applied(draft) == {"md-1": False, "md-2": False}


def test_failed_and_empty_apply_do_not_mark_items_applied(session, monkeypatch):
    _entry, control, _editor, service = session
    monkeypatch.setattr(
        control, "set_parameter", MagicMock(side_effect=RuntimeError("write failed"))
    )
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.r_f", "d", proposed_value=1.0)],
        parameters=control.read_context(),
    )
    with pytest.raises(RuntimeError, match="write failed"):
        draft.apply()
    assert service.get_all_applied(draft) == {"md-1": False}
    draft.edit("md-1", selected=False)
    assert draft.apply()["applied_ids"] == []
    assert service.get_all_applied(draft) == {"md-1": False}


def test_explicit_apply_ignores_selection_without_changing_checkboxes(session):
    _entry, control, _editor, service = session
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.a", "d", proposed_value=1.0),
            MetaDictWriteback("general.ext.b", "d", proposed_value=2.0),
        ],
        parameters=control.read_context(),
    )
    draft.edit("md-1", selected=False)
    selection = [item.selected for item in draft.items]
    assert draft.apply(item_ids=("md-1",))["applied_ids"] == ["md-1"]
    current = control.read_context()
    assert current.value("general.ext.a") == 1.0
    assert not current.lookup("general.ext.b").present
    assert [item.selected for item in draft.items] == selection
    assert service.get_all_applied(draft) == {"md-1": True, "md-2": False}


@pytest.mark.parametrize("item_ids", [("md-1", "missing"), ("md-1", "md-1")])
def test_explicit_apply_rejects_invalid_ids_before_any_write(session, item_ids):
    entry, control, _editor, service = session
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.a", "d", proposed_value=1.0)],
        parameters=control.read_context(),
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError):
        draft.apply(item_ids=item_ids)
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"md-1": False}


def test_explicit_empty_apply_does_not_fall_back_to_selected_items(session):
    entry, control, _editor, service = session
    draft = service.create_draft(
        [MetaDictWriteback("general.ext.a", "d", proposed_value=1.0)],
        parameters=control.read_context(),
    )
    count = len(entry.ledger.events())
    assert draft.apply(item_ids=())["applied_ids"] == []
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"md-1": False}


def test_ordered_write_returns_actual_values_and_leaves_selection_alone(session):
    _entry, control, _editor, service = session
    control.set_parameter(control.read_context().target, "general.ext.dest", [1])
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.a", "d", proposed_value=2),
            MetaDictWriteback("general.ext.b", "d", proposed_value=3),
        ],
        parameters=control.read_context(),
    )
    draft.edit("md-1", selected=False)
    result = service.write_draft(
        draft,
        (WritebackEdit("md-1", target_name="general.ext.dest", proposed_value=None),),
    )
    assert result == [WritebackWritten("md-1", "md", "general.ext.dest", [1], None)]
    current = control.read_context()
    assert current.value("general.ext.dest") is None
    assert not current.lookup("general.ext.b").present
    assert [item.selected for item in draft.items] == [False, True]
    assert service.get_all_applied(draft) == {"md-1": True, "md-2": False}


def test_ordered_write_stops_on_unknown_item_and_preserves_draft_prefix(session):
    entry, control, _editor, service = session
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.a", "d", proposed_value=1),
            MetaDictWriteback("general.ext.b", "d", proposed_value=2),
        ],
        parameters=control.read_context(),
    )
    count = len(entry.ledger.events())
    with pytest.raises(InvalidInputError):
        service.write_draft(
            draft,
            (
                WritebackEdit("md-1", proposed_value=11),
                WritebackEdit("missing"),
                WritebackEdit("md-2", proposed_value=22),
            ),
        )
    values = []
    for item in draft.items:
        assert isinstance(item, MetaDictWriteback)
        values.append(item.proposed_value)
    assert values == [11, 2]
    assert len(entry.ledger.events()) == count
    assert not control.read_context().lookup("general.ext.a").present
    assert service.get_all_applied(draft) == {"md-1": False, "md-2": False}


def test_ordered_write_rejects_invalid_cfg_before_context_apply(session):
    entry, control, _editor, _service = session
    editor = MagicMock()
    editor.open_seeded.return_value = ("editor", ())
    editor.set_fields.return_value = CfgEditResult(valid=False)
    service = WritebackService(editor, control, entry_ledger=lambda _id: entry.ledger)
    draft = service.create_draft(
        [ModuleWriteback("Q1.pulses.a", "d", edit_schema=MagicMock())],
        parameters=control.read_context(),
    )
    count = len(entry.ledger.events())
    with pytest.raises(FailedPreconditionError, match="ml-1"):
        service.write_draft(
            draft,
            (WritebackEdit("ml-1", edits=[{"path": "gain", "value": "invalid"}]),),
        )
    assert len(entry.ledger.events()) == count
    assert "Q1.pulses.a" not in control.read_context().library.module_keys()


def test_apply_draft_records_accepted_before_selected_setter(session, monkeypatch):
    entry, control, _editor, service = session
    draft = service.create_draft(
        [
            MetaDictWriteback("general.ext.r_f", "d", proposed_value=1.0),
            MetaDictWriteback("general.ext.skip", "d", proposed_value=2.0),
        ],
        parameters=control.read_context(),
    )
    draft.edit("md-2", selected=False)
    original = control.set_parameter
    setter = MagicMock(wraps=original)

    def after_accepted(*args, **kwargs):
        assert entry.ledger.events()[-1].kind == "accepted"
        return setter(*args, **kwargs)

    monkeypatch.setattr(control, "set_parameter", after_accepted)
    result = draft.apply()
    assert result == {
        "applied_ids": ["md-1"],
        "written": {"md": ["general.ext.r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    assert setter.call_count == 1
    current = control.read_context()
    assert current.value("general.ext.r_f") == 1.0
    assert not current.lookup("general.ext.skip").present
