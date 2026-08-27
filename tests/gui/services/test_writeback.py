"""Unit tests for WritebackService — persistent draft (ADR-0008).

Items are computed once into the opaque writeback draft; apply reads the draft
as-is (no recompute) and sends one ContextWritePort batch, bumping ``context``
so concurrency guards detect the change.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.adapter import (
    ContextReadiness,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
)
from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload
from zcu_tools.gui.app.main.services.guard import WritebackPermit
from zcu_tools.gui.app.main.services.ports import CfgEdit, CfgEditResult
from zcu_tools.gui.app.main.services.writeback import WritebackService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.expected_error import (
    ExpectedErrorCategory,
    FailedPreconditionError,
    InvalidInputError,
)
from zcu_tools.gui.session.events import (
    MdChangedPayload,
    MlChangedPayload,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_state_with_tab(tab_id: str = "t1") -> State:
    state = State(
        ExpContext(
            md=MetaDict(),
            ml=ModuleLibrary(),
            soc=None,
            soccfg=None,
            result_dir="",
            readiness=ContextReadiness.ACTIVE,
        )
    )
    state.add_tab(
        tab_id,
        Session(adapter_name="fake", adapter=MagicMock(), cfg_schema=MagicMock()),
    )
    return state


def _make_write_port(state: State, bus: EventBus):
    """A ContextWritePort stand-in that reproduces ContextService.apply_writes'
    observable effects (ADR-0006) without lowering the items' mock schemas: it
    sets md attrs, registers ml entries (the schema is the registered object —
    enough for "did it land" assertions), bumps "context" once, emits per kind,
    and dumps ml when persistent.
    """
    port = MagicMock()

    def _apply_writes(writes) -> None:
        ctx = state.exp_context
        for key, value in writes.md.items():
            setattr(ctx.md, key, value)
        for name, schema in writes.ml_modules.items():
            ctx.ml.register_module(**{name: schema})
        for name, schema in writes.ml_waveforms.items():
            ctx.ml.register_waveform(**{name: schema})
        touched_ml = bool(writes.ml_modules or writes.ml_waveforms)
        if touched_ml and getattr(ctx.ml, "has_persistence", False):
            ctx.ml.dump()
        if writes.md or touched_ml:
            state.version.bump("context")
        if writes.md:
            bus.emit(MdChangedPayload(md=ctx.md))
        if touched_ml:
            bus.emit(MlChangedPayload(ml=ctx.ml))

    port.apply_writes.side_effect = _apply_writes
    return port


def _svc(
    state: State, bus: EventBus | None = None, cfg_editor: MagicMock | None = None
) -> WritebackService:
    """Build a WritebackService with a MagicMock CfgEditorService + a write port
    that reproduces ContextService's observable effects."""
    bus = bus or EventBus()
    return WritebackService(
        state, cfg_editor or MagicMock(), _make_write_port(state, bus)
    )


# ---------------------------------------------------------------------------
# Opaque transactional draft API
# ---------------------------------------------------------------------------


def test_create_draft_keeps_two_owners_independent_and_hides_editor_id():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.open_seeded.side_effect = [
        ("editor-a", ()),
        ("editor-b", ()),
    ]
    svc = WritebackService(state, cfg_editor, MagicMock())

    draft_a = svc.create_draft(
        [ModuleWriteback(target_name="a", description="a", edit_schema=MagicMock())]
    )
    draft_b = svc.create_draft(
        [ModuleWriteback(target_name="b", description="b", edit_schema=MagicMock())]
    )

    assert draft_a is not draft_b
    assert [item.session_id for item in draft_a.preview()] == ["ml-1"]
    assert [item.session_id for item in draft_b.preview()] == ["ml-1"]
    assert not hasattr(draft_a.preview()[0], "editor_id")
    draft_a.edit("ml-1", target_name="a-tuned")
    assert draft_a.preview()[0].target_name == "a-tuned"
    assert draft_b.preview()[0].target_name == "b"
    assert cfg_editor.open_seeded.call_count == 2
    assert (
        cfg_editor.open_seeded.call_args_list[0].kwargs["owner_key"]
        != (cfg_editor.open_seeded.call_args_list[1].kwargs["owner_key"])
    )


def test_create_draft_cleans_all_opened_sessions_when_a_later_item_fails():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.open_seeded.side_effect = [
        ("editor-1", ()),
        RuntimeError("open failed"),
    ]
    svc = WritebackService(state, cfg_editor, MagicMock())

    proposals = [
        ModuleWriteback(target_name="a", description="a", edit_schema=MagicMock()),
        WaveformWriteback(target_name="b", description="b", edit_schema=MagicMock()),
    ]
    with pytest.raises(RuntimeError, match="open failed"):
        svc.create_draft(proposals)

    cfg_editor.teardown.assert_called_once_with("editor-1")
    assert not hasattr(proposals[0], "editor_id")


def test_draft_teardown_is_idempotent_even_when_cleanup_raises():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.open_seeded.return_value = ("editor-1", ())
    cfg_editor.teardown.side_effect = RuntimeError("close failed")
    svc = WritebackService(state, cfg_editor, MagicMock())
    draft = svc.create_draft(
        [ModuleWriteback(target_name="a", description="a", edit_schema=MagicMock())]
    )

    draft.teardown()
    draft.teardown()

    assert draft.is_active is False
    cfg_editor.teardown.assert_called_once_with("editor-1")


def test_draft_cfg_edits_use_private_editor_session():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.open_seeded.return_value = ("editor-1", ())
    cfg_editor.set_fields.return_value = CfgEditResult(valid=True)
    svc = WritebackService(state, cfg_editor, MagicMock())
    draft = svc.create_draft(
        [ModuleWriteback(target_name="a", description="a", edit_schema=MagicMock())]
    )

    result = draft.edit("ml-1", edits=[{"path": "freq", "value": 5000.0}])

    assert result == {"valid": True, "removed": [], "added": []}
    cfg_editor.set_fields.assert_called_once_with("editor-1", [CfgEdit("freq", 5000.0)])


def test_teardown_draft_detaches_state_reference():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.open_seeded.return_value = ("editor-1", ())
    svc = WritebackService(state, cfg_editor, MagicMock())
    draft = svc.create_draft(
        [ModuleWriteback(target_name="a", description="a", edit_schema=MagicMock())]
    )
    state.get_tab("t1").analysis.writeback_draft = draft

    draft.teardown()

    assert state.get_tab("t1").analysis.writeback_draft is None


def test_apply_draft_sends_one_context_batch_for_selected_items():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    write_port = MagicMock()
    svc = WritebackService(state, cfg_editor, write_port)
    draft = svc.create_draft(
        [
            MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0),
            MetaDictWriteback(target_name="skip", description="d", proposed_value=2.0),
        ]
    )
    draft.edit("md-2", selected=False)

    result = draft.apply()

    assert result == {
        "applied_ids": ["md-1"],
        "written": {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    write_port.apply_writes.assert_called_once()
    writes = write_port.apply_writes.call_args.args[0]
    assert writes.md == {"r_f": 1.0}
    assert writes.ml_modules == {}
    assert writes.ml_waveforms == {}
