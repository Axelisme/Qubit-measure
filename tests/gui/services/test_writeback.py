"""Unit tests for WritebackService — persistent draft (ADR-0008).

Items are computed once into Session.writeback_items; apply reads that draft
as-is (no recompute) and writes md/ml directly, bumping ``context`` so
concurrency guards detect the change.
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
from zcu_tools.gui.app.main.services.guard import WritebackPermit
from zcu_tools.gui.app.main.services.writeback import WritebackService
from zcu_tools.gui.app.main.state import ExpContext, Session, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
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
        state, bus, cfg_editor or MagicMock(), _make_write_port(state, bus)
    )


def _put_items(state: State, *items, tab_id: str = "t1") -> None:
    """Place persistent items on the tab (as compute would)."""
    state.get_tab(tab_id).writeback_items = list(items)


# ---------------------------------------------------------------------------
# apply (reads persistent draft, writes md/ml, bumps context)
# ---------------------------------------------------------------------------


def test_apply_md_writeback_bumps_context_version():
    state = _make_state_with_tab()
    svc = _svc(state)
    before = state.version.get("context")

    item = MetaDictWriteback(
        target_name="r_f", description="update r_f", proposed_value=6100.0
    )
    item.session_id = "md-1"
    _put_items(state, item)

    applied = svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))

    assert applied["applied_ids"] == ["md-1"]
    assert applied["written"] == {"md": ["r_f"], "ml_modules": [], "ml_waveforms": []}
    assert state.exp_context.md.r_f == 6100.0
    assert state.version.get("context") == before + 1


def test_apply_nothing_selected_does_not_bump_context():
    state = _make_state_with_tab()
    svc = _svc(state)
    before = state.version.get("context")

    item = MetaDictWriteback(
        target_name="r_f", description="update r_f", proposed_value=6100.0
    )
    item.session_id = "md-1"
    item.selected = False
    _put_items(state, item)

    applied = svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))

    assert applied["applied_ids"] == []
    assert applied["written"] == {"md": [], "ml_modules": [], "ml_waveforms": []}
    assert state.version.get("context") == before


def test_apply_uses_target_name_as_destination():
    """target_name (not session_id) is the md attr written."""
    state = _make_state_with_tab()
    svc = _svc(state)

    item = MetaDictWriteback(
        target_name="r_f_alt", description="d", proposed_value=42.0
    )
    item.session_id = "md-1"
    _put_items(state, item)

    svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))
    assert state.exp_context.md.r_f_alt == 42.0


def test_apply_non_scalar_md_value_lands_verbatim():
    """A nested-list md proposed_value (e.g. the singleshot confusion matrix) is
    applied verbatim — md holds nested lists, no scalar coercion on apply."""
    state = _make_state_with_tab()
    svc = _svc(state)

    matrix = [[0.95, 0.03, 0.02], [0.03, 0.95, 0.02], [0.0, 0.0, 1.0]]
    item = MetaDictWriteback(
        target_name="confusion_matrix", description="d", proposed_value=matrix
    )
    item.session_id = "md-1"
    _put_items(state, item)

    applied = svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))

    assert applied["applied_ids"] == ["md-1"]
    assert state.exp_context.md.confusion_matrix == matrix


# ---------------------------------------------------------------------------
# set_item_field — the editing surface (selected / target_name / proposed_value
# metadict facet / edits module-waveform facet)
# ---------------------------------------------------------------------------


def test_set_item_field_edits_route_through_item_editor():
    """The ``edits`` facet resolves the item's editor_id internally and applies
    each edit via CfgEditorPort.set_field, returning the aggregated result."""
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    cfg_editor.set_field.return_value = {
        "valid": True,
        "removed": ["qub.old"],
        "added": ["qub.new"],
    }
    svc = _svc(state, cfg_editor=cfg_editor)

    item = ModuleWriteback(target_name="qub", description="d", edit_schema=MagicMock())
    item.session_id = "ml-1"
    item.editor_id = "editor-7"
    _put_items(state, item)

    agg = svc.set_item_field(
        "t1", "ml-1", edits=[{"path": "qub.freq", "value": 5000.0}]
    )

    cfg_editor.set_field.assert_called_once_with("editor-7", "qub.freq", 5000.0)
    assert agg == {"valid": True, "removed": ["qub.old"], "added": ["qub.new"]}


def test_set_item_field_edits_on_metadict_item_raises():
    state = _make_state_with_tab()
    svc = _svc(state)
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0)
    item.session_id = "md-1"
    _put_items(state, item)
    with pytest.raises(RuntimeError, match="not a module/waveform item"):
        svc.set_item_field("t1", "md-1", edits=[{"path": "p", "value": 1}])


def test_set_item_field_metadict_value_returns_empty_aggregate():
    state = _make_state_with_tab()
    svc = _svc(state)
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0)
    item.session_id = "md-1"
    _put_items(state, item)
    agg = svc.set_item_field("t1", "md-1", proposed_value=6100.0)
    assert item.proposed_value == 6100.0
    assert agg == {"valid": True, "removed": [], "added": []}


# ---------------------------------------------------------------------------
# compute_items_for_tab + get (pure read)
# ---------------------------------------------------------------------------


def test_get_writeback_items_is_a_pure_read():
    state = _make_state_with_tab()
    svc = _svc(state)
    # empty by default
    assert svc.get_tab_writeback_items("t1") == []
    # returns exactly what is persisted
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0)
    item.session_id = "md-1"
    _put_items(state, item)
    assert svc.get_tab_writeback_items("t1") == [item]


def test_compute_stamps_per_kind_session_ids():
    state = _make_state_with_tab()
    state.update_tab_result("t1", object())
    tab = state.get_tab("t1")
    analyze_result = MagicMock()  # passed in explicitly, not read from State

    md1 = MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0)
    wf1 = WaveformWriteback(target_name="wf_a", description="d")  # no edit_schema
    ml1 = ModuleWriteback(target_name="mod_a", description="d")  # no edit_schema
    wf2 = WaveformWriteback(target_name="wf_b", description="d")
    adapter: MagicMock = tab.adapter  # type: ignore[assignment]
    adapter.get_writeback_items.return_value = [md1, wf1, ml1, wf2]

    svc = _svc(state)
    items = svc.compute_items_for_tab("t1", analyze_result)

    ids = [it.session_id for it in items]
    assert ids == ["md-1", "wf-1", "ml-1", "wf-2"]
    assert all(it.selected for it in items)


# ---------------------------------------------------------------------------
# apply module/waveform via edited schema (no editor model)
# ---------------------------------------------------------------------------


def test_apply_module_writeback_registers_and_emits():
    state = _make_state_with_tab()
    bus = EventBus()
    received: list = []
    bus.subscribe(MlChangedPayload, lambda p: received.append(p))
    svc = _svc(state, bus)
    before = state.version.get("context")

    item = ModuleWriteback(target_name="qub", description="d", edit_schema=MagicMock())
    item.session_id = "ml-1"  # no editor_id → falls back to edit_schema
    _put_items(state, item)

    applied = svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))

    assert applied["applied_ids"] == ["ml-1"]
    assert applied["written"]["ml_modules"] == ["qub"]
    assert state.version.get("context") == before + 1
    assert len(received) == 1


def test_apply_waveform_writeback_registers():
    state = _make_state_with_tab()
    svc = _svc(state)

    item = WaveformWriteback(
        target_name="gauss", description="d", edit_schema=MagicMock()
    )
    item.session_id = "wf-1"
    _put_items(state, item)

    applied = svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))
    assert applied["applied_ids"] == ["wf-1"]
    assert applied["written"]["ml_waveforms"] == ["gauss"]


def test_apply_ml_writeback_calls_dump_when_has_persistence():
    state = _make_state_with_tab()
    mock_ml = MagicMock()
    mock_ml.has_persistence = True
    from dataclasses import replace as dc_replace

    state.set_context(dc_replace(state.exp_context, ml=mock_ml))
    svc = _svc(state)

    item = ModuleWriteback(target_name="qub", description="d", edit_schema=MagicMock())
    item.session_id = "ml-1"
    _put_items(state, item)

    svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))
    mock_ml.dump.assert_called_once()


def test_apply_module_no_editable_schema_raises():
    state = _make_state_with_tab()
    svc = _svc(state)

    item = ModuleWriteback(target_name="qub", description="d")  # no schema, no editor
    item.session_id = "ml-1"
    _put_items(state, item)

    with pytest.raises(RuntimeError, match="no editable schema"):
        svc.apply_tab_writeback(WritebackPermit(tab_id="t1"))


# ---------------------------------------------------------------------------
# set_item_field (tab.writeback_set)
# ---------------------------------------------------------------------------


def test_set_item_field_edits_persistent_item():
    state = _make_state_with_tab()
    svc = _svc(state)
    item = MetaDictWriteback(target_name="r_f", description="d", proposed_value=1.0)
    item.session_id = "md-1"
    _put_items(state, item)

    svc.set_item_field(
        "t1", "md-1", selected=False, target_name="r_f2", proposed_value=9.0
    )
    assert item.selected is False
    assert item.target_name == "r_f2"
    assert item.proposed_value == 9.0


def test_set_item_field_proposed_value_on_module_rejected():
    state = _make_state_with_tab()
    svc = _svc(state)
    item = ModuleWriteback(target_name="qub", description="d", edit_schema=MagicMock())
    item.session_id = "ml-1"
    _put_items(state, item)

    with pytest.raises(RuntimeError, match="not a metadict"):
        svc.set_item_field("t1", "ml-1", proposed_value=1.0)


def test_set_item_field_unknown_id_raises():
    state = _make_state_with_tab()
    svc = _svc(state)
    with pytest.raises(RuntimeError, match="unknown writeback session_id"):
        svc.set_item_field("t1", "md-99", selected=True)


# ---------------------------------------------------------------------------
# teardown_tab_items (reanalyze / rerun)
# ---------------------------------------------------------------------------


def test_teardown_tab_items_tears_down_editor_models():
    state = _make_state_with_tab()
    cfg_editor = MagicMock()
    svc = WritebackService(state, EventBus(), cfg_editor, MagicMock())

    item = ModuleWriteback(target_name="qub", description="d", edit_schema=MagicMock())
    item.session_id = "ml-1"
    item.editor_id = "editor-7"
    _put_items(state, item)

    svc.teardown_tab_items("t1")
    cfg_editor.teardown.assert_called_once_with("editor-7")
