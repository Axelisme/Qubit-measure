"""CfgEditorService — headless ml editing sessions.

Drives the service directly (no wire) against a MagicMock controller backed by
a real MetaDict + ModuleLibrary, so we exercise the full open → set_field →
commit path including eval-on-commit and ModuleRef sub-tree re-binding.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from zcu_tools.gui.app.main.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorError, CfgEditorService
from zcu_tools.gui.app.main.services.ports import CfgEdit
from zcu_tools.gui.cfg import (
    CenteredSweepSpec,
    CfgSchema,
    CfgSectionSpec,
    SweepSpec,
    make_default_value,
)
from zcu_tools.gui.cfg.binding import SettablePathError
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.value_lookup import (
    MissingValue,
    ProviderError,
    UnavailableValue,
    ValueInfo,
    ValueTypeError,
)
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


@pytest.fixture()
def ml():
    return ModuleLibrary()


@pytest.fixture()
def md():
    m = MetaDict()
    m.r_f = 6012.5
    return m


@pytest.fixture()
def ctrl(ml, md):
    """MagicMock controller whose set_ml_*_from_schema lands into the real ml.

    Mirrors ContextService (ADR-0006 single write authority): the write port
    lowers the CfgSchema against the live ml/md, then registers.
    """
    from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = md
    c.get_current_ml.return_value = ml
    c.list_device_names.return_value = []
    c.list_arb_waveforms.return_value = []
    c.has_soc.return_value = False

    def _set_module(name, schema):
        raw = schema_to_raw_dict(schema, md, ml)
        ml.register_module(**{name: ModuleCfgFactory.from_raw(raw, ml=ml)})

    def _set_waveform(name, schema):
        raw = schema_to_raw_dict(schema, md, ml)
        ml.register_waveform(**{name: WaveformCfgFactory.from_raw(raw, ml=ml)})

    c.set_ml_module_from_schema.side_effect = _set_module
    c.set_ml_waveform_from_schema.side_effect = _set_waveform
    return c


@pytest.fixture()
def service(ctrl):
    # The Repository takes the reactive-env ctrl, the context read + write ports
    # (ADR-0006), the version bump/drop callbacks, and the EventBus (for
    # owned-model refresh). The MagicMock ctrl satisfies all facets.
    return CfgEditorService(
        ctrl,
        read_port=ctrl,
        write_port=ctrl,
        version_bump=ctrl.bump_editor_version,
        version_drop=ctrl.drop_editor_version,
        bus=EventBus(),
    )


def _paths(entries):
    from zcu_tools.gui.app.main.services.remote.path_resolver import project_targets

    return {e["path"]: e for e in project_targets(entries)}


# ---------------------------------------------------------------------------
# M2 — CfgEditorSession is a rich aggregate (behaviour on the entity itself,
# not on the service). docs/adr/0008 §Aggregate Root.
# ---------------------------------------------------------------------------


def test_session_is_the_aggregate_with_behaviour(service, ml):
    """The session the Repository hands out operates its own draft directly:
    set_field / commit live on the aggregate, reachable without going back
    through the service."""
    from zcu_tools.gui.app.main.services.cfg_editor import CfgEditorSession

    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)  # Repository looks up the aggregate
    assert isinstance(session, CfgEditorSession)
    assert session.item_kind == "module"  # ml-entry session → committable

    # set_field is the aggregate's own behaviour: returns validity + ref-switch
    # diff, NOT cfg content (no lowering/eval side effect). The new value is read
    # back via current_targets.
    out = session.set_field("freq", 5000.0)
    assert out.valid is True
    assert out.removed == () and out.added == ()
    assert _paths(session.current_targets())["freq"]["value"] == 5000.0

    # commit_schema is the aggregate's own behaviour: it yields the un-lowered
    # CfgSchema (ADR-0006 — lowering + register belong to the write authority).
    session.set_field("ch", 0)
    from zcu_tools.gui.cfg import CfgSchema

    schema = session.commit_schema()
    assert isinstance(schema, CfgSchema)
    # The service-level commit routes that schema through the write port.
    service.commit(editor_id, "agg_pulse")
    assert "agg_pulse" in ml.modules


def test_seeded_session_rejects_commit_on_the_aggregate(service):
    """commit-guard (item_kind is None) is enforced on the aggregate, not just
    the service facade. A seeded session (tab cfg / writeback draft) is
    teardown-only."""
    from zcu_tools.gui.app.main.specs import make_pulse_spec
    from zcu_tools.gui.cfg import (
        CfgSchema,
        make_default_value,
    )

    spec = make_pulse_spec()
    seed = CfgSchema(spec=spec, value=make_default_value(spec))
    editor_id, _ = service.open_seeded(seed, gc=False, owner_key="owner-x")
    session = service._require(editor_id)
    assert session.item_kind is None
    with pytest.raises(CfgEditorError):
        session.commit_schema()


# ---------------------------------------------------------------------------
# open
# ---------------------------------------------------------------------------


def test_open_blank_module_returns_paths(service):
    editor_id, paths = service.open("module", discriminator="pulse")
    assert editor_id.startswith("editor-")
    keys = _paths(paths)
    assert "freq" in keys
    assert keys["freq"]["kind"] == "scalar"


def test_open_unknown_kind_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("widget", discriminator="pulse")


def test_open_needs_discriminator_or_from_name(service):
    with pytest.raises(CfgEditorError):
        service.open("module")


def test_open_unknown_module_type_fails(service):
    with pytest.raises(CfgEditorError):
        service.open("module", discriminator="nonsense")


def test_open_from_existing_entry_returns_concrete_values(service, ml):
    # Seed an entry, then open it for editing.
    editor_id, _ = service.open("waveform", discriminator="const")
    service.commit(editor_id, "wf_seed")
    assert "wf_seed" in ml.waveforms

    _, paths = service.open("waveform", from_name="wf_seed")
    keys = _paths(paths)
    assert "length" in keys
    # scalar comes back as a concrete value, not an EvalValue/dict.
    assert not isinstance(keys["length"]["value"], dict)


# ---------------------------------------------------------------------------
# set_field + commit (eval-on-commit)
# ---------------------------------------------------------------------------


def test_set_scalar_then_commit_lands_concrete(service, ctrl, ml):
    editor_id, _ = service.open("module", discriminator="pulse")
    res = service.set_field(editor_id, "freq", 5000.0)
    assert res.valid is True
    service.commit(editor_id, "agent_pulse")

    ctrl.set_ml_module_from_schema.assert_called_once()
    assert "agent_pulse" in ml.modules
    assert ml.modules["agent_pulse"].to_dict()["freq"] == 5000.0


def test_eval_value_resolved_against_md_on_commit(service, ml, md):
    editor_id, _ = service.open("module", discriminator="pulse")
    service.set_field(editor_id, "freq", {"__kind": "eval", "expr": "r_f"})
    service.commit(editor_id, "agent_eval")

    # commit lowered EvalValue('r_f') against md -> concrete md.r_f.
    assert ml.modules["agent_eval"].to_dict()["freq"] == md.r_f


def test_eval_value_requires_string_expr(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    with pytest.raises(CfgEditorError):
        service.set_field(editor_id, "freq", {"__kind": "eval", "expr": 123})


def test_value_ref_resolves_to_concrete_direct_value_on_set_field(service, ctrl, ml):
    ctrl.read_value_source.return_value = (
        ValueInfo("device.flux.value", float, "device:flux"),
        6.25,
    )
    editor_id, _ = service.open("module", discriminator="pulse")

    service.set_field(
        editor_id,
        "freq",
        {
            "__kind": "value_ref",
            "key": "device.flux.value",
            "type": "float",
        },
    )
    service.commit(editor_id, "agent_ref")

    assert ml.modules["agent_ref"].to_dict()["freq"] == 6.25
    ctrl.read_value_source.assert_called_once_with("device.flux.value", "float")


def test_value_ref_requires_string_key(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    with pytest.raises(CfgEditorError, match="string 'key'"):
        service.set_field(
            editor_id,
            "freq",
            {"__kind": "value_ref", "key": 123},
        )


@pytest.mark.parametrize(
    "error",
    (
        MissingValue("device.flux.value", "missing"),
        UnavailableValue("device.flux.value", "unavailable"),
        ProviderError("device.flux.value", "device:flux", RuntimeError("boom")),
    ),
)
def test_value_ref_lookup_error_identity_and_category_are_preserved(
    service, ctrl, error
):
    ctrl.read_value_source.side_effect = error
    editor_id, _ = service.open("module", discriminator="pulse")

    with pytest.raises(type(error)) as exc:
        service.set_field(
            editor_id,
            "freq",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
        )

    assert exc.value is error


def test_value_ref_type_error_keeps_nominal_invalid_input(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    with pytest.raises(ValueTypeError):
        service.set_field(
            editor_id,
            "freq",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "str"},
        )


@pytest.mark.parametrize("path", ("sweep.start", "sweep.stop", "centered.center"))
def test_wire_eval_value_is_rejected_for_sweep_edges(service, path: str):
    spec = CfgSectionSpec(
        fields={"sweep": SweepSpec(), "centered": CenteredSweepSpec()}
    )
    editor_id, _ = service.open_seeded(
        CfgSchema(spec, make_default_value(spec)), gc=False
    )

    with pytest.raises(SettablePathError, match="only valid for scalar"):
        service.set_field(editor_id, path, {"__kind": "eval", "expr": "r_f"})


# ---------------------------------------------------------------------------
# ModuleRef key switch rebinds the sub-tree
# ---------------------------------------------------------------------------


def test_ref_switch_returns_new_subtree_and_diff(service):
    # A pulse module references a waveform; switching the waveform style
    # rebuilds which sub-fields exist (Const has length only; Gauss adds sigma).
    # ModuleRef sub-fields descend directly — no 'value' wrapper segment.
    editor_id, paths = service.open("module", discriminator="pulse")
    keys = _paths(paths)
    assert "waveform.ref" in keys
    assert "waveform.sigma" not in keys  # const has no sigma

    # Switch the waveform ref key; set_field returns only a removed/added diff of
    # settable paths (no cfg content — no lowering/eval side effect).
    res = service.set_field(editor_id, "waveform.ref", "<Custom:Gauss>")
    assert isinstance(res.valid, bool)
    # The diff names the appeared path explicitly so the agent need not re-list.
    assert "waveform.sigma" in res.added
    assert "waveform.sigma" not in res.removed
    assert "waveform.sigma" in _paths(service._require(editor_id).current_targets())


def test_non_shape_batch_does_not_materialize_path_sets(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)
    iter_targets = MagicMock(wraps=session.draft.iter_settable_targets)
    session.draft.iter_settable_targets = iter_targets

    result = session.set_fields((CfgEdit("freq", 5000.0), CfgEdit("gain", 0.4)))

    assert result.removed == () and result.added == ()
    iter_targets.assert_not_called()


def test_shape_batch_lists_once_before_and_after_and_returns_net_diff(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)
    iter_targets = MagicMock(wraps=session.draft.iter_settable_targets)
    session.draft.iter_settable_targets = iter_targets

    result = session.set_fields(
        (
            CfgEdit("waveform.ref", "Gauss"),
            CfgEdit("waveform.ref", "Const"),
        )
    )

    assert result.removed == () and result.added == ()
    assert iter_targets.call_count == 2


def test_shape_batch_failure_does_not_materialize_after_path_set(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)
    iter_targets = MagicMock(wraps=session.draft.iter_settable_targets)
    session.draft.iter_settable_targets = iter_targets

    with pytest.raises(SettablePathError, match="unknown settable path"):
        session.set_fields(
            (
                CfgEdit("waveform.ref", "Gauss"),
                CfgEdit("waveform.does_not_exist", 1),
            )
        )

    assert iter_targets.call_count == 1
    assert session.draft.resolve_target("waveform.ref").get_value() == "<Custom:Gauss>"


def test_batch_bumps_editor_version_once_per_successful_edit(service, ctrl):
    editor_id, _ = service.open("module", discriminator="pulse")
    ctrl.bump_editor_version.reset_mock()

    service.set_fields(editor_id, (CfgEdit("freq", 5000.0), CfgEdit("gain", 0.25)))

    assert ctrl.bump_editor_version.call_args_list == [call(editor_id), call(editor_id)]


def test_batch_is_ordered_fail_fast_non_atomic(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)

    with pytest.raises(SettablePathError, match="unknown settable path"):
        session.set_fields(
            (
                CfgEdit("freq", 4321.0),
                CfgEdit("does.not.exist", 1),
                CfgEdit("gain", 0.9),
            )
        )

    paths = _paths(session.current_targets())
    assert paths["freq"]["value"] == 4321.0
    assert paths["gain"]["value"] != 0.9


def test_empty_batch_returns_live_validity_without_path_listing(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)
    iter_targets = MagicMock(wraps=session.draft.iter_settable_targets)
    session.draft.iter_settable_targets = iter_targets

    result = session.set_fields(())

    assert result.valid is session.draft.is_valid()
    assert result.removed == () and result.added == ()
    iter_targets.assert_not_called()


# ---------------------------------------------------------------------------
# discard + unknown id
# ---------------------------------------------------------------------------


def test_unknown_editor_id_fails(service):
    with pytest.raises(CfgEditorError):
        service.set_field("editor-nope", "freq", 1.0)
    with pytest.raises(CfgEditorError):
        service.commit("editor-nope", "x")
    with pytest.raises(CfgEditorError):
        service.discard("editor-nope")


def test_discard_removes_session(service):
    editor_id, _ = service.open("module", discriminator="pulse")
    service.discard(editor_id)
    with pytest.raises(CfgEditorError):
        service.get_draft(editor_id)


def test_discard_for_client_batch_ignores_unknown(service):
    id1, _ = service.open("module", discriminator="pulse")
    id2, _ = service.open("waveform", discriminator="const")
    service.discard_for_client([id1, id2, "editor-unknown"])
    with pytest.raises(CfgEditorError):
        service.get_draft(id1)
    with pytest.raises(CfgEditorError):
        service.get_draft(id2)


def test_commit_failure_keeps_session(service, ctrl):
    editor_id, _ = service.open("module", discriminator="pulse")
    ctrl.set_ml_module_from_schema.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        service.commit(editor_id, "bad")
    # session survives so the agent can fix and retry.
    assert service.get_draft(editor_id)


# ---------------------------------------------------------------------------
# Seeded (UI-owned, gc=False) sessions — tab cfg / inspect / writeback
# ---------------------------------------------------------------------------


def _make_tab_seed():
    """Build a CfgSchema seed as a tab cfg / writeback draft would carry."""
    from zcu_tools.gui.app.main.specs import make_pulse_spec
    from zcu_tools.gui.cfg import (
        CfgSchema,
        make_default_value,
    )

    spec = make_pulse_spec()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_open_seeded_owns_model_and_is_addressable(service):
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    # Owner-keyed sessions derive a readable id from the owner (vs the opaque
    # 'editor-<hash>' of owner-less ml-entry sessions).
    assert editor_id == "tab-1-ed"
    assert service.editor_id_for_owner("tab-1") == editor_id

    # The widget attaches via get_draft; an agent edit mutates the same draft.
    draft = service.get_draft(editor_id)
    service.set_field(editor_id, "freq", 4321.0)
    val = draft.root.fields["freq"].get_value()
    assert getattr(val, "value", None) == 4321.0


def test_teardown_seeded_session_drops_it(service):
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    service.teardown(editor_id)
    assert service.editor_id_for_owner("tab-1") is None
    with pytest.raises(CfgEditorError):
        service.get_draft(editor_id)


def test_reopen_same_owner_tears_down_previous(service):
    id1, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    draft1 = service.get_draft(id1)
    id2, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    # Owner-keyed id is deterministic, so re-open reuses it — but the previous
    # session was torn down and replaced (a new draft tree), which is correct:
    # it is still that owner's editor, now pointing at the fresh draft.
    assert id2 == id1
    assert service.editor_id_for_owner("tab-1") == id2
    assert service.get_draft(id2) is not draft1


def test_teardown_unknown_is_noop(service):
    service.teardown("editor-never")  # no raise


def test_commit_rejects_seeded_session(service):
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    with pytest.raises(CfgEditorError):
        service.commit(editor_id, "x")


def test_discard_for_client_skips_gc_false_sessions(service):
    tab_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    gc_id, _ = service.open("module", discriminator="pulse")  # gc=True default
    service.discard_for_client([tab_id, gc_id])
    # gc=True reclaimed, gc=False (UI-owned) session untouched.
    with pytest.raises(CfgEditorError):
        service.get_draft(gc_id)
    assert service.editor_id_for_owner("tab-1") == tab_id


# ---------------------------------------------------------------------------
# gc LRU orphan protection
# ---------------------------------------------------------------------------


def test_gc_lru_evicts_oldest(service):
    from zcu_tools.gui.app.main.services.cfg_editor import _MAX_HEADLESS_EDITORS

    ids = [
        service.open("module", discriminator="pulse")[0]
        for _ in range(_MAX_HEADLESS_EDITORS + 3)
    ]
    # Oldest 3 evicted; newest cap survive.
    for evicted in ids[:3]:
        with pytest.raises(CfgEditorError):
            service.get_draft(evicted)
    for alive in ids[3:]:
        assert service.get_draft(alive) is not None


def test_lru_does_not_count_gc_false_sessions(service):
    from zcu_tools.gui.app.main.services.cfg_editor import _MAX_HEADLESS_EDITORS

    # Many gc=False sessions must not push gc=True ones out (different owners).
    for i in range(_MAX_HEADLESS_EDITORS + 5):
        service.open_seeded(_make_tab_seed(), gc=False, owner_key=f"tab-{i}")
    gc_id, _ = service.open("module", discriminator="pulse")
    assert service.get_draft(gc_id) is not None


# ---------------------------------------------------------------------------
# Per-session change stream (editor_changed / editor_closed)
# ---------------------------------------------------------------------------


def test_change_stream_emits_editor_changed(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )

    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    service.set_field(editor_id, "freq", 5000.0)

    changed = [e for e in events if e[1] == "editor_changed" and e[0] == editor_id]
    assert changed
    # payload carries the full current path list.
    assert any("freq" in {target.path for target in e[2]} for e in changed)


def test_change_stream_emits_editor_closed_with_reason(service):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )

    # seeded (UI-owned) teardown → tab_closed (default reason)
    tab_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    service.teardown(tab_id)
    # ml-entry discard → discarded
    h1, _ = service.open("module", discriminator="pulse")
    service.discard(h1)
    # ml-entry commit → committed
    h2, _ = service.open("waveform", discriminator="const")
    service.commit(h2, "wf_x")

    closed = {
        (eid, payload["reason"]) for eid, ev, payload in events if ev == "editor_closed"
    }
    assert (tab_id, "tab_closed") in closed
    assert (h1, "discarded") in closed
    assert (h2, "committed") in closed


def test_change_stream_no_listener_is_safe(service):
    # No listener set → operations must not raise.
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    service.set_field(editor_id, "freq", 1.0)
    service.teardown(editor_id)


def test_closed_draft_rejects_residual_field_edits(service):
    """After teardown, cached fields fail fast and cannot re-emit."""
    events = []
    service.set_change_listener(
        lambda eid, ev, payload_factory: events.append((eid, ev, payload_factory()))
    )
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    draft = service.get_draft(editor_id)
    field = draft.root.fields["freq"]
    service.teardown(editor_id)
    events.clear()
    with pytest.raises(RuntimeError, match="closed"):
        field.set_value(99.0)
    assert events == []


def test_change_stream_defers_current_targets_but_always_bumps_version(service, ctrl):
    factories = []
    service.set_change_listener(
        lambda _eid, _ev, payload_factory: factories.append(payload_factory)
    )
    editor_id, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    session = service._require(editor_id)
    current_targets = MagicMock(wraps=session.current_targets)
    session.current_targets = current_targets
    ctrl.bump_editor_version.reset_mock()

    service.set_field(editor_id, "freq", 5000.0)

    ctrl.bump_editor_version.assert_called_once_with(editor_id)
    current_targets.assert_not_called()
    assert len(factories) == 1
    payload = factories[0]()
    assert "freq" in {target.path for target in payload}
    current_targets.assert_called_once_with()


def _hook_count(draft):
    return len(draft.on_change._callbacks)


def test_no_dangling_hook_after_every_removal_path(service):
    """Every removal path disconnects the change hook from its root.

    Covers teardown / discard / re-open. The session's root is service-owned;
    after removal its on_change hook must be gone.
    """
    # seeded: teardown
    eid, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-x")
    draft = service.get_draft(eid)
    assert _hook_count(draft) == 1
    service.teardown(eid)
    assert _hook_count(draft) == 0

    # re-open same owner tears down the previous → its hook is gone.
    id1, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-y")
    draft1 = service.get_draft(id1)
    id2, _ = service.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-y")
    assert _hook_count(draft1) == 0  # previous draft's hook gone
    assert _hook_count(service.get_draft(id2)) == 1  # new one live


def _service_with_version_table(ctrl):
    """A CfgEditorService wired to a real VersionTable via bump/drop, so we can
    assert the editor:<id> resource version lifecycle (edit bumps, teardown drops)."""
    from zcu_tools.gui.app.main.state import VersionTable
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    table = VersionTable()

    def _bump(eid: str) -> None:
        table.bump(f"editor:{eid}")

    def _drop(eid: str) -> None:
        table.drop_prefix(f"editor:{eid}")

    svc = CfgEditorService(
        ctrl,
        read_port=ctrl,
        write_port=ctrl,
        version_bump=_bump,
        version_drop=_drop,
        bus=EventBus(),
    )
    return svc, table


def test_commit_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open("module", discriminator="pulse")
    svc.set_field(editor_id, "freq", 5000.0)  # edit → bump
    svc.set_field(editor_id, "ch", 0)
    assert table.get(f"editor:{editor_id}") > 0
    svc.commit(editor_id, "committed_pulse")  # teardown → drop
    assert table.get(f"editor:{editor_id}") == 0


def test_discard_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open("waveform", discriminator="gauss")
    svc.set_field(editor_id, "length", 0.1)
    assert table.get(f"editor:{editor_id}") > 0
    svc.discard(editor_id)
    assert table.get(f"editor:{editor_id}") == 0


def test_teardown_seeded_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    editor_id, _ = svc.open_seeded(_make_tab_seed(), gc=False, owner_key="tab-1")
    svc.set_field(editor_id, "freq", 7.0)  # edit → bump
    assert table.get(f"editor:{editor_id}") > 0
    svc.teardown(editor_id)  # UI-owned teardown → drop
    assert table.get(f"editor:{editor_id}") == 0
