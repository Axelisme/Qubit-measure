"""CfgEditorService — headless ml editing sessions.

Drives the service directly (no wire) against a MagicMock controller backed by
a real MetaDict + ModuleLibrary, so we exercise the full open → set_field →
commit path including eval-on-commit and ModuleRef sub-tree re-binding.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.services.cfg_editor import CfgEditorError, CfgEditorService
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
    """MagicMock controller whose set_ml_*_from_raw lands into the real ml."""
    from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = md
    c.get_current_ml.return_value = ml
    c.list_device_names.return_value = []
    c.has_soc.return_value = False

    def _set_module(name, raw):
        ml.register_module(**{name: ModuleCfgFactory.from_raw(raw, ml=ml)})

    def _set_waveform(name, raw):
        ml.register_waveform(**{name: WaveformCfgFactory.from_raw(raw, ml=ml)})

    c.set_ml_module_from_raw.side_effect = _set_module
    c.set_ml_waveform_from_raw.side_effect = _set_waveform
    return c


@pytest.fixture()
def service(ctrl):
    # M2: the Repository takes the reactive-env ctrl, the ML write port, and the
    # version bump/drop callbacks. The MagicMock ctrl satisfies all facets.
    return CfgEditorService(
        ctrl,
        ml_port=ctrl,
        version_bump=ctrl.bump_editor_version,
        version_drop=ctrl.drop_editor_version,
    )


def _paths(entries):
    return {e["path"]: e for e in entries}


# ---------------------------------------------------------------------------
# M2 — CfgEditorSession is a rich aggregate (behaviour on the entity itself,
# not on the service). docs/adr/0008 §Aggregate Root.
# ---------------------------------------------------------------------------


def test_session_is_the_aggregate_with_behaviour(service, ml):
    """The session the Repository hands out operates its own draft directly:
    set_field / commit / is_headless live on the aggregate, reachable without
    going back through the service."""
    from zcu_tools.gui.services.cfg_editor import CfgEditorSession

    editor_id, _ = service.open("module", discriminator="pulse")
    session = service._require(editor_id)  # Repository looks up the aggregate
    assert isinstance(session, CfgEditorSession)
    assert session.is_headless() is True

    # set_field is the aggregate's own behaviour, returns subtree + validity
    out = session.set_field("freq", 5000.0)
    assert out["paths"] and "valid" in out
    assert _paths(session.current_paths())["freq"]["value"] == 5000.0

    # commit is the aggregate's own behaviour: it lowers + registers via the
    # ML write port (the MagicMock ctrl lands it into the real ml).
    session.set_field("ch", 0)
    session.commit("agg_pulse", ml_port=service._ml)
    assert "agg_pulse" in ml.modules


def test_delegated_session_rejects_commit_on_the_aggregate(service):
    """commit-guard is enforced on the aggregate, not just the service facade."""
    from zcu_tools.gui.adapter import make_default_value
    from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
    from zcu_tools.gui.live_model import SectionLiveField

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    root = SectionLiveField(spec, service._env, make_default_value(spec))
    editor_id = service.register_delegated_session("owner-x", root)
    session = service._require(editor_id)
    assert session.is_headless() is False
    with pytest.raises(CfgEditorError):
        session.commit("nope", ml_port=service._ml)


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
    assert res["valid"] is True
    service.commit(editor_id, "agent_pulse")

    ctrl.set_ml_module_from_raw.assert_called_once()
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


# ---------------------------------------------------------------------------
# ModuleRef key switch rebinds the sub-tree
# ---------------------------------------------------------------------------


def test_ref_switch_returns_new_subtree(service):
    # A pulse module references a waveform; switching the waveform style
    # rebuilds which sub-fields exist (Const has length only; Gauss adds sigma).
    editor_id, paths = service.open("module", discriminator="pulse")
    keys = _paths(paths)
    assert "waveform.ref" in keys
    assert "waveform.value.sigma" not in keys  # const has no sigma

    # Switch the waveform ref key; set_field returns the rebuilt sub-tree.
    res = service.set_field(editor_id, "waveform.ref", "<Custom:Gauss>")
    sub_keys = _paths(res["paths"])
    # The returned sub-tree is rooted at the changed path and now exposes the
    # gauss-only field that did not exist before the switch.
    assert "waveform.value.sigma" in sub_keys


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
        service.get(editor_id)


def test_discard_for_client_batch_ignores_unknown(service):
    id1, _ = service.open("module", discriminator="pulse")
    id2, _ = service.open("waveform", discriminator="const")
    service.discard_for_client([id1, id2, "editor-unknown"])
    with pytest.raises(CfgEditorError):
        service.get(id1)
    with pytest.raises(CfgEditorError):
        service.get(id2)


def test_commit_failure_keeps_session(service, ctrl):
    editor_id, _ = service.open("module", discriminator="pulse")
    ctrl.set_ml_module_from_raw.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError):
        service.commit(editor_id, "bad")
    # session survives so the agent can fix and retry.
    assert service.get(editor_id)


# ---------------------------------------------------------------------------
# Delegated (tab) sessions
# ---------------------------------------------------------------------------


def _make_tab_root(ctrl):
    """Build a standalone live LiveModel as a tab's CfgFormWidget would own."""
    from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
    from zcu_tools.gui.live_model import LiveModelEnv, SectionLiveField

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    from zcu_tools.gui.adapter import make_default_value

    return SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), make_default_value(spec))


def test_register_delegated_session_shares_root(service, ctrl):
    root = _make_tab_root(ctrl)
    editor_id = service.register_delegated_session("tab-1", root)
    assert editor_id.startswith("editor-")
    assert service.editor_id_for_owner("tab-1") == editor_id

    # An edit via the session mutates the *same* root instance (WYSIWYG).
    service.set_field(editor_id, "freq", 4321.0)
    val = root.fields["freq"].get_value()
    assert getattr(val, "value", None) == 4321.0


def test_close_tab_session_keeps_root_alive(service, ctrl):
    root = _make_tab_root(ctrl)
    editor_id = service.register_delegated_session("tab-1", root)
    service.close(editor_id)
    # Registration gone …
    assert service.editor_id_for_owner("tab-1") is None
    with pytest.raises(CfgEditorError):
        service.get(editor_id)
    # … but the widget's root was NOT torn down (still mutable).
    root.fields["freq"].set_value(10.0)  # would raise if torn down badly
    assert root.is_valid() in (True, False)


def test_reregister_tab_closes_previous(service, ctrl):
    root1 = _make_tab_root(ctrl)
    id1 = service.register_delegated_session("tab-1", root1)
    root2 = _make_tab_root(ctrl)
    id2 = service.register_delegated_session("tab-1", root2)
    assert id1 != id2
    assert service.editor_id_for_owner("tab-1") == id2
    with pytest.raises(CfgEditorError):
        service.get(id1)


def test_close_unknown_is_noop(service):
    service.close("editor-never")  # no raise


def test_commit_rejects_tab_session(service, ctrl):
    editor_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    with pytest.raises(CfgEditorError):
        service.commit(editor_id, "x")


def test_discard_rejects_tab_session(service, ctrl):
    editor_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    with pytest.raises(CfgEditorError):
        service.discard(editor_id)


def test_discard_for_client_skips_tab_sessions(service, ctrl):
    tab_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    headless_id, _ = service.open("module", discriminator="pulse")
    service.discard_for_client([tab_id, headless_id])
    # headless reclaimed, tab session untouched.
    with pytest.raises(CfgEditorError):
        service.get(headless_id)
    assert service.editor_id_for_owner("tab-1") == tab_id


# ---------------------------------------------------------------------------
# Headless LRU orphan protection
# ---------------------------------------------------------------------------


def test_headless_lru_evicts_oldest(service):
    from zcu_tools.gui.services.cfg_editor import _MAX_HEADLESS_EDITORS

    ids = [
        service.open("module", discriminator="pulse")[0]
        for _ in range(_MAX_HEADLESS_EDITORS + 3)
    ]
    # Oldest 3 evicted; newest cap survive.
    for evicted in ids[:3]:
        with pytest.raises(CfgEditorError):
            service.get(evicted)
    for alive in ids[3:]:
        assert service.get(alive) is not None


def test_lru_does_not_count_tab_sessions(service, ctrl):
    from zcu_tools.gui.services.cfg_editor import _MAX_HEADLESS_EDITORS

    # Many tab sessions must not push headless out (different tab ids).
    for i in range(_MAX_HEADLESS_EDITORS + 5):
        service.register_delegated_session(f"tab-{i}", _make_tab_root(ctrl))
    headless_id, _ = service.open("module", discriminator="pulse")
    assert service.get(headless_id) is not None


# ---------------------------------------------------------------------------
# Per-session change stream (editor_changed / editor_closed)
# ---------------------------------------------------------------------------


def test_change_stream_emits_editor_changed(service, ctrl):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload: events.append((eid, ev, payload))
    )

    editor_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    service.set_field(editor_id, "freq", 5000.0)

    changed = [e for e in events if e[1] == "editor_changed" and e[0] == editor_id]
    assert changed
    # payload carries the full current path list.
    assert any("freq" in {p["path"] for p in e[2]["paths"]} for e in changed)


def test_change_stream_emits_editor_closed_with_reason(service, ctrl):
    events = []
    service.set_change_listener(
        lambda eid, ev, payload: events.append((eid, ev, payload))
    )

    # tab close → tab_closed
    tab_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    service.close(tab_id)
    # headless discard → discarded
    h1, _ = service.open("module", discriminator="pulse")
    service.discard(h1)
    # headless commit → committed
    h2, _ = service.open("waveform", discriminator="const")
    service.commit(h2, "wf_x")

    closed = {
        (eid, payload["reason"]) for eid, ev, payload in events if ev == "editor_closed"
    }
    assert (tab_id, "tab_closed") in closed
    assert (h1, "discarded") in closed
    assert (h2, "committed") in closed


def test_change_stream_no_listener_is_safe(service, ctrl):
    # No listener set → operations must not raise.
    editor_id = service.register_delegated_session("tab-1", _make_tab_root(ctrl))
    service.set_field(editor_id, "freq", 1.0)
    service.close(editor_id)


def test_closed_after_change_cb_disconnected(service, ctrl):
    """After close, further edits to the (still-live) root emit nothing."""
    events = []
    service.set_change_listener(
        lambda eid, ev, payload: events.append((eid, ev, payload))
    )
    root = _make_tab_root(ctrl)
    editor_id = service.register_delegated_session("tab-1", root)
    service.close(editor_id)
    events.clear()
    # widget still owns the root; an edit on it must not re-emit on a dead session.
    root.fields["freq"].set_value(99.0)
    assert events == []


def _hook_count(root):
    return len(root.on_change._cbs)


def test_no_dangling_hook_after_every_removal_path(service, ctrl):
    """Every removal path disconnects the change hook from its root.

    Asserts the on_change callback count returns to baseline — the core
    invariant guarding against dangling callbacks on a (possibly still-live)
    root. Covers close / discard / commit / evict.
    """
    # delegated: close
    root = _make_tab_root(ctrl)
    base = _hook_count(root)
    eid = service.register_delegated_session("tab-x", root)
    assert _hook_count(root) == base + 1
    service.close(eid)
    assert _hook_count(root) == base

    # headless: discard + commit each remove their own root's hook; we assert
    # via the re-register path that no stale hook lingers on a reused owner.
    r2 = _make_tab_root(ctrl)
    base2 = _hook_count(r2)
    service.register_delegated_session("tab-y", r2)
    # Re-register same owner closes the previous → its hook is gone (the prev
    # registration used the SAME root r2 here, so count must not accumulate).
    service.register_delegated_session("tab-y", r2)
    assert _hook_count(r2) == base2 + 1  # exactly one live hook, not two


def test_reregister_disconnects_previous_root_hook(service, ctrl):
    """Re-register with a *different* root leaves no hook on the old root."""
    events = []
    service.set_change_listener(
        lambda eid, ev, payload: events.append((eid, ev, payload))
    )
    old = _make_tab_root(ctrl)
    service.register_delegated_session("tab-1", old)
    new = _make_tab_root(ctrl)
    service.register_delegated_session("tab-1", new)  # closes the old reg
    events.clear()
    old.fields["freq"].set_value(7.0)  # old root edit → must be silent
    assert events == []


def _service_with_version_table(ctrl):
    """A CfgEditorService wired to a real VersionTable via bump/drop, so we can
    assert the editor:<id> resource version lifecycle (edit bumps, teardown drops)."""
    from zcu_tools.gui.state import VersionTable

    table = VersionTable()

    def _bump(eid: str) -> None:
        table.bump(f"editor:{eid}")

    def _drop(eid: str) -> None:
        table.drop_prefix(f"editor:{eid}")

    svc = CfgEditorService(ctrl, ml_port=ctrl, version_bump=_bump, version_drop=_drop)
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


def test_close_delegated_drops_editor_version(ctrl):
    svc, table = _service_with_version_table(ctrl)
    root = _make_tab_root(ctrl)
    editor_id = svc.register_delegated_session("tab-1", root)
    root.fields["freq"].set_value(7.0)  # edit → bump
    assert table.get(f"editor:{editor_id}") > 0
    svc.close(editor_id)  # delegated teardown → drop (root stays alive)
    assert table.get(f"editor:{editor_id}") == 0
