"""RemoteControlAdapter tab-cfg editing via the CfgEditorService session.

A tab's cfg draft is a service-owned ``CfgEditorSession`` keyed by the tab_id
(the same draft the open form attaches to). Agents read it with
``tab.get_cfg`` and edit it with ``tab.set_cfg`` or ``editor.set_field`` on the
tab's ``editor_id`` (from ``tab.snapshot``) — the same draft the GUI form uses,
so user + agent share one model (ADR-0013 F11).

Here the fixture opens a real seeded session owned by the tab on the real
Controller, then drives edits through ``editor.set_field`` and discovery
through ``tab.get_cfg`` (which reads that same session). Path-resolver edge
cases (sweep edges, literal rejection, unknown paths) each get a focused case.
"""

from __future__ import annotations

from typing import Any

import pytest

from ._helpers import Fixture, call, open_client

# ---------------------------------------------------------------------------
# Fixture: open a real seeded cfg-editor session owned by one tab
# ---------------------------------------------------------------------------


class _LiveFixture(Fixture):
    """Fixture with a real CfgEditorService session owned by one tab."""

    def __init__(self) -> None:
        super().__init__()
        from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
        from zcu_tools.gui.app.main.state import Session

        cfg = FakeAdapter().make_default_cfg(self.state.exp_context)
        self._tab_id = "tab-live"
        # Inject a Session so has_tab(tab-live) is True.
        self.state.add_tab(
            self._tab_id,
            Session(
                adapter_name="fake",
                adapter=FakeAdapter(),
                cfg_schema=cfg,
            ),
        )
        # Open the tab's cfg-editor session keyed by tab_id — exactly what
        # MainWindow.populate_cfg does. editor_id_for_owner(tab_id) now resolves
        # it, so tab.get_cfg reads it and tab.set_cfg/editor.set_field mutates it.
        self.editor_id, _ = self.ctrl.open_seeded_cfg_editor(
            cfg, gc=False, owner_key=self._tab_id
        )

    def get_value(self, path: str):
        """Read the current value of a path off the live session draft."""
        from zcu_tools.gui.app.main.services.remote.path_resolver import (
            list_settable_paths_full,
        )

        root = self.ctrl.get_cfg_editor_root(self.editor_id)
        for entry in list_settable_paths_full(root):
            if entry["path"] == path:
                return entry["value"]
        raise KeyError(path)


@pytest.fixture()
def lf(qapp):  # noqa: ARG001
    f = _LiveFixture()
    f.start()
    yield f
    f.stop()


def _set_field(sock, lf, path, value, rid="1"):
    return call(
        sock,
        "editor.set_field",
        {"editor_id": lf.editor_id, "path": path, "value": value},
        rid=rid,
    )


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------


def test_set_field_scalar_updates_session(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "reps", 42)
        assert resp["ok"] is True
        assert lf.get_value("reps") == 42
    finally:
        sock.close()


def test_set_field_scalar_float(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "gain", 0.25)
        assert resp["ok"] is True
        assert lf.get_value("gain") == 0.25
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Sweep edges
# ---------------------------------------------------------------------------


def test_set_field_sweep_expts(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "sweep.expts", 5)
        assert resp["ok"] is True
        assert lf.get_value("sweep.expts") == 5
    finally:
        sock.close()


def test_set_field_sweep_start_stop(lf):
    sock = open_client(lf.service.port)
    try:
        _set_field(sock, lf, "sweep.start", 2.0, rid="a")
        _set_field(sock, lf, "sweep.stop", 8.0, rid="b")
        assert lf.get_value("sweep.start") == 2.0
        assert lf.get_value("sweep.stop") == 8.0
    finally:
        sock.close()


def test_set_field_sweep_step(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "sweep.step", 0.5)
        assert resp["ok"] is True
    finally:
        sock.close()


def test_set_field_sweep_expts_non_integer_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "sweep.expts", 3.5)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_set_field_unknown_path_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(sock, lf, "does_not_exist", 1)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_set_field_unknown_editor_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "editor.set_field",
            {"editor_id": "ghost", "path": "reps", "value": 1},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_set_field_section_target_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        # 'sweep' alone targets a sweep container, not a leaf.
        resp = _set_field(sock, lf, "sweep", 1)
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Context / device queries
# ---------------------------------------------------------------------------


def test_context_get_md_keys(lf):
    # MagicMock md.keys() returns a MagicMock; patch to a concrete list.
    lf.state.exp_context.md.keys = lambda: ["t1", "freq"]
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get")
        assert resp["ok"] is True
        assert set(resp["result"]["keys"]) == {"t1", "freq"}
    finally:
        sock.close()


def test_context_get_md_attr_roundtrip(lf):
    md = lf.state.exp_context.md
    store = {"t1": 12.5}
    md.get = lambda key, default=None: store.get(key, default)
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get_attr", {"key": "t1"})
        assert resp["ok"] is True
        assert resp["result"]["value"] == 12.5
    finally:
        sock.close()


def test_context_get_md_attr_unknown_rejected(lf):
    md = lf.state.exp_context.md
    md.get = lambda key, default=None: default
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get_attr", {"key": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_context_get_ml_names(lf):
    # context.ml_get now surfaces each entry's discriminator: modules carry the
    # 'type' tag, waveforms the 'style' tag — so the stored values need those attrs.
    from types import SimpleNamespace

    ml = lf.state.exp_context.ml
    ml.modules = {
        "readout": SimpleNamespace(type="pulse"),
        "pi": SimpleNamespace(type="pulse"),
    }
    ml.waveforms = {"gauss": SimpleNamespace(style="gauss")}
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.ml_get")
        assert resp["ok"] is True
        assert resp["result"]["modules"] == [
            {"name": "pi", "kind": "pulse"},
            {"name": "readout", "kind": "pulse"},
        ]
        assert resp["result"]["waveforms"] == [{"name": "gauss", "style": "gauss"}]
    finally:
        sock.close()


def test_device_list_and_snapshot(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "device.list")
        assert resp["ok"] is True
        assert isinstance(resp["result"]["devices"], list)

        # An unknown device name is now a hard error (INVALID_PARAMS), not a
        # {snapshot: null} reply (C8: device.snapshot raises on unknown name).
        resp = call(sock, "device.snapshot", {"name": "does-not-exist"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# tab.get_cfg — returns a NESTED current-value tree built off the tab's
# cfg-editor session (ADR-0013 F11). Reserved '$'-keys: a dict with $value is
# an enum leaf, a dict with $ref is a ref node, any other dict is a sub-tree,
# a non-dict is a bare scalar value (null = unset, ADR-0010).
# ---------------------------------------------------------------------------


def test_tab_get_cfg_returns_nested_tree_with_scalar_values(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})
        assert resp["ok"] is True
        tree = resp["result"]["tree"]
        assert isinstance(tree, dict)
        # A scalar leaf is its bare current value (not a {path, kind, ...} entry).
        assert "reps" in tree
        assert not isinstance(tree["reps"], dict)
    finally:
        sock.close()


def test_tab_get_cfg_sweep_is_subtree_of_bare_edges(lf):
    sock = open_client(lf.service.port)
    try:
        tree = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})["result"]["tree"]
        # The fake adapter's sweep is exposed as a sub-tree of bare edges.
        assert "sweep" in tree
        sweep = tree["sweep"]
        assert isinstance(sweep, dict)
        assert set(sweep) == {"start", "stop", "expts", "step"}
        assert all(not isinstance(v, dict) for v in sweep.values())
    finally:
        sock.close()


def test_tab_get_cfg_prefix_returns_subtree(lf):
    sock = open_client(lf.service.port)
    try:
        full = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})["result"]["tree"]
        scoped = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id, "prefix": "sweep"})[
            "result"
        ]["tree"]
        # The prefix sub-tree equals the corresponding sub-dict of the full tree.
        assert scoped == full["sweep"]
    finally:
        sock.close()


def test_tab_get_cfg_prefix_scalar_leaf_wrapped_under_its_name(lf):
    sock = open_client(lf.service.port)
    try:
        # 'reps' is a scalar leaf; the prefix reply wraps it so the result is
        # always a dict keyed by the leaf name.
        resp = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id, "prefix": "reps"})
        tree = resp["result"]["tree"]
        assert set(tree) == {"reps"}
        assert not isinstance(tree["reps"], dict)
    finally:
        sock.close()


def test_tab_get_cfg_prefix_no_match_returns_empty_dict(lf):
    # A prefix matching nothing yields {} (graceful, not a fast-fail).
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id, "prefix": "nope.x"})
        assert resp["ok"] is True
        assert resp["result"]["tree"] == {}
    finally:
        sock.close()


def test_tab_get_cfg_unknown_tab_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_tab_get_cfg_form_not_populated_rejected(qapp):  # noqa: ARG001
    """A tab with no cfg-editor session yet → precondition_failed."""
    from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
    from zcu_tools.gui.app.main.state import Session

    f = Fixture()
    f.start()
    try:
        cfg = FakeAdapter().make_default_cfg(f.state.exp_context)
        f.state.add_tab(
            "bare",
            Session(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg),
        )
        sock = open_client(f.service.port)
        try:
            resp = call(sock, "tab.get_cfg", {"tab_id": "bare"})
            assert resp["ok"] is False
            assert resp["error"]["code"] == "precondition_failed"
        finally:
            sock.close()
    finally:
        f.stop()


# ---------------------------------------------------------------------------
# tab.set_cfg — batch setter; applies ordered {path, value} edits to the tab's
# cfg-editor session via the same controller path as editor.set_field.
# ---------------------------------------------------------------------------


def test_tab_set_cfg_scalar(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {"tab_id": lf._tab_id, "edits": [{"path": "reps", "value": 77}]},
        )
        assert resp["ok"] is True
        assert resp["result"]["valid"] is True
        assert lf.get_value("reps") == 77
    finally:
        sock.close()


def test_tab_set_cfg_sweep_edge(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {"tab_id": lf._tab_id, "edits": [{"path": "sweep.expts", "value": 9}]},
        )
        assert resp["ok"] is True
        assert lf.get_value("sweep.expts") == 9
    finally:
        sock.close()


def test_tab_set_cfg_batch_applies_in_order(lf):
    """Multiple edits in one call; result aggregates removed/added across them."""
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {
                "tab_id": lf._tab_id,
                "edits": [
                    {"path": "reps", "value": 10},
                    {"path": "sweep.expts", "value": 3},
                ],
            },
        )
        assert resp["ok"] is True
        assert lf.get_value("reps") == 10
        assert lf.get_value("sweep.expts") == 3
    finally:
        sock.close()


def test_tab_set_cfg_unknown_tab_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {"tab_id": "no-such-tab", "edits": [{"path": "reps", "value": 1}]},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_tab_set_cfg_bad_path_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "tab.set_cfg",
            {"tab_id": lf._tab_id, "edits": [{"path": "no.such.path", "value": 1}]},
        )
        assert resp["ok"] is False
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# build_settable_tree — direct unit tests for the tree node shapes (enum leaf,
# ref node with current+options+chosen-variant-only sub-tree). These use the
# fakefreq root (which has Module/Waveform refs + an enum scalar) so the $ref /
# $value / $choices shapes are exercised without a socket.
# ---------------------------------------------------------------------------


def _node(tree: dict[str, object], dotted: str) -> Any:
    """Walk a nested tree by a dotted path, returning the node (typed Any).

    The tree value type is ``object`` (a JSON-ish nested dict), so chained
    indexing trips the type checker; this helper narrows once at the boundary so
    the assertions stay readable.
    """
    node: Any = tree
    for seg in dotted.split("."):
        node = node[seg]
    return node


def test_tree_enum_scalar_leaf_has_value_and_choices(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import build_settable_tree

    root = _fakefreq_root()
    tree = build_settable_tree(root)
    # 'nqz' is an enum scalar (choices [1, 2]) under the readout pulse cfg.
    nqz = _node(tree, "modules.readout.pulse_cfg.nqz")
    assert nqz == {"$value": 2, "$choices": [1, 2]}


def test_tree_moduleref_node_current_options_and_variant_subtree(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import build_settable_tree

    root = _fakefreq_root()
    readout = _node(build_settable_tree(root), "modules.readout")
    # A ref node carries $ref {current, options} plus the chosen variant's
    # settable sub-tree (siblings of $ref).
    assert readout["$ref"]["current"] == "<Custom:Pulse Readout>"
    assert readout["$ref"]["options"] == ["Direct Readout", "Pulse Readout"]
    # Only the chosen ('Pulse Readout') variant is expanded — it has pulse_cfg.
    assert "pulse_cfg" in readout
    assert "ro_cfg" in readout


def test_tree_moduleref_only_chosen_variant_expanded(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import (
        build_settable_tree,
        resolve_and_set,
    )

    root = _fakefreq_root()
    # Switch to the 'Direct Readout' variant; the tree must now expand THAT
    # variant's sub-tree, not the previous one.
    resolve_and_set(root, "modules.readout.ref", "Direct Readout")
    readout = _node(build_settable_tree(root), "modules.readout")
    assert readout["$ref"]["current"] == "<Custom:Direct Readout>"
    # Direct Readout has no pulse_cfg sub-section (it is a different shape).
    assert "pulse_cfg" not in readout


def test_tree_omits_immutable_literal_fields(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import build_settable_tree

    root = _fakefreq_root()
    pulse_cfg = _node(build_settable_tree(root), "modules.readout.pulse_cfg")
    # 'type'/'freq' are literal (immutable) fields — not settable, so omitted
    # (they must NOT read as a settable null scalar).
    assert "type" not in pulse_cfg
    assert "freq" not in pulse_cfg


def test_tree_deviceref_node_current_and_options(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import build_settable_tree

    root = _fluxdep_root(["flux_yoko", "flux_yoko_2"])
    flux_dev = _node(build_settable_tree(root, "dev"), "flux_dev")
    # A device ref is a leaf selector: $ref with current + the live device list,
    # no settable sub-tree.
    assert flux_dev["$ref"]["current"] == "flux_yoko"
    assert flux_dev["$ref"]["options"] == ["flux_yoko", "flux_yoko_2"]
    assert set(flux_dev) == {"$ref"}


# ---------------------------------------------------------------------------
# ModuleRef key normalization — a bare variant label (as list_paths advertises
# in 'choices') is accepted and stored as the tagged <Custom:label> chosen_key,
# not mistaken for a (non-existent) library entry name. Regression: a bare label
# used to be stored verbatim → empty sub-field → lowering "Unknown module
# reference". Pure resolve_and_set unit (no socket): fake/freq has a readout
# ModuleRef with variant labels "Direct Readout" / "Pulse Readout".
# ---------------------------------------------------------------------------


def _fakefreq_root():
    from unittest.mock import MagicMock

    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    cfg = FakeFreqAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.get_current_md.return_value = ctx.md
    env = LiveModelEnv(ctrl=ctrl)
    return SectionLiveField(cfg.spec, env, initial_val=cfg.value)


def test_moduleref_bare_label_normalized_to_custom_tag(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import (
        list_settable_paths_full,
        resolve_and_set,
    )

    root = _fakefreq_root()
    # Bare label, exactly as tab.get_cfg advertises in 'choices'.
    resolve_and_set(root, "modules.readout.ref", "Direct Readout")

    entry = next(
        e for e in list_settable_paths_full(root) if e["path"] == "modules.readout.ref"
    )
    # Stored as the tagged key, not the bare label.
    assert entry["value"] == "<Custom:Direct Readout>"


def test_moduleref_tagged_key_passes_through(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import (
        list_settable_paths_full,
        resolve_and_set,
    )

    root = _fakefreq_root()
    # An already-tagged key is stored verbatim (no double-wrapping).
    resolve_and_set(root, "modules.readout.ref", "<Custom:Direct Readout>")

    entry = next(
        e for e in list_settable_paths_full(root) if e["path"] == "modules.readout.ref"
    )
    assert entry["value"] == "<Custom:Direct Readout>"


# ---------------------------------------------------------------------------
# DeviceRef path resolution — list_paths advertises a DeviceRef as
# '<path>.device' (mirroring a ModuleRef's '.ref'); the resolver must accept
# that exact advertised path. Regression: the '.device' suffix was advertised
# but the resolver had no branch for it, so set_field('dev.flux_dev.device')
# failed "descends into non-container DeviceRefLiveField" while list_paths told
# the agent to use it (discovery and setter were out of sync). onetone/flux_dep
# has a 'dev' section with a flux-device ref defaulting to 'flux_yoko'.
# ---------------------------------------------------------------------------


def _fluxdep_root(device_names: list[str]):
    from unittest.mock import MagicMock

    from zcu_tools.experiment.v2_gui.adapters.onetone.flux_dep import (
        OneToneFluxDepAdapter,
    )
    from zcu_tools.gui.app.main.adapter import ExpContext
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField
    from zcu_tools.meta_tool import MetaDict, ModuleLibrary

    ctx = ExpContext(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    cfg = OneToneFluxDepAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.get_current_md.return_value = ctx.md
    ctrl.list_device_names.return_value = list(device_names)
    env = LiveModelEnv(ctrl=ctrl)
    return SectionLiveField(cfg.spec, env, initial_val=cfg.value)


def _deviceref_value(root, path: str = "dev.flux_dev.device"):
    from zcu_tools.gui.app.main.services.remote.path_resolver import (
        list_settable_paths_full,
    )

    return next(e for e in list_settable_paths_full(root) if e["path"] == path)["value"]


def test_deviceref_advertised_path_is_dotted_device(qapp):  # noqa: ARG001
    """list_paths advertises the DeviceRef as '<path>.device' (kind deviceref)."""
    from zcu_tools.gui.app.main.services.remote.path_resolver import (
        list_settable_paths_full,
    )

    root = _fluxdep_root(["flux_yoko"])
    entry = next(
        e for e in list_settable_paths_full(root) if e["path"] == "dev.flux_dev.device"
    )
    assert entry["kind"] == "deviceref"


def test_deviceref_device_path_resolves(qapp):  # noqa: ARG001
    """The advertised '<path>.device' path round-trips through resolve_and_set."""
    from zcu_tools.gui.app.main.services.remote.path_resolver import resolve_and_set

    root = _fluxdep_root(["flux_yoko", "flux_yoko_2"])
    resolve_and_set(root, "dev.flux_dev.device", "flux_yoko_2")
    assert _deviceref_value(root) == "flux_yoko_2"


def test_deviceref_bare_leaf_path_also_resolves(qapp):  # noqa: ARG001
    """The bare-leaf form '<path>' (no '.device') is accepted as an alias."""
    from zcu_tools.gui.app.main.services.remote.path_resolver import resolve_and_set

    root = _fluxdep_root(["flux_yoko", "flux_yoko_2"])
    resolve_and_set(root, "dev.flux_dev", "flux_yoko_2")
    assert _deviceref_value(root) == "flux_yoko_2"


def test_deviceref_non_string_value_rejected(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import resolve_and_set
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

    root = _fluxdep_root(["flux_yoko"])
    with pytest.raises(RemoteError) as exc:
        resolve_and_set(root, "dev.flux_dev.device", 42)
    assert exc.value.code == ErrorCode.INVALID_PARAMS


def test_deviceref_bad_trailing_segment_rejected(qapp):  # noqa: ARG001
    from zcu_tools.gui.app.main.services.remote.path_resolver import resolve_and_set
    from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

    root = _fluxdep_root(["flux_yoko"])
    with pytest.raises(RemoteError) as exc:
        resolve_and_set(root, "dev.flux_dev.bogus", "flux_yoko")
    assert exc.value.code == ErrorCode.INVALID_PARAMS
