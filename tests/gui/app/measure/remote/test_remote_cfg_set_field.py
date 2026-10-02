"""RemoteControlAdapter tab-cfg editing via the CfgEditorService session.

A tab's cfg draft is a service-owned ``CfgEditorSession`` keyed by the tab_id
(the same draft the open form attaches to). Agents read it with
``tab.get_cfg`` and edit it with ``tab.set_cfg`` or ``editor.set_field`` on the
tab's ``editor_id`` (from ``tab.snapshot``) — the same draft the GUI form uses,
so user + agent share one model (ADR-0068).

Here the fixture opens a real seeded session owned by the tab on the real
Controller, then drives edits through ``editor.set_field`` and discovery
through ``tab.get_cfg`` (which reads that same session). Path-resolver edge
cases (sweep edges, literal rejection, unknown paths) each get a focused case.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue
from zcu_tools.gui.cfg.binding import CfgDraft

from ._helpers import Fixture, call, open_client


def _make_draft(
    ctrl: MagicMock, spec: CfgSectionSpec, value: CfgSectionValue
) -> CfgDraft:
    return MeasureCfgBindings(ctrl).new_draft(CfgSchema(spec, value))


def _set(draft: CfgDraft, path: str, value: object) -> None:
    draft.set_target(path, value)


# ---------------------------------------------------------------------------
# Fixture: open a real seeded cfg-editor session owned by one tab
# ---------------------------------------------------------------------------


class _LiveFixture(Fixture):
    """Fixture with a real CfgEditorService session owned by one tab."""

    def __init__(self) -> None:
        super().__init__()
        from zcu_tools.experiment.v2_gui.measure.adapters.fake import FakeAdapter

        cfg = FakeAdapter().make_default_cfg(self.state.session_env)
        self._tab_id = "tab-live"
        self.prepare_tab(self._tab_id, FakeAdapter(), cfg)
        # Independent library editor; it never publishes tab cfg.
        self.editor_id, _ = self.ctrl.open_seeded_cfg_editor(cfg, gc=False)

    def get_value(self, path: str):
        """Read the current value of a path off the live session draft."""
        from zcu_tools.gui.app.measure.remote.path_resolver import (
            project_target_entries,
        )

        draft = self.ctrl.get_cfg_editor_draft(self.editor_id)
        for entry in project_target_entries(draft):
            if entry["path"] == path:
                return entry["value"]
        raise KeyError(path)


@pytest.fixture()
def lf(qapp):
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


def test_value_ref_provider_failure_maps_to_controller_error(lf, monkeypatch):
    from zcu_tools.gui.session.value_lookup import ProviderError

    error = ProviderError("device.flux.value", "device:flux", RuntimeError("boom"))

    def fail_read(*_args):
        raise error

    monkeypatch.setattr(lf.ctrl, "read_value_source", fail_read)
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(
            sock,
            lf,
            "gain",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
        )
        assert resp["ok"] is False
        assert resp["error"] == {
            "code": "controller_error",
            "message": str(error),
        }
    finally:
        sock.close()


def test_value_ref_unavailable_maps_to_precondition_failed(lf, monkeypatch):
    from zcu_tools.gui.session.value_lookup import UnavailableValue

    error = UnavailableValue("device.flux.value", "flux device is unavailable")

    def fail_read(*_args):
        raise error

    monkeypatch.setattr(lf.ctrl, "read_value_source", fail_read)
    sock = open_client(lf.service.port)
    try:
        resp = _set_field(
            sock,
            lf,
            "gain",
            {"__kind": "value_ref", "key": "device.flux.value", "type": "float"},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
        assert resp["error"]["message"] == str(error)
        assert resp["error"].get("reason", "") == ""
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


def _centered_sweep_root():
    from zcu_tools.gui.cfg import (
        CenteredSweepSpec,
        CenteredSweepValue,
        CfgSectionSpec,
        CfgSectionValue,
    )

    ctrl = MagicMock()
    spec = CfgSectionSpec(fields={"sweep": CenteredSweepSpec(label="Freq")})
    value = CfgSectionValue(
        fields={"sweep": CenteredSweepValue(center=0.0, span=10.0, expts=11)}
    )
    return _make_draft(ctrl, spec, value)


def _locked_centered_sweep_root():
    from zcu_tools.gui.cfg import (
        CenteredSweepSpec,
        CenteredSweepValue,
        CfgSectionSpec,
        CfgSectionValue,
    )

    ctrl = MagicMock()
    spec = CfgSectionSpec(
        fields={
            "sweep": CenteredSweepSpec(
                label="Freq", center_editable=False, locked_center=0.0
            )
        }
    )
    value = CfgSectionValue(
        fields={"sweep": CenteredSweepValue(center=0.0, span=10.0, expts=11)}
    )
    return _make_draft(ctrl, spec, value)


def _single_point_centered_sweep_root():
    from zcu_tools.gui.cfg import (
        CenteredSweepSpec,
        CenteredSweepValue,
        CfgSectionSpec,
        CfgSectionValue,
    )

    ctrl = MagicMock()
    spec = CfgSectionSpec(fields={"sweep": CenteredSweepSpec(label="Freq")})
    value = CfgSectionValue(
        fields={"sweep": CenteredSweepValue(center=0.0, span=0.0, expts=1)}
    )
    return _make_draft(ctrl, spec, value)


def test_resolver_centered_sweep_edges(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.app.measure.remote.path_resolver import (
        project_target_entries,
    )

    root = _centered_sweep_root()

    _set(root, "sweep.center", 5.0)
    _set(root, "sweep.span", 20.0)
    _set(root, "sweep.expts", 5)

    entries = {entry["path"]: entry["value"] for entry in project_target_entries(root)}
    assert entries["sweep.center"] == 5.0
    assert entries["sweep.span"] == 20.0
    assert entries["sweep.expts"] == 5
    assert entries["sweep.step"] == pytest.approx(5.0)
    assert {
        key: item["resolved"]
        for key, item in _node(build_cfg_observation(root), "sweep")["inputs"].items()
    } == {
        "center": 5.0,
        "span": 20.0,
        "expts": 5,
        "step": 5.0,
    }


def test_resolver_centered_sweep_rejects_start_stop_edges(qapp):
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _centered_sweep_root()

    with pytest.raises(SettablePathError) as exc:
        _set(root, "sweep.start", 1.0)

    assert "unknown settable path" in str(exc.value)


def test_resolver_centered_sweep_rejects_locked_center_mismatch(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _locked_centered_sweep_root()

    with pytest.raises(SettablePathError) as exc:
        _set(root, "sweep.center", 5.0)

    assert "locked to 0.0" in str(exc.value)
    sweep = _node(build_cfg_observation(root), "sweep")
    assert sweep["inputs"]["center"]["resolved"] == 0.0


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        ("sweep.span", -1.0, "span"),
        ("sweep.span", 0.0, "span"),
        ("sweep.expts", 0, "expts"),
        ("sweep.step", -0.5, "step"),
    ),
)
def test_resolver_centered_sweep_value_errors_are_remote_errors(
    qapp,
    path: str,
    value: object,
    match: str,
):
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _centered_sweep_root()

    with pytest.raises(SettablePathError) as exc:
        _set(root, path, value)

    assert match in str(exc.value)


def test_resolver_centered_sweep_rejects_zero_span_promoted_to_multi_point(
    qapp,
):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _single_point_centered_sweep_root()

    with pytest.raises(SettablePathError) as exc:
        _set(root, "sweep.expts", 2)

    assert "span" in str(exc.value)
    assert {
        key: item["resolved"]
        for key, item in _node(build_cfg_observation(root), "sweep")["inputs"].items()
    } == {
        "center": 0.0,
        "span": 0.0,
        "expts": 1,
        "step": 0.0,
    }


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
    lf.state.session_env.md.update({"t1": 12.5, "freq": 5.0})
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get")
        assert resp["ok"] is True
        assert set(resp["result"]["keys"]) == {"t1", "freq"}
    finally:
        sock.close()


def test_context_get_md_attr_roundtrip(lf):
    md = lf.state.session_env.md
    md.update({"t1": 12.5})
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get_attr", {"key": "t1"})
        assert resp["ok"] is True
        assert resp["result"]["value"] == 12.5
    finally:
        sock.close()


def test_context_get_md_attr_unknown_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.md_get_attr", {"key": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_context_get_ml_names(lf):
    from types import SimpleNamespace

    ml = lf.state.session_env.ml
    ml.modules = {
        "readout": SimpleNamespace(type="pulse"),
        "pi": SimpleNamespace(type="pulse"),
    }
    ml.waveforms = {"gauss": SimpleNamespace(style="gauss")}
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.ml_get")
        assert resp["ok"] is True
        index = resp["result"]
        assert [m["name"] for m in index["modules"]] == ["pi", "readout"]
        assert all(m["kind"] == "pulse" for m in index["modules"])
        assert [(w["name"], w["style"]) for w in index["waveforms"]] == [
            ("gauss", "gauss")
        ]
        assert all(e["description"] for entries in index.values() for e in entries)
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
# tab.get_cfg returns complete cached model observations from the tab session.
# ---------------------------------------------------------------------------


def test_tab_get_cfg_returns_nested_tree_with_scalar_values(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})
        assert resp["ok"] is True
        tree = resp["result"]["tree"]
        assert isinstance(tree, dict)
        reps = tree["children"]["reps"]
        assert reps["kind"] == "scalar"
        assert reps["path"] == ["reps"]
        assert reps["input"]["resolved"] == lf.get_value("reps")
    finally:
        sock.close()


def test_tab_get_cfg_sweep_contains_input_states(lf):
    sock = open_client(lf.service.port)
    try:
        tree = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})["result"]["tree"]
        sweep = tree["children"]["sweep"]
        assert sweep["kind"] == "sweep"
        assert set(sweep["inputs"]) == {"start", "stop", "expts", "step"}
        assert sweep["inputs"]["expts"]["resolved"] == lf.get_value("sweep.expts")
    finally:
        sock.close()


def test_editor_read_rejects_unrepresentable_cfg_instead_of_stringifying(lf):
    from zcu_tools.gui.cfg import LiteralSpec, make_default_value
    from zcu_tools.gui.remote.errors import ErrorCode

    spec = CfgSectionSpec(fields={"fixed": LiteralSpec(object())})
    editor_id, _ = lf.ctrl.open_seeded_cfg_editor(
        CfgSchema(spec, make_default_value(spec)), gc=False
    )
    sock = open_client(lf.service.port)
    try:
        response = call(sock, "editor.get", {"editor_id": editor_id})
        assert not response["ok"]
        assert response["error"]["code"] == ErrorCode.CONTROLLER_ERROR.value
        assert "Unsupported cfg observation value" in response["error"]["message"]
    finally:
        sock.close()


def test_tab_get_cfg_unknown_tab_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
        assert resp["error"]["reason"] == "resource_gone"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# tab.set_cfg — batch setter; applies ordered {path, value} edits to the tab's
# cfg-editor session via the same controller path as editor.set_field.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Complete observations of production cfg shapes, including immutable fields
# and cached choices, through the public projection contract.
# ---------------------------------------------------------------------------


def _node(tree: dict[str, object], dotted: str) -> Any:
    """Walk a nested tree by a dotted path, returning the node (typed Any).

    The tree value type is ``object`` (a JSON-ish nested dict), so chained
    indexing trips the type checker; this helper narrows once at the boundary so
    the assertions stay readable.
    """
    node: Any = tree
    for seg in dotted.split("."):
        node = node["children"][seg]
    return node


def test_tree_enum_scalar_leaf_has_value_and_choices(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )

    root = _fakefreq_root()
    tree = build_cfg_observation(root)
    # 'nqz' is an enum scalar (choices [1, 2]) under the readout pulse cfg.
    nqz = _node(tree, "modules.readout.pulse_cfg.nqz")
    assert nqz["input"]["resolved"] == 2
    assert nqz["choices"] == [1, 2]


def test_tree_moduleref_node_current_options_and_variant_subtree(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )

    root = _fakefreq_root()
    readout = _node(build_cfg_observation(root), "modules.readout")
    assert readout["ref"] == "<Custom:Pulse Readout>"
    assert readout["choices"] == ["Direct Readout", "Pulse Readout"]
    assert "pulse_cfg" in readout["children"]
    assert "ro_cfg" in readout["children"]


def test_tree_moduleref_only_chosen_variant_expanded(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )

    root = _fakefreq_root()
    # Switch to the 'Direct Readout' variant; the tree must now expand THAT
    # variant's sub-tree, not the previous one.
    _set(root, "modules.readout.ref", "Direct Readout")
    readout = _node(build_cfg_observation(root), "modules.readout")
    assert readout["ref"] == "<Custom:Direct Readout>"
    assert "pulse_cfg" not in readout["children"]


def test_tree_includes_immutable_literal_fields(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )

    root = _fakefreq_root()
    pulse_cfg = _node(build_cfg_observation(root), "modules.readout.pulse_cfg")
    for name in ("type", "freq"):
        leaf = pulse_cfg["children"][name]
        assert leaf["kind"] == "literal"
        assert not leaf["editable"]
        assert leaf["input"]["resolved"] is not None


def test_tree_device_scalar_has_value_and_dynamic_choices(qapp):
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )

    root = _fluxdep_root(["flux_yoko", "flux_yoko_2"])
    flux_dev = _node(build_cfg_observation(root, "dev"), "flux_dev")
    assert flux_dev["input"]["resolved"] == "flux_yoko"
    assert flux_dev["choices"] == ["flux_yoko", "flux_yoko_2"]


# ---------------------------------------------------------------------------
# ModuleRef key normalization — a bare variant label (as list_paths advertises
# in 'choices') is accepted and stored as the tagged <Custom:label> chosen_key,
# not mistaken for a (non-existent) library entry name. Regression: a bare label
# used to be stored verbatim → empty sub-field → lowering "Unknown module
# reference". Pure binding-target unit (no socket): fake/freq has a readout
# ModuleRef with variant labels "Direct Readout" / "Pulse Readout".
# ---------------------------------------------------------------------------


def _fakefreq_root():
    from zcu_tools.experiment.v2_gui.measure.adapters.fake.freq import FakeFreqAdapter
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    cfg = FakeFreqAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.get_current_md.return_value = ctx.md
    return _make_draft(ctrl, cfg.spec, cfg.value)


def test_unknown_field_suggests_matching_descendant_path(qapp):
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _fakefreq_root()
    with pytest.raises(SettablePathError) as exc:
        _set(root, "modules.readout.gain", 0.25)

    assert "modules.readout.pulse_cfg.gain" in str(exc.value)


def test_moduleref_bare_label_normalized_to_custom_tag(qapp):
    from zcu_tools.gui.app.measure.remote.path_resolver import (
        project_target_entries,
    )

    root = _fakefreq_root()
    # Bare label, exactly as tab.get_cfg advertises in 'choices'.
    _set(root, "modules.readout.ref", "Direct Readout")

    entry = next(
        e for e in project_target_entries(root) if e["path"] == "modules.readout.ref"
    )
    # Stored as the tagged key, not the bare label.
    assert entry["value"] == "<Custom:Direct Readout>"


def test_moduleref_tagged_key_passes_through(qapp):
    from zcu_tools.gui.app.measure.remote.path_resolver import (
        project_target_entries,
    )

    root = _fakefreq_root()
    # An already-tagged key is stored verbatim (no double-wrapping).
    _set(root, "modules.readout.ref", "<Custom:Direct Readout>")

    entry = next(
        e for e in project_target_entries(root) if e["path"] == "modules.readout.ref"
    )
    assert entry["value"] == "<Custom:Direct Readout>"


# ---------------------------------------------------------------------------
# Device selectors are ordinary required dynamic scalar fields. Their wire path
# is the scalar leaf itself; there is no device-ref alias segment.
# ---------------------------------------------------------------------------


def _fluxdep_root(device_names: list[str]):
    from zcu_tools.experiment.v2_gui.measure.adapters.onetone.flux_dep import (
        OneToneFluxDepAdapter,
    )
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.resources.context import MetaDict, ModuleLibrary

    ctx = SessionEnv(md=MetaDict(), ml=ModuleLibrary(), soc=None, soccfg=None)
    cfg = OneToneFluxDepAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.get_current_md.return_value = ctx.md
    ctrl.list_device_names.return_value = list(device_names)
    return _make_draft(ctrl, cfg.spec, cfg.value)


def _device_value(root, path: str = "dev.flux_dev"):
    from zcu_tools.gui.app.measure.remote.path_resolver import (
        project_target_entries,
    )

    return next(e for e in project_target_entries(root) if e["path"] == path)["value"]


def test_device_selector_advertises_scalar_leaf_path(qapp):
    from zcu_tools.gui.app.measure.remote.path_resolver import (
        project_target_entries,
    )

    root = _fluxdep_root(["flux_yoko"])
    entry = next(e for e in project_target_entries(root) if e["path"] == "dev.flux_dev")
    assert entry["kind"] == "scalar"
    assert entry["choices"] == ["flux_yoko"]


def test_device_selector_scalar_path_resolves(qapp):
    root = _fluxdep_root(["flux_yoko", "flux_yoko_2"])
    _set(root, "dev.flux_dev", "flux_yoko_2")
    assert _device_value(root) == "flux_yoko_2"


def test_device_selector_non_string_value_rejected(qapp):
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _fluxdep_root(["flux_yoko"])
    with pytest.raises(SettablePathError):
        _set(root, "dev.flux_dev", 42)


def test_device_selector_has_no_legacy_alias_segment(qapp):
    from zcu_tools.gui.cfg.binding import SettablePathError

    root = _fluxdep_root(["flux_yoko"])
    with pytest.raises(SettablePathError):
        _set(root, "dev.flux_dev.device", "flux_yoko")
