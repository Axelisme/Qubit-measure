"""CfgDraft path resolution and tree projection, and tab Form observations.

Draft-level tests drive the old CfgDraft projection directly. Tab reads use
the Form resource over the socket.
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
    return MeasureCfgBindings(ctrl).new_draft(CfgSchema(spec, value), parameters=None)


def _set(draft: CfgDraft, path: str, value: object) -> None:
    draft.set_target(path, value)


# ---------------------------------------------------------------------------
# Fixture: one live tab behind the socket
# ---------------------------------------------------------------------------


class _LiveFixture(Fixture):
    """Fixture with one synthetic tab."""

    def __init__(self) -> None:
        super().__init__()
        self._tab_id = self.ctrl.new_tab("fake")


@pytest.fixture()
def lf(qapp):
    f = _LiveFixture()
    f.start()
    yield f
    f.stop()


# ---------------------------------------------------------------------------
# Centered sweep edges
# ---------------------------------------------------------------------------


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
# Context / device queries
# ---------------------------------------------------------------------------


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


def test_tab_get_cfg_returns_form_scalar_values(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})
        assert resp["ok"] is True
        observed = resp["result"]
        assert observed["input"]["reps"] == 100
        leaves = {tuple(item["path"]): item["value"] for item in observed["values"]}
        assert leaves[("reps",)] == 100
        assert observed["schema"]["properties"]["reps"]
        assert observed["errors"] == []
    finally:
        sock.close()


def test_tab_get_cfg_range_has_authored_model_fields(lf):
    sock = open_client(lf.service.port)
    try:
        observed = call(sock, "tab.get_cfg", {"tab_id": lf._tab_id})["result"]
        assert observed["input"]["sweep"] == {
            "kind": "linrange",
            "start": 5.0,
            "stop": 6.0,
            "n": 11,
        }
        leaves = {tuple(item["path"]): item["value"] for item in observed["values"]}
        assert leaves[("sweep", "n")] == 11
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
    from zcu_tools.gui.app.measure.adapter import SessionEnv
    from zcu_tools.gui.app.measure.remote.cfg_observation import (
        build_cfg_observation,
    )
    from zcu_tools.gui.cfg.model import DirectValue

    from tests.gui.app.measure._flux_adapter import SpectrumAdapter

    cfg = SpectrumAdapter().make_default_cfg(
        SessionEnv(parameters=None, soc=None, soccfg=None)
    )
    devices = cfg.value.fields["dev"]
    assert isinstance(devices, CfgSectionValue)
    devices.fields["flux_dev"] = DirectValue("flux_yoko")
    ctrl = MagicMock()
    ctrl.list_device_names.return_value = ["flux_yoko", "flux_yoko_2"]
    root = _make_draft(ctrl, cfg.spec, cfg.value)
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
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    from tests.gui.app.measure._synthetic import PairAdapter

    ctx = SessionEnv(parameters=None, soc=None, soccfg=None)
    cfg = PairAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
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
    from zcu_tools.gui.app.measure.adapter import SessionEnv

    from tests.gui.app.measure._flux_adapter import SpectrumAdapter

    ctx = SessionEnv(parameters=None, soc=None, soccfg=None)
    cfg = SpectrumAdapter().make_default_cfg(ctx)
    ctrl = MagicMock()
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
