"""Phase 81b — RemoteControlService cfg.set_field + context/device queries.

``cfg.set_field`` mutates the tab's *live* ``SectionLiveField`` (so the form
stays WYSIWYG and auto-commits). In these tests the mock View returns a real
``SectionLiveField`` built from the FakeAdapter default cfg, and we assert the
mutation lands on that live tree. Path-resolver edge cases (sweep edges,
literal rejection, unknown paths) each get a focused case.
"""

from __future__ import annotations

import pytest
from zcu_tools.gui.adapter import DirectValue, EvalValue
from zcu_tools.gui.live_model import (
    LiveModelEnv,
    ScalarLiveField,
    SectionLiveField,
    SweepLiveField,
)

from ._helpers import Fixture, call, open_client

# ---------------------------------------------------------------------------
# Fixture: wire a real live model behind the mock View
# ---------------------------------------------------------------------------


class _LiveFixture(Fixture):
    """Fixture whose View serves a real SectionLiveField for one tab."""

    def __init__(self) -> None:
        super().__init__()
        # Build a live model from the FakeAdapter default cfg, using the real
        # Controller as the LiveModelEnv controller (it implements the
        # ControllerProtocol surface LiveFields need).
        schema = self.ctrl.get_adapter_names()  # touch to ensure registry ready
        del schema
        from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter

        cfg = FakeAdapter().make_default_cfg(self.state.exp_context)
        env = LiveModelEnv(ctrl=self.ctrl)
        self.root = SectionLiveField(cfg.spec, env, initial_val=cfg.value)
        self.view.get_tab_live_model_root = lambda tab_id: self.root
        # cfg.set_field also checks has_tab; register a dummy tab id.
        self._tab_id = "tab-live"
        # Inject a minimal TabState so has_tab(tab-live) is True.
        from zcu_tools.gui.state import TabState

        self.state.add_tab(
            self._tab_id,
            TabState(
                adapter_name="fake",
                adapter=FakeAdapter(),
                cfg_schema=cfg,
            ),
        )


@pytest.fixture()
def lf(qapp):  # noqa: ARG001
    f = _LiveFixture()
    f.start()
    yield f
    f.stop()


def _field(root: SectionLiveField, name: str):
    return root.fields[name]


# ---------------------------------------------------------------------------
# Scalar
# ---------------------------------------------------------------------------


def test_set_field_scalar_updates_live_model(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "reps", "value": 42},
        )
        assert resp["ok"] is True
        reps = _field(lf.root, "reps")
        assert isinstance(reps, ScalarLiveField)
        val = reps.get_value()
        assert isinstance(val, DirectValue)
        assert val.value == 42
    finally:
        sock.close()


def test_set_field_scalar_eval_expression(lf):
    sock = open_client(lf.service.port)
    try:
        # ScalarLiveField.set_value accepts EvalValue; the resolver passes the
        # raw value straight through, so a plain number is the common path.
        # Here we confirm a float lands too.
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "gain", "value": 0.25},
        )
        assert resp["ok"] is True
        gain = _field(lf.root, "gain")
        assert isinstance(gain, ScalarLiveField)
        val = gain.get_value()
        assert isinstance(val, (DirectValue, EvalValue))
        assert isinstance(val, DirectValue) and val.value == 0.25
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Sweep edges
# ---------------------------------------------------------------------------


def test_set_field_sweep_expts(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep.expts", "value": 5},
        )
        assert resp["ok"] is True
        sweep = _field(lf.root, "sweep")
        assert isinstance(sweep, SweepLiveField)
        assert sweep.get_value().expts == 5
    finally:
        sock.close()


def test_set_field_sweep_start_stop(lf):
    sock = open_client(lf.service.port)
    try:
        call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep.start", "value": 2.0},
            rid="a",
        )
        call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep.stop", "value": 8.0},
            rid="b",
        )
        sweep = _field(lf.root, "sweep")
        assert isinstance(sweep, SweepLiveField)
        sv = sweep.get_value()
        assert sv.start == 2.0
        assert sv.stop == 8.0
    finally:
        sock.close()


def test_set_field_sweep_step(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep.step", "value": 0.5},
        )
        assert resp["ok"] is True
    finally:
        sock.close()


def test_set_field_sweep_expts_non_integer_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep.expts", "value": 3.5},
        )
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
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "does_not_exist", "value": 1},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_set_field_unknown_tab_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": "ghost", "path": "reps", "value": 1},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_set_field_missing_value_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "reps"},
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_set_field_section_target_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        # 'sweep' alone targets a sweep container, not a leaf.
        resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": "sweep", "value": 1},
        )
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
        resp = call(sock, "context.get_md")
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
        resp = call(sock, "context.get_md_attr", {"key": "t1"})
        assert resp["ok"] is True
        assert resp["result"]["value"] == 12.5
    finally:
        sock.close()


def test_context_get_md_attr_unknown_rejected(lf):
    md = lf.state.exp_context.md
    md.get = lambda key, default=None: default
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.get_md_attr", {"key": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_context_get_ml_names(lf):
    ml = lf.state.exp_context.ml
    ml.modules = {"readout": object(), "pi": object()}
    ml.waveforms = {"gauss": object()}
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "context.get_ml")
        assert resp["ok"] is True
        assert set(resp["result"]["modules"]) == {"readout", "pi"}
        assert resp["result"]["waveforms"] == ["gauss"]
    finally:
        sock.close()


def test_device_list_and_snapshot(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "device.list")
        assert resp["ok"] is True
        assert isinstance(resp["result"]["devices"], list)

        resp = call(sock, "device.snapshot", {"name": "does-not-exist"})
        assert resp["ok"] is True
        assert resp["result"]["snapshot"] is None
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# tab.list_paths
# ---------------------------------------------------------------------------


def test_list_paths_enumerates_settable_leaves(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.list_paths", {"tab_id": lf._tab_id})
        assert resp["ok"] is True
        paths = resp["result"]["paths"]
        by_path = {p["path"]: p for p in paths}

        # Scalar leaf present with value + type.
        assert "reps" in by_path
        assert by_path["reps"]["kind"] == "scalar"
        assert by_path["reps"]["type"] in ("int", "float")

        # A sweep edge is exposed as <path>.expts (integer) etc.
        sweep_edges = [p for p in paths if p["kind"] == "sweep_edge"]
        assert sweep_edges, "expected at least one sweep edge"
        assert any(p["path"].endswith(".expts") for p in sweep_edges)

        # Every listed path is non-empty and dotted-or-plain.
        assert all(p["path"] for p in paths)
    finally:
        sock.close()


def test_list_paths_round_trips_through_set_field(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.list_paths", {"tab_id": lf._tab_id})
        scalar = next(
            p
            for p in resp["result"]["paths"]
            if p["kind"] == "scalar" and p["type"] in ("int", "float")
        )
        new_value = 7 if scalar["type"] == "int" else 0.5
        set_resp = call(
            sock,
            "cfg.set_field",
            {"tab_id": lf._tab_id, "path": scalar["path"], "value": new_value},
        )
        assert set_resp["ok"] is True, scalar["path"]
    finally:
        sock.close()


def test_list_paths_unknown_tab_rejected(lf):
    sock = open_client(lf.service.port)
    try:
        resp = call(sock, "tab.list_paths", {"tab_id": "nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()
