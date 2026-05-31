"""RemoteControlService tab-cfg editing via the CfgEditorService session.

A tab's cfg draft is a service-owned ``CfgEditorSession`` keyed by the tab_id
(the same draft the open form attaches to). Agents edit it with
``editor.set_field`` on the tab's ``editor_id`` (from ``tab.snapshot``) — the
same path the GUI form uses, so user + agent share one model (ADR-0013 F11).

Here the fixture opens a real seeded session owned by the tab on the real
Controller, then drives edits through ``editor.set_field`` and discovery
through ``tab.list_paths`` (which reads that same session). Path-resolver edge
cases (sweep edges, literal rejection, unknown paths) each get a focused case.
"""

from __future__ import annotations

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
        from zcu_tools.gui.state import TabState

        cfg = FakeAdapter().make_default_cfg(self.state.exp_context)
        self._tab_id = "tab-live"
        # Inject a TabState so has_tab(tab-live) is True.
        self.state.add_tab(
            self._tab_id,
            TabState(
                adapter_name="fake",
                adapter=FakeAdapter(),
                cfg_schema=cfg,
            ),
        )
        # Open the tab's cfg-editor session keyed by tab_id — exactly what
        # MainWindow.populate_cfg does. editor_id_for_owner(tab_id) now resolves
        # it, so tab.list_paths reads it and editor.set_field mutates it.
        self.editor_id, _ = self.ctrl.open_seeded_cfg_editor(
            cfg, gc=False, owner_key=self._tab_id
        )

    def get_value(self, path: str):
        """Read the current value of a path off the live session draft."""
        from zcu_tools.gui.services.remote.path_resolver import list_settable_paths

        root = self.ctrl.get_cfg_editor_root(self.editor_id)
        for entry in list_settable_paths(root):
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
# tab.list_paths — now reads the tab's cfg-editor session (ADR-0013 F11)
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
        set_resp = _set_field(sock, lf, scalar["path"], new_value)
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


def test_list_paths_form_not_populated_rejected(qapp):  # noqa: ARG001
    """A tab with no cfg-editor session yet → precondition_failed."""
    from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
    from zcu_tools.gui.state import TabState

    f = Fixture()
    f.start()
    try:
        cfg = FakeAdapter().make_default_cfg(f.state.exp_context)
        f.state.add_tab(
            "bare",
            TabState(adapter_name="fake", adapter=FakeAdapter(), cfg_schema=cfg),
        )
        sock = open_client(f.service.port)
        try:
            resp = call(sock, "tab.list_paths", {"tab_id": "bare"})
            assert resp["ok"] is False
            assert resp["error"]["code"] == "precondition_failed"
        finally:
            sock.close()
    finally:
        f.stop()
