"""autofluxdep-gui workflow persistence."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, cast

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.services import (
    APP_STATE_VERSION,
    AppPersistedState,
    PersistedFluxSweep,
    PersistedNode,
    PersistedStartup,
    PersistedUiPrefs,
    PersistedWorkflow,
    PersistenceCaretaker,
    PersistenceError,
    RestoreReport,
)
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow
from zcu_tools.gui.session.services.startup import (
    StartupConnectionRequest,
    StartupProjectRequest,
)


def _list_labels(win: MainWindow) -> list[str]:
    items = [win._list._list.item(i) for i in range(win._list._list.count())]
    return [item.text() for item in items if item is not None]


def _pump_for(qapp, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    qapp.processEvents()


def test_workflow_persistence_roundtrip(tmp_path: Path):
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.rename_node(0, "freq_scan")
    ctrl.set_node_params(
        0,
        {
            "qub_gain": "0.2",
            "drive_gain_mode": "fixed",
            "earlystop_snr": "12.5",
        },
    )
    ctrl.add_node_by_type("lenrabi")
    ctrl.set_node_enabled(1, False)
    ctrl.set_flux_sweep_expressions("span / 2", "-span / 2", "2 * count + 1")
    ctrl.set_flux_values([0.002, 0.001, 0.0, -0.001, -0.002])
    ctrl.set_auto_follow_tabs(False)
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)
    ctrl.persist_all()

    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["workflow"]["nodes"][0]["enabled"] is True
    assert payload["workflow"]["nodes"][1]["enabled"] is False
    generation_raw = payload["workflow"]["nodes"][0]["cfg_raw"]["generation"]
    assert generation_raw["drive_gain_mode"] == {
        "__kind": "direct",
        "value": "fixed",
    }
    assert generation_raw["earlystop_snr"] == {"__kind": "direct", "value": 12.5}
    assert "feedback" not in generation_raw
    assert "safety" not in generation_raw

    restored = build_core()
    restored.attach_caretaker(PersistenceCaretaker(restored, cache_dir=tmp_path))
    outcome = restored.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert isinstance(outcome.report, RestoreReport)
    assert outcome.report.rejected_nodes == ()
    assert restored.state.node_names() == ["freq_scan", "lenrabi"]
    assert restored.state.nodes[0].type_name == "qubit_freq"
    assert restored.state.nodes[0].enabled is True
    assert restored.state.nodes[1].enabled is False
    knobs = restored.state.nodes[0].schema.read_knobs()
    assert knobs["qub_gain"] == pytest.approx(0.2)
    assert knobs["drive_gain_mode"] == "fixed"
    assert knobs["earlystop_snr"] == pytest.approx(12.5)
    assert restored.get_flux_sweep_expressions() == (
        "span / 2",
        "-span / 2",
        "2 * count + 1",
    )
    assert restored.state.flux_values == pytest.approx(
        [0.002, 0.001, 0.0, -0.001, -0.002]
    )
    assert restored.get_auto_follow_tabs() is False


def test_restore_old_node_without_enabled_defaults_true(tmp_path: Path):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    caretaker.state_path.parent.mkdir(parents=True, exist_ok=True)
    caretaker.state_path.write_text(
        json.dumps(
            {
                "version": APP_STATE_VERSION,
                "workflow": {
                    "nodes": [{"type_name": "qubit_freq", "name": "freq_scan"}]
                },
            }
        ),
        encoding="utf-8",
    )
    ctrl.attach_caretaker(caretaker)

    outcome = ctrl.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert ctrl.state.node_names() == ["freq_scan"]
    assert ctrl.state.nodes[0].enabled is True


def test_startup_memento_persistence_roundtrip(tmp_path: Path):
    ctrl = build_core(project_root=str(tmp_path))
    ctrl.apply_startup_project(StartupProjectRequest("chip", "qub", "res"))
    ctrl.remember_startup_connection(StartupConnectionRequest(ip="10.0.0.2", port=7000))
    scope_id = ctrl.get_persisted_startup().scope_id
    ctrl.attach_caretaker(PersistenceCaretaker(ctrl, cache_dir=tmp_path))
    ctrl.persist_all()

    restored = build_core(project_root=str(tmp_path))
    restored.attach_caretaker(PersistenceCaretaker(restored, cache_dir=tmp_path))
    outcome = restored.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    startup = restored.get_persisted_startup()
    assert startup.scope_id == scope_id
    assert startup.ip == "10.0.0.2"
    assert startup.port == 7000
    assert restored.state.project is None
    assert restored.state.exp_context.soc is None
    assert restored.state.exp_context.soccfg is None


def test_restore_old_memento_without_ui_defaults_auto_follow_true(tmp_path: Path):
    ctrl = build_core()
    ctrl.set_auto_follow_tabs(False)
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    caretaker.state_path.parent.mkdir(parents=True, exist_ok=True)
    caretaker.state_path.write_text(
        json.dumps(
            {
                "version": APP_STATE_VERSION,
                "workflow": {"nodes": []},
                "flux": {
                    "start_expr": "0.0",
                    "stop_expr": "1.0",
                    "npts_expr": "3",
                    "values": [0.0, 0.5, 1.0],
                },
            }
        ),
        encoding="utf-8",
    )
    ctrl.attach_caretaker(caretaker)

    outcome = ctrl.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert ctrl.get_auto_follow_tabs() is True
    assert ctrl.get_persisted_startup() == PersistedStartup()
    assert ctrl.state.flux_values == pytest.approx([0.0, 0.5, 1.0])


def test_restore_rejects_invalid_node_and_keeps_valid_nodes():
    ctrl = build_core()
    state = AppPersistedState(
        workflow=PersistedWorkflow(
            nodes=(
                PersistedNode(type_name="missing_node", name="bad"),
                PersistedNode(type_name="qubit_freq", name="freq_scan"),
            )
        )
    )

    report = ctrl.restore_persisted_state(state)

    assert report.restored_nodes == 1
    assert len(report.rejected_nodes) == 1
    assert "bad" in report.rejected_nodes[0].subject
    assert ctrl.state.node_names() == ["freq_scan"]


def test_window_restore_workflow_view_updates_list_and_flux_fields(qapp):  # noqa: ARG001
    ctrl = build_core()
    win = MainWindow(ctrl)
    try:
        ctrl.restore_persisted_state(
            AppPersistedState(
                workflow=PersistedWorkflow(
                    nodes=(
                        PersistedNode(
                            type_name="qubit_freq",
                            name="freq_scan",
                            enabled=False,
                        ),
                    )
                ),
                flux=PersistedFluxSweep(
                    start_expr="phi0 - span",
                    stop_expr="phi0 + span",
                    npts_expr="n_flux",
                    values=(0.1, 0.2),
                ),
                ui=PersistedUiPrefs(auto_follow_tabs=False),
            )
        )
        win.restore_workflow_view()

        assert _list_labels(win) == ["freq_scan"]
        assert win._list._flux_start.text() == "phi0 - span"
        assert win._list._flux_stop.text() == "phi0 + span"
        assert win._list._flux_npts.text() == "n_flux"
        assert not win._list._auto_follow_tabs.isChecked()
        item = win._list._list.item(0)
        assert item is not None
        row = win._list._list.itemWidget(item)
        assert row is not None
        assert not cast(Any, row)._checkbox.isChecked()
    finally:
        ctrl._background_svc.quiesce()
        win.close()
        win.deleteLater()


def test_autosave_debounce_writes_workflow_flux_and_ui_prefs(tmp_path: Path, qapp):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)

    ctrl.add_node_by_type("qubit_freq")
    ctrl.set_node_enabled(0, False)
    ctrl.set_flux_sweep_expressions("0.0", "1.0", "5")
    ctrl.set_auto_follow_tabs(False)

    assert not caretaker.state_path.exists()
    _pump_for(qapp, 0.7)

    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["workflow"]["nodes"][0]["enabled"] is False
    assert payload["flux"]["start_expr"] == "0.0"
    assert payload["flux"]["stop_expr"] == "1.0"
    assert payload["flux"]["npts_expr"] == "5"
    assert payload["ui"]["auto_follow_tabs"] is False


def test_caretaker_wraps_capture_failure(tmp_path: Path):
    class BadOriginator:
        def capture_persisted_state(self) -> AppPersistedState:
            raise RuntimeError("bad capture")

        def restore_persisted_state(self, state: AppPersistedState) -> object:
            return state

    caretaker = PersistenceCaretaker(BadOriginator(), cache_dir=tmp_path)

    with pytest.raises(PersistenceError, match="Failed to save autofluxdep GUI state"):
        caretaker.flush()

    assert not caretaker.state_path.exists()


def test_persist_all_swallows_persistence_error(tmp_path: Path, monkeypatch):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)

    def raise_persistence_error() -> None:
        raise PersistenceError("bad persist")

    monkeypatch.setattr(caretaker, "flush", raise_persistence_error)

    ctrl.persist_all()


def test_persist_all_flushes_immediately_and_cancels_pending_timer(
    tmp_path: Path, qapp
):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)

    ctrl.add_node_by_type("qubit_freq")
    assert ctrl._persist_timer.isActive()

    ctrl.persist_all()

    assert not ctrl._persist_timer.isActive()
    assert caretaker.state_path.exists()
    _pump_for(qapp, 0.7)
    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["workflow"]["nodes"][0]["name"] == "qubit_freq"
