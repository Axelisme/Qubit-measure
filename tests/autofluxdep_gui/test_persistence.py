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
    PersistedPredictorDialogState,
    PersistedPredictorModel,
    PersistedStartup,
    PersistedUiPrefs,
    PersistedWorkflow,
    PersistenceError,
    RestoreReport,
)
from zcu_tools.gui.app.autofluxdep.services import (
    create_persistence_caretaker as PersistenceCaretaker,
)
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow
from zcu_tools.gui.session.services.predictor import SetModelParamsRequest
from zcu_tools.gui.session.services.startup import (
    StartupConnectionRequest,
    StartupProjectRequest,
)

from ._helpers import set_node_cfg_knobs


def _list_labels(win: MainWindow) -> list[str]:
    labels: list[str] = []
    for row in range(win._list._list.count()):
        item = win._list._list.item(row)
        assert item is not None
        widget = win._list._list.itemWidget(item)
        assert widget is not None
        labels.append(cast(Any, widget)._label.text())
    return labels


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
    set_node_cfg_knobs(
        ctrl,
        0,
        {
            "qub_gain": "0.2",
            "drive_gain_mode": "fixed",
            "earlystop_snr": "12.5",
            "acquire_retry": "2",
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
    assert "physical_recovery_min_points" not in generation_raw
    assert "physical_recovery_max_rms_mhz" not in generation_raw
    assert generation_raw["earlystop_snr"] == {"__kind": "direct", "value": 12.5}
    assert generation_raw["acquire_retry"] == {"__kind": "direct", "value": 2}
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
    assert "physical_recovery_min_points" not in knobs
    assert "physical_recovery_max_rms_mhz" not in knobs
    assert knobs["earlystop_snr"] == pytest.approx(12.5)
    assert knobs["acquire_retry"] == 2
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


def test_predictor_model_persistence_roundtrip(tmp_path: Path):
    ctrl = build_core()
    req = SetModelParamsRequest(
        EJ=4.2,
        EC=1.1,
        EL=0.7,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.05,
    )
    ctrl.predictor_control.set_predictor_model_params(req)
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)
    ctrl.persist_all()

    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["predictor"]["EJ"] == pytest.approx(4.2)
    assert payload["predictor"]["EC"] == pytest.approx(1.1)
    assert payload["predictor"]["EL"] == pytest.approx(0.7)
    assert payload["predictor"]["flux_half"] == pytest.approx(0.3)
    assert payload["predictor"]["flux_period"] == pytest.approx(0.8)
    assert payload["predictor"]["flux_bias"] == pytest.approx(0.05)

    restored = build_core()
    restored.attach_caretaker(PersistenceCaretaker(restored, cache_dir=tmp_path))
    outcome = restored.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert isinstance(outcome.report, RestoreReport)
    assert outcome.report.restored_predictor is True
    info = restored.predictor_control.get_predictor_info()
    assert info is not None
    assert info["EJ"] == pytest.approx(req.EJ)
    assert info["EC"] == pytest.approx(req.EC)
    assert info["EL"] == pytest.approx(req.EL)
    assert info["flux_half"] == pytest.approx(req.flux_half)
    assert info["flux_period"] == pytest.approx(req.flux_period)
    assert info["flux_bias"] == pytest.approx(req.flux_bias)


def test_predictor_model_mutation_autosaves_after_debounce(tmp_path: Path, qapp):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)
    req = SetModelParamsRequest(
        EJ=4.2,
        EC=1.1,
        EL=0.7,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.05,
    )

    assert not caretaker.state_path.exists()
    ctrl.predictor_control.set_predictor_model_params(req)

    assert ctrl._persist_timer.isActive()
    _pump_for(qapp, 0.7)
    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["predictor"]["EJ"] == pytest.approx(req.EJ)
    assert payload["predictor"]["EC"] == pytest.approx(req.EC)
    assert payload["predictor"]["EL"] == pytest.approx(req.EL)
    assert payload["predictor"]["flux_half"] == pytest.approx(req.flux_half)
    assert payload["predictor"]["flux_period"] == pytest.approx(req.flux_period)
    assert payload["predictor"]["flux_bias"] == pytest.approx(req.flux_bias)


def test_predictor_clear_autosaves_null_after_debounce(tmp_path: Path, qapp):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)
    req = SetModelParamsRequest(
        EJ=4.2,
        EC=1.1,
        EL=0.7,
        flux_half=0.3,
        flux_period=0.8,
        flux_bias=0.05,
    )
    ctrl.predictor_control.set_predictor_model_params(req)
    _pump_for(qapp, 0.7)
    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["predictor"] is not None

    ctrl.predictor_control.clear_predictor()

    assert ctrl._persist_timer.isActive()
    _pump_for(qapp, 0.7)
    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["predictor"] is None


def test_restore_invalid_predictor_reports_issue_without_rejecting_workflow():
    ctrl = build_core()
    state = AppPersistedState(
        predictor=PersistedPredictorModel(
            EJ=4.2,
            EC=1.1,
            EL=0.7,
            flux_half=0.3,
            flux_period=0.0,
        ),
        workflow=PersistedWorkflow(
            nodes=(PersistedNode(type_name="qubit_freq", name="freq_scan"),)
        ),
    )

    report = ctrl.restore_persisted_state(state)

    assert report.restored_nodes == 1
    assert report.rejected_nodes == ()
    assert report.restored_predictor is False
    assert report.predictor_issue is not None
    assert report.predictor_issue.subject == "predictor"
    assert ctrl.state.node_names() == ["freq_scan"]
    assert ctrl.predictor_control.get_predictor_info() is None


def test_restore_without_predictor_defaults_none_and_default_dialog_state(
    tmp_path: Path,
):
    ctrl = build_core()
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    caretaker.state_path.parent.mkdir(parents=True, exist_ok=True)
    caretaker.state_path.write_text(
        json.dumps(
            {
                "version": APP_STATE_VERSION,
                "workflow": {"nodes": []},
                "ui": {"auto_follow_tabs": False},
            }
        ),
        encoding="utf-8",
    )
    ctrl.attach_caretaker(caretaker)

    outcome = ctrl.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert ctrl.predictor_control.get_predictor_info() is None
    assert ctrl.get_predictor_dialog_state() == PersistedPredictorDialogState()
    assert ctrl.get_auto_follow_tabs() is False


def test_wrong_version_restores_default_with_load_error(tmp_path: Path):
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    caretaker.state_path.write_text(
        json.dumps({"version": APP_STATE_VERSION + 1}), encoding="utf-8"
    )
    ctrl.attach_caretaker(caretaker)

    outcome = ctrl.restore_all()

    assert outcome is not None
    assert isinstance(outcome.load_error, PersistenceError)
    assert "Unsupported autofluxdep GUI state version" in str(outcome.load_error)
    assert ctrl.state.node_names() == []


def test_predictor_dialog_state_persistence_roundtrip(tmp_path: Path):
    dialog_state = PersistedPredictorDialogState(
        tracked_transitions=((2, 3), (3, 5)),
        tab_index=2,
        params_path_text="/saved/params.json",
    )
    ctrl = build_core()
    ctrl.set_predictor_dialog_state(dialog_state)
    caretaker = PersistenceCaretaker(ctrl, cache_dir=tmp_path)
    ctrl.attach_caretaker(caretaker)
    ctrl.persist_all()

    payload = json.loads(caretaker.state_path.read_text(encoding="utf-8"))
    assert payload["ui"]["predictor_dialog"]["tracked_transitions"] == [
        [2, 3],
        [3, 5],
    ]
    assert payload["ui"]["predictor_dialog"]["tab_index"] == 2
    assert payload["ui"]["predictor_dialog"]["params_path_text"] == "/saved/params.json"

    restored = build_core()
    restored.attach_caretaker(PersistenceCaretaker(restored, cache_dir=tmp_path))
    outcome = restored.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert restored.get_predictor_dialog_state() == dialog_state


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


def test_restore_rejects_node_with_removed_generation_key():
    ctrl = build_core()
    node = ctrl.add_node_by_type("qubit_freq")
    cfg_raw = node.schema.to_persisted_raw()
    generation = cfg_raw["generation"]
    assert isinstance(generation, dict)
    generation["physical_recovery_min_points"] = {"__kind": "direct", "value": 12}
    state = AppPersistedState(
        workflow=PersistedWorkflow(
            nodes=(
                PersistedNode(
                    type_name="qubit_freq",
                    name="freq_scan",
                    cfg_raw=cfg_raw,
                ),
            )
        )
    )

    report = ctrl.restore_persisted_state(state)

    assert report.restored_nodes == 0
    assert len(report.rejected_nodes) == 1
    assert "physical_recovery_min_points" in report.rejected_nodes[0].message
    assert ctrl.state.node_names() == []


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
        assert win._list._flux_start.expression_text() == "phi0 - span"
        assert win._list._flux_stop.expression_text() == "phi0 + span"
        assert win._list._flux_npts.expression_text() == "n_flux"
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

        def restore_persisted_state(self, state: AppPersistedState) -> RestoreReport:
            return RestoreReport(restored_nodes=len(state.workflow.nodes))

    caretaker = PersistenceCaretaker(BadOriginator(), cache_dir=tmp_path)

    with pytest.raises(PersistenceError, match="Failed to save GUI state"):
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
