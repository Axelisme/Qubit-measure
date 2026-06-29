"""autofluxdep-gui workflow persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from zcu_tools.gui.app.autofluxdep.app import build_core
from zcu_tools.gui.app.autofluxdep.services import (
    APP_STATE_VERSION,
    AppPersistedState,
    PersistedFluxSweep,
    PersistedNode,
    PersistedUiPrefs,
    PersistedWorkflow,
    PersistenceCaretaker,
    RestoreReport,
)
from zcu_tools.gui.app.autofluxdep.ui.main_window import MainWindow


def _list_labels(win: MainWindow) -> list[str]:
    items = [win._list._list.item(i) for i in range(win._list._list.count())]
    return [item.text() for item in items if item is not None]


def test_workflow_persistence_roundtrip(tmp_path: Path):
    ctrl = build_core()
    ctrl.add_node_by_type("qubit_freq")
    ctrl.rename_node(0, "freq_scan")
    ctrl.set_node_params(0, {"qub_gain": "0.2"})
    ctrl.add_node_by_type("lenrabi")
    ctrl.set_flux_sweep_expressions("span / 2", "-span / 2", "2 * count + 1")
    ctrl.set_flux_values([0.002, 0.001, 0.0, -0.001, -0.002])
    ctrl.set_auto_follow_tabs(False)
    ctrl.attach_caretaker(PersistenceCaretaker(ctrl, cache_dir=tmp_path))
    ctrl.persist_all()

    restored = build_core()
    restored.attach_caretaker(PersistenceCaretaker(restored, cache_dir=tmp_path))
    outcome = restored.restore_all()

    assert outcome is not None
    assert outcome.load_error is None
    assert isinstance(outcome.report, RestoreReport)
    assert outcome.report.rejected_nodes == ()
    assert restored.state.node_names() == ["freq_scan", "lenrabi"]
    assert restored.state.nodes[0].type_name == "qubit_freq"
    assert restored.state.nodes[0].schema.read_knobs()["qub_gain"] == pytest.approx(0.2)
    assert restored.get_flux_sweep_expressions() == (
        "span / 2",
        "-span / 2",
        "2 * count + 1",
    )
    assert restored.state.flux_values == pytest.approx(
        [0.002, 0.001, 0.0, -0.001, -0.002]
    )
    assert restored.get_auto_follow_tabs() is False


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
                    nodes=(PersistedNode(type_name="qubit_freq", name="freq_scan"),)
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
    finally:
        ctrl._background_svc.quiesce()
        win.close()
        win.deleteLater()
