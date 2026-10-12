"""Workflow input mementos through public Window and app lifecycle seams."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormInput
from zcu_tools.gui.session.persistence import SingleFileCaretaker

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab import ALIAS_PRESET
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.gui.coherence import CoherenceTunablesForm
from zcu_lab.workflows.gui.plan import CoherencePlanForm
from zcu_lab.workflows.registry import build_registry

from ._coherence_input import t1_input


@pytest.fixture
def window(qapp: QApplication) -> Iterator[WorkflowWindow[Calibrated]]:
    harness = StartHarness()
    view = WorkflowWindow(
        harness.state, harness.services, harness.owner, registrations=build_registry()
    )
    try:
        yield view
    finally:
        view.close()
        view.deleteLater()
        qapp.processEvents()


def test_original_expressions_roundtrip_and_restore_base(
    window: WorkflowWindow[Calibrated], tmp_path: Path
) -> None:
    window.plan_panel.state.write(
        {("pi_length_us",): {"expr": "0.1 + 0.2"}}, window.plan_panel.env
    )
    window.tunables_panel.state.write(
        {("t1", "guess_us"): {"expr": "7 * 3"}}, window.tunables_panel.env
    )
    saved = window.capture_persisted_state()
    from zcu_tools.gui.app.workflows.persistence import WorkflowInputCodec

    caretaker = SingleFileCaretaker(
        window, codec=WorkflowInputCodec(), filename="input.json", cache_dir=tmp_path
    )
    caretaker.flush()
    raw = json.loads(caretaker.state_path.read_text())
    assert raw["plan"]["pi_length_us"] == {"expr": "0.1 + 0.2"}
    assert t1_input(raw["knobs"])["guess_us"] == {"expr": "7 * 3"}
    window.plan_panel.state.reset(
        CoherencePlanForm.default_input(window.plan_panel.env)
    )
    window.tunables_panel.state.reset(
        CoherenceTunablesForm.default_input(window.tunables_panel.env)
    )
    outcome = caretaker.restore_all()
    assert outcome.load_error is None
    assert window.capture_persisted_state() == saved
    assert window.plan_panel.state.base.data == saved.plan
    assert window.tunables_panel.state.base.data == saved.knobs


def test_incompatible_fields_drop_without_losing_other_input(
    window: WorkflowWindow[Calibrated],
) -> None:
    defaults = window.capture_persisted_state()
    from zcu_tools.gui.app.workflows.persistence import WorkflowInputSnapshot

    report = window.restore_persisted_state(
        WorkflowInputSnapshot(
            plan={"pi_length_us": ["wrong shape"], "removed": 4},
            knobs={"t1": {"guess_us": {"expr": "8 * 4"}}, "removed_knob": True},
        )
    )
    assert set(report.plan_dropped) == {("pi_length_us",), ("removed",)}
    assert report.knobs_dropped == (("removed_knob",),)
    restored = window.capture_persisted_state()
    assert restored.plan == defaults.plan
    assert t1_input(restored.knobs)["guess_us"] == {"expr": "8 * 4"}


def test_open_restore_and_close_flush(qapp: QApplication, tmp_path: Path) -> None:
    path = tmp_path / "t1_workflow_state_v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "plan": {"pi_length_us": {"expr": "1 / 4"}},
                "knobs": {"t1": {"guess_us": 35.0}},
            }
        )
    )
    app = WorkflowsGuiBehavior(
        registrations=build_registry(), alias_rows=ALIAS_PRESET, cache_dir=tmp_path
    )
    app.assemble(None)
    try:
        assert app.window.capture_persisted_state().plan["pi_length_us"] == {
            "expr": "1 / 4"
        }
        assert t1_input(app.window.capture_persisted_state().knobs)["guess_us"] == 35.0
        app.window.tunables_panel.state.write(
            {("t1", "guess_us"): {"expr": "9 * 5"}}, app.window.tunables_panel.env
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert t1_input(raw["knobs"])["guess_us"] == {"expr": "9 * 5"}
        assert raw["plan"]["pi_length_us"] == {"expr": "1 / 4"}
    finally:
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize(
    "payload",
    [
        "not JSON",
        '{"version": 2, "plan": {}, "knobs": {}}',
        '{"version": 1, "plan": [], "knobs": {}}',
    ],
)
def test_unreadable_snapshot_reports_and_uses_defaults(
    qapp: QApplication, tmp_path: Path, payload: str
) -> None:
    (tmp_path / "t1_workflow_state_v1.json").write_text(payload)
    app = WorkflowsGuiBehavior(
        registrations=build_registry(), alias_rows=ALIAS_PRESET, cache_dir=tmp_path
    )
    app.assemble(None)
    try:
        assert (
            app.window.plan_panel.state.input.data
            == CoherencePlanForm.default_input(app.window.plan_panel.env).data
        )
        assert (
            app.window.tunables_panel.state.input.data
            == CoherenceTunablesForm.default_input(app.window.tunables_panel.env).data
        )
        assert any(
            "Failed to read GUI state" in label.text()
            for label in app.window.findChildren(QLabel)
        )
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


def test_clean_skips_read_but_close_replaces_snapshot(
    qapp: QApplication, tmp_path: Path
) -> None:
    path = tmp_path / "t1_workflow_state_v1.json"
    path.write_text("deliberately invalid JSON")
    app = WorkflowsGuiBehavior(
        registrations=build_registry(),
        alias_rows=ALIAS_PRESET,
        clean=True,
        cache_dir=tmp_path,
    )
    app.assemble(None)
    try:
        assert path.read_text() == "deliberately invalid JSON"
        assert not any(
            "Failed to read GUI state" in label.text()
            for label in app.window.findChildren(QLabel)
        )
        assert (
            app.window.tunables_panel.state.input.data
            == CoherenceTunablesForm.default_input(app.window.tunables_panel.env).data
        )
        app.window.plan_panel.state.reset(
            FormInput(CoherencePlanForm, {"pi_length_us": {"expr": "3 / 8"}})
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert raw["plan"]["pi_length_us"] == {"expr": "3 / 8"}
    finally:
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()
