"""Workflow input mementos through public Window and app lifecycle seams."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from qtpy.QtWidgets import QApplication, QLabel
from zcu_tools.datafile import RunSnapshot
from zcu_tools.experiment.workflows import (
    Done,
    InitEnv,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.app import WorkflowsGuiBehavior
from zcu_tools.gui.app.workflows.attachment import WorkflowAttachment
from zcu_tools.gui.app.workflows.persistence import (
    WorkflowInputCodec,
    WorkflowInputSnapshot,
)
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, ModuleInput
from zcu_tools.gui.session.persistence import SingleFileCaretaker

from tests.experiment.workflows._engine_fakes import Plan, State, Tunables
from tests.gui.app.workflows._start_fakes import StartHarness
from tests.gui.app.workflows._synthetic import PlanForm, TunablesForm

from ._sample_input import sample_input
from ._synthetic import ALIAS_PRESET, Context, build_registry


@pytest.fixture
def window(qapp: QApplication) -> Iterator[WorkflowWindow[Context]]:
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
    window: WorkflowWindow[Context], tmp_path: Path
) -> None:
    window.plan_panel.state.write(
        {("scale",): {"expr": "0.1 + 0.2"}}, window.plan_panel.env
    )
    window.tunables_panel.state.write(
        {("sample", "value"): {"expr": "7 * 3"}}, window.tunables_panel.env
    )
    saved = window.capture_persisted_state()
    caretaker = SingleFileCaretaker(
        window, codec=WorkflowInputCodec(), filename="input.json", cache_dir=tmp_path
    )
    caretaker.flush()
    raw = json.loads(caretaker.state_path.read_text())
    assert raw["plan"]["scale"] == {"expr": "0.1 + 0.2"}
    assert sample_input(raw["knobs"])["value"] == {"expr": "7 * 3"}
    window.plan_panel.state.reset(PlanForm.default_input(window.plan_panel.env))
    window.tunables_panel.state.reset(
        TunablesForm.default_input(window.tunables_panel.env)
    )
    outcome = caretaker.restore_all()
    assert outcome.load_error is None
    assert window.capture_persisted_state() == saved
    assert window.plan_panel.state.base.data == saved.plan
    assert window.tunables_panel.state.base.data == saved.knobs


def test_incompatible_fields_drop_without_losing_other_input(
    window: WorkflowWindow[Context],
) -> None:
    defaults = window.capture_persisted_state()
    report = window.restore_persisted_state(
        WorkflowInputSnapshot(
            plan={"scale": ["wrong shape"], "removed": 4},
            knobs={"sample": {"value": {"expr": "8 * 4"}}, "removed_knob": True},
        )
    )
    assert set(report.plan_dropped) == {("scale",), ("removed",)}
    assert report.knobs_dropped == (("removed_knob",),)
    restored = window.capture_persisted_state()
    assert restored.plan == defaults.plan
    assert sample_input(restored.knobs)["value"] == {"expr": "8 * 4"}


def test_open_restore_and_close_flush(qapp: QApplication, tmp_path: Path) -> None:
    path = tmp_path / "t1_workflow_state_v1.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "plan": {"scale": {"expr": "1 / 4"}},
                "knobs": {"sample": {"value": 35.0}},
            }
        )
    )
    app = WorkflowsGuiBehavior(
        registrations=build_registry(), alias_rows=ALIAS_PRESET, cache_dir=tmp_path
    )
    app.assemble(None)
    try:
        assert app.window.capture_persisted_state().plan["scale"] == {"expr": "1 / 4"}
        assert sample_input(app.window.capture_persisted_state().knobs)["value"] == 35.0
        app.window.tunables_panel.state.write(
            {("sample", "value"): {"expr": "9 * 5"}}, app.window.tunables_panel.env
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert sample_input(raw["knobs"])["value"] == {"expr": "9 * 5"}
        assert raw["plan"]["scale"] == {"expr": "1 / 4"}
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
            == PlanForm.default_input(app.window.plan_panel.env).data
        )
        assert (
            app.window.tunables_panel.state.input.data
            == TunablesForm.default_input(app.window.tunables_panel.env).data
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
            == TunablesForm.default_input(app.window.tunables_panel.env).data
        )
        app.window.plan_panel.state.reset(
            FormInput(PlanForm, {"scale": {"expr": "3 / 8"}})
        )
        assert app.window.close()
        raw = json.loads(path.read_text())
        assert raw["plan"]["scale"] == {"expr": "3 / 8"}
    finally:
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()


class DrivePlanForm(FormModel):
    """Synthetic Start plan with one library reference and one ordinary leaf."""

    drive: ModuleInput
    count: int = 2


class ValueKnobsForm(FormModel):
    """Synthetic tunables form matching the native positive value."""

    value: int = 1


def drive_registration() -> WorkflowAttachment[
    Plan, Tunables, State, int, DrivePlanForm, ValueKnobsForm, None
]:
    """Declare a fresh context-free workflow; restoration never runs it."""

    def init(env: InitEnv[None], plan: Plan) -> State:
        del env, plan
        return State()

    def step(
        env: WorkflowEnv[None], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        return Done()

    def plan_to_cfg(form: DrivePlanForm, env: FormEnv, snapshot: RunSnapshot) -> Plan:
        del env, snapshot
        return Plan(count=form.count)

    declared = workflow("synthetic", plan=Plan, tunables=Tunables, init=init)(step)
    return WorkflowAttachment(
        step=declared,
        plan_model=DrivePlanForm,
        knobs_model=ValueKnobsForm,
        plan_to_cfg=plan_to_cfg,
        knobs_to_cfg=lambda form: Tunables(value=form.value),
        seed=lambda plan, form: None,
    )


def test_saved_reference_without_captured_library_is_discarded_on_open(
    qapp: QApplication, tmp_path: Path
) -> None:
    (tmp_path / "t1_workflow_state_v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "plan": {"drive": {"kind": "ref", "name": "Q1.pulses.pi"}, "count": 5},
                "knobs": {"value": 3},
            }
        )
    )
    app = WorkflowsGuiBehavior(registrations=[drive_registration()], cache_dir=tmp_path)
    app.assemble(None)
    try:
        assert app.window.capture_persisted_state() == WorkflowInputSnapshot(
            plan={"drive": {"kind": "ref"}, "count": 5}, knobs={"value": 3}
        )
        assert any(
            label.text() == "Discarded plan: drive.name"
            for label in app.window.findChildren(QLabel)
        )
    finally:
        app.window.close()
        app.shutdown()
        app.window.deleteLater()
        qapp.processEvents()
