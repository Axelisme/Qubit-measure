"""Saved workflow input restoration through the generic host's assembly seam."""

import json
from pathlib import Path

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
from zcu_tools.gui.app.workflows.persistence import WorkflowInputSnapshot
from zcu_tools.gui.form import FormEnv, FormModel, ModuleInput

from tests.experiment.workflows._engine_fakes import Plan, State, Tunables


class DrivePlanForm(FormModel):
    """Synthetic Start plan with one library reference and one ordinary leaf."""

    drive: ModuleInput
    count: int = 2


class ValueKnobsForm(FormModel):
    """Synthetic tunables form matching the native positive value."""

    value: int = 1


def registration() -> WorkflowAttachment[
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
    app = WorkflowsGuiBehavior(registrations=[registration()], cache_dir=tmp_path)
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
