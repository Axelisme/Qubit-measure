"""Native workflow declarations and binding builders for Controller contracts."""

from collections.abc import Callable

from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Done,
    RunIdentity,
    RunStatus,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormModel, FormState

from tests.experiment.workflows._engine_fakes import (
    Plan,
    Rig,
    State,
    Tunables,
)

from ._start_fakes import StartHarness


def identity(run_id: str = "run") -> RunIdentity:
    return RunIdentity(
        run_id,
        SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname="offline",
        ),
    )


class ValueForm(FormModel):
    """Ordinary input corresponding to the example workflow tunable."""

    value: int = 1

    def to_cfg(self) -> Tunables:
        return Tunables(value=self.value)


def create_binding(run_id: str = "run") -> TunablesHost[ValueForm, int]:
    env = FormEnv()
    return TunablesHost(
        FormState(ValueForm.default_input(env)), env, run_id, to_cfg=ValueForm.to_cfg
    )


def create_controller(
    harness: StartHarness,
    rig: Rig,
    finished: Callable[[RunStatus | None, Exception | None], None],
) -> WorkflowController[ValueForm, int]:
    return WorkflowController(
        harness.runner,
        "example",
        on_finished=finished,
    )


def finish(
    env: WorkflowEnv[int],
    plan: Plan,
    tun: Tunables,
    state: State,
) -> Step[State, int]:
    del plan, tun, state
    env.pbar("steps", 1).set_progress(1)
    yield from ()
    return Done()


def declare(rig: Rig):
    def invoke(
        env: WorkflowEnv[int],
        plan: Plan,
        tun: Tunables,
        state: State,
    ) -> Step[State, int]:
        return finish(env, plan, tun, state)

    return workflow("example", plan=Plan, tunables=Tunables, init=rig.init)(invoke)
