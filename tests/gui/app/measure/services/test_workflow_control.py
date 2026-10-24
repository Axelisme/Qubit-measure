"""Headless workflow hosting through synthetic registrations and shared session."""

from threading import Event

import pytest
from pydantic import BaseModel, ConfigDict
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import (
    Actor,
    Live,
    RunMismatch,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.measure.services import app_services
from zcu_tools.gui.form import FormErrors, FormModel
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    PointCreateRequest,
    PointUseRequest,
)
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.resources.entry import ResultEntry

from ._workflow_harness import StartHarness

RUN_ID = "21ed5156dc9b441a8b5cda3cf26633f7"
OTHER_RUN_ID = "ffffffffffffffffffffffffffffffff"


class Values(FormModel):
    value: int = 3


class Native(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    value: int = 3


def make_control(tmp_path):
    facet_type = app_services.WorkflowControlFacet
    from zcu_tools.gui.app.measure.services.workflow_attachment import (
        WorkflowAttachment,
    )

    harness = StartHarness()
    entry = ResultEntry.create(
        "synthetic",
        result_root=tmp_path / "result",
        database_root=tmp_path / "database",
    )
    entry.new_point("source", description="synthetic")
    harness.services.context.install_entry(entry)
    harness.services.context.use_point(PointUseRequest("source"))

    from zcu_tools.experiment.simulation import white_noise

    board = white_noise()
    harness.services.soc_connection.install_soc(board.soc, board.soccfg)
    control = facet_type(harness.state, harness.services)

    @workflow("synthetic", "Synthetic form and record hosting")
    async def script(env: WorkflowEnv[None], plan: Native, tun: Live[Native]) -> None:
        await env.wait_until(env.started_at)
        env.commit(Native(value=tun.current.value))

    attachment = WorkflowAttachment(
        script=script,
        plan_model=Values,
        knobs_model=Values,
        plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
        knobs_to_cfg=lambda form: Native(value=form.value),
        seed=lambda plan, form: None,
    )
    control.configure((attachment,))
    return control, harness


@pytest.mark.parametrize(
    "run_id",
    [
        "q1-parity-51-20261025",
        "",
        "21ed5156dc9b441a8b5cda3cf26633f",
        "021ed5156dc9b441a8b5cda3cf26633f7",
        "21ED5156DC9B441A8B5CDA3CF26633F7",
        "g1ed5156dc9b441a8b5cda3cf26633f7",
        "２１ed5156dc9b441a8b5cda3cf26633f7",
        "21ed5156dc9b441a8b5cda3cf26633f7\n",
        "20261025T120000Z-ab12cd",
        "21ed5156dc9b441a8b5cda3cf26633f7-000001-1",
    ],
)
def test_start_rejects_invalid_root_before_operation_admission(tmp_path, run_id):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    with pytest.raises(ValueError, match="run_id.*32.*lowercase.*hex"):
        control.start(tab, run_id)
    snapshot = control.tabs()[0]
    assert snapshot.run_id is None
    assert snapshot.operation_id is None
    assert harness.background.work is None
    assert control.close(tab)


def test_same_name_tabs_keep_independent_drafts_and_close_without_session_shutdown(
    tmp_path,
):
    control, harness = make_control(tmp_path)
    one = control.open("synthetic")
    two = control.open("synthetic")
    assert one != two
    assert [tab.name for tab in control.tabs()] == ["synthetic", "synthetic"]
    control.write_form(one, "plan", {("value",): 7})
    assert control.form_state(one, "plan").input.data["value"] == 7
    assert control.form_state(two, "plan").input.data["value"] == 3
    assert control.close(one)
    assert control.close(two)
    assert control.tabs() == ()
    assert harness.state.session_env.soc is not None
    assert harness.services.context.read_context().capture_storage().point == "source"
    # Closing the last borrower does not stop admission on the session runner.
    third = control.open("synthetic")
    token = control.start(third, RUN_ID)
    harness.complete_segment()
    assert not isinstance(token, FormErrors)
    outcome = harness.handles.await_outcome(token, timeout=0)
    assert outcome is not None and outcome.reason == "completed"


def test_start_records_tunables_and_stale_run_targets_use_same_headless_owner(tmp_path):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    token = control.start(tab, RUN_ID, tunables_batch={("value",): 11})
    assert not isinstance(token, FormErrors)
    assert token > 0
    harness.complete_segment()
    status = control.status(RUN_ID)
    assert status is not None and status.lifecycle == "done"
    assert control.tunables(RUN_ID).values == {"value": 11}
    assert len(control.records(RUN_ID)) == 1
    assert control.figures(RUN_ID) == ()
    assert control.pending_question(RUN_ID) is None
    with pytest.raises(RunMismatch):
        control.stop("not-this-run")
    assert control.close(tab)


def test_pending_and_paused_run_retain_global_lease_and_reject_tab_close(tmp_path):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    other = control.open("synthetic")
    control.start(tab, RUN_ID)
    assert not control.close(tab)
    with pytest.raises(OperationConflictError):
        control.start(other, OTHER_RUN_ID)
    harness.background.execute()
    # Producer return is not owner delivery: close and competing Run still reject.
    assert not control.close(tab)
    harness.background.deliver()
    assert control.close(tab)
    control.start(other, OTHER_RUN_ID)
    harness.complete_segment()
    status = control.status(OTHER_RUN_ID)
    assert status is not None and status.lifecycle == "done"


def test_pause_resume_stop_share_one_operation_and_keep_tunables_binding(tmp_path):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    token = control.start(tab, RUN_ID)
    # Stop before native acceptance, then register a script with a public pause gate.
    from zcu_tools.gui.app.measure.services.workflow_attachment import (
        WorkflowAttachment,
    )

    # Stop the initial pending run without creating files, then use a pausing registration.
    control.stop(RUN_ID)
    harness.complete_segment()
    assert control.close(tab)

    entered = Event()

    def operation(ctx: CallContext) -> None:
        entered.set()
        if not ctx.scope.stop.event.wait(30):
            raise TimeoutError("Scenario did not interrupt the operation")

    @workflow("pausing", "Synthetic retained-worker pause")
    async def script(env: WorkflowEnv[None], plan: Native, tun: Live[Native]) -> None:
        await env.call(operation)
        env.commit(Native(value=tun.current.value))

    control.configure(
        (
            WorkflowAttachment(
                script=script,
                plan_model=Values,
                knobs_model=Values,
                plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
                knobs_to_cfg=lambda form: Native(value=form.value),
                seed=lambda plan, form: None,
            ),
        )
    )
    tab = control.open("pausing")
    token = control.start(tab, RUN_ID)
    harness.background.start()
    try:
        assert entered.wait(5), "The retained operation must start before Pause"
        control.pause(RUN_ID)
        harness.wait_for(
            lambda: (
                (status := control.status(RUN_ID)) is not None
                and status.lifecycle == "paused"
            ),
            label="native pause gate",
        )
        status = control.status(RUN_ID)
        assert status is not None and status.lifecycle == "paused"
        assert not control.close(tab)
        snapshot = control.apply_tunables(
            RUN_ID, {("value",): 9}, actor=Actor("user", "test")
        )
        assert not isinstance(snapshot, FormErrors)
        assert snapshot.revision == 1
        assert control.resume(RUN_ID) == token
    finally:
        control.stop(RUN_ID)
        harness.finish_worker()
    status = control.status(RUN_ID)
    assert status is not None and status.lifecycle == "cancelled"
    assert control.close(tab)


def test_prepare_forms_recaptures_after_terminal_without_losing_authored_draft(
    tmp_path,
):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    control.start(tab, RUN_ID)
    harness.complete_segment()
    control.write_form(tab, "tunables", {("value",): 19})
    old = control.form_env(tab, "tunables")
    harness.services.context.new_point(
        PointCreateRequest(
            "next", description="new source", origin=CommandOrigin("gui")
        )
    )
    harness.services.context.use_point(PointUseRequest("next"))
    assert control.form_env(tab, "tunables") is old
    control.prepare_forms(tab)
    refreshed = control.form_env(tab, "tunables")
    assert refreshed is not old
    assert refreshed.parameters is not None
    assert refreshed.parameters.capture_storage().point == "next"
    assert control.form_state(tab, "tunables").input.data["value"] == 19
    assert control.tunables(RUN_ID).values == {"value": 3}


def test_invalid_start_batch_returns_errors_without_accepting_draft(tmp_path):
    control, harness = make_control(tmp_path)
    tab = control.open("synthetic")
    control.write_form(tab, "plan", {("value",): 17})
    try:
        result = control.start(tab, RUN_ID, plan_batch={("unknown",): 1})
    except ValueError as error:
        result = error
    assert isinstance(result, FormErrors)
    assert control.form_state(tab, "plan").input.data["value"] == 17
    assert control.tabs()[0].run_id is None
    assert harness.background.work is None
