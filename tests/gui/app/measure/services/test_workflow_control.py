"""Headless workflow hosting through synthetic registrations and shared session."""

from threading import Event

import pytest
from pydantic import BaseModel
from zcu_tools.experiment.workflows import Actor, Done, InitEnv, Next, Step, WorkflowEnv, workflow
from zcu_tools.gui.app.measure.services import app_services
from zcu_tools.gui.form import FormErrors, FormModel
from zcu_tools.gui.session.context_types import PointUseRequest
from zcu_tools.resources.entry import ResultEntry

from ._workflow_harness import StartHarness


class Values(FormModel):
    value: int = 3


class Native(BaseModel):
    value: int = 3


def make_control(tmp_path, *, blocked=False):
    facet_type = app_services.WorkflowControlFacet
    from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment

    harness = StartHarness()
    entry = ResultEntry.create(
        "synthetic", result_root=tmp_path / "result", database_root=tmp_path / "database"
    )
    entry.new_point("source", description="synthetic")
    harness.services.context.install_entry(entry)
    harness.services.context.use_point(PointUseRequest("source"))

    from zcu_tools.experiment.simulation import white_noise

    board = white_noise()
    harness.services.soc_connection.install_soc(board.soc, board.soccfg)
    control = facet_type(harness.state, harness.services, harness.owner)
    entered = Event()
    release = Event()

    def init(env: InitEnv[None], plan: Native) -> int:
        return 0

    @workflow("synthetic", plan=Native, tunables=Native, init=init)
    def step(
        env: WorkflowEnv[None], plan: Native, tun: Native, state: int
    ) -> Step[int, int]:
        if state:
            return Done()
        entered.set()
        if blocked:
            # The test releases this before running the deferred producer.
            assert release.wait(30)
        yield from env.wait_until(env.started_at)
        return Next(tun.value, 1)

    attachment = WorkflowAttachment(
        step=step,
        plan_model=Values,
        knobs_model=Values,
        plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
        knobs_to_cfg=lambda form: Native(value=form.value),
        seed=lambda plan, form: None,
    )
    control.configure((attachment,))
    return control, harness, entered, release


def test_same_name_tabs_keep_independent_drafts_and_close_without_session_shutdown(tmp_path):
    control, harness, _, _ = make_control(tmp_path)
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
    token = control.start(third, "after-close")
    harness.complete_segment()
    assert harness.handles.await_outcome(token, timeout=0).reason == "completed"


def test_start_records_tunables_and_stale_run_targets_use_same_headless_owner(tmp_path):
    control, harness, _, _ = make_control(tmp_path)
    tab = control.open("synthetic")
    token = control.start(tab, "first", tunables_batch={("value",): 11})
    assert token > 0
    harness.complete_segment()
    assert control.status("first").lifecycle == "done"
    assert control.tunables("first").values == {"value": 11}
    assert len(control.records("first")) == 1
    assert control.figures("first") == ()
    assert control.pending_question("first") is None
    assert control.answer("first", {"value": 3}) is None
    with pytest.raises(Exception, match="stale|match|Unknown"):
        control.stop("not-this-run")
    assert control.close(tab)


def test_pending_and_paused_run_retain_global_lease_and_reject_tab_close(tmp_path):
    control, harness, _, _ = make_control(tmp_path)
    tab = control.open("synthetic")
    other = control.open("synthetic")
    control.start(tab, "held")
    assert not control.close(tab)
    with pytest.raises(Exception):
        control.start(other, "competing")
    harness.background.execute()
    # Producer return is not owner delivery: close and competing Run still reject.
    assert not control.close(tab)
    harness.background.deliver()
    assert control.close(tab)
    control.start(other, "next")
    harness.complete_segment()
    assert control.status("next").lifecycle == "done"


def test_pause_resume_stop_share_one_operation_and_keep_tunables_binding(tmp_path):
    control, harness, _, _ = make_control(tmp_path)
    tab = control.open("synthetic")
    token = control.start(tab, "paused")
    # Pause from the native init boundary before the first generator commits.
    # Native producer is deferred; pause is requested by the first plot-free step.
    # The registration boundary below uses the public command, not engine internals.
    from zcu_tools.gui.app.measure.services.workflow_attachment import WorkflowAttachment

    def init(env: InitEnv[None], plan: Native) -> int:
        return 0

    # Stop the initial pending run without creating files, then use a pausing registration.
    control.stop("paused")
    harness.complete_segment()
    assert control.close(tab)

    @workflow("pausing", plan=Native, tunables=Native, init=init)
    def step(env: WorkflowEnv[None], plan: Native, tun: Native, state: int) -> Step[int, int]:
        if state:
            return Done()
        control.pause("pausing-run")
        yield from env.wait_until(env.started_at)
        return Next(tun.value, 1)

    control.configure((WorkflowAttachment(
        step=step, plan_model=Values, knobs_model=Values,
        plan_to_cfg=lambda form, env, snapshot: Native(value=form.value),
        knobs_to_cfg=lambda form: Native(value=form.value), seed=lambda plan, form: None,
    ),))
    tab = control.open("pausing")
    token = control.start(tab, "pausing-run")
    harness.complete_segment()
    assert control.status("pausing-run").lifecycle == "paused"
    assert not control.close(tab)
    snapshot = control.apply_tunables("pausing-run", {("value",): 9}, actor=Actor("gui", "test"))
    assert not isinstance(snapshot, FormErrors)
    assert snapshot.revision == 1
    assert control.resume("pausing-run") == token
    control.stop("pausing-run")
    harness.complete_segment()
    assert control.status("pausing-run").lifecycle == "stopped"
    assert control.close(tab)
