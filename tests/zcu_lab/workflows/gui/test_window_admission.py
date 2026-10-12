"""Window admission contracts through public native/GUI seams."""

from collections.abc import Callable

import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    InitEnv,
    InvalidRunState,
    Next,
    Step,
    TunablesSnapshot,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.form import FormErrors
from zcu_tools.gui.session.events import GateChangedPayload
from zcu_tools.resources.entry import (
    ResultEntry,
)

from tests.gui.app.workflows._start_fakes import StartHarness
from zcu_lab.workflows.calibration import Calibrated
from zcu_lab.workflows.coherence import CoherencePlan, CoherenceTunables
from zcu_lab.workflows.gui.window import T1WorkflowWindow

from ._coherence_input import t1_input
from ._reload_source import InjectedCatalog
from ._window_scenarios import admission_canceller, install_admission_observer


@pytest.mark.parametrize("action", ["stop", "close"])
@pytest.mark.parametrize("previous_start", [False, True])
@pytest.mark.parametrize("notification", ["progress", "gate"])
def test_admission_notification_can_cancel_current_start(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    action: str,
    previous_start: bool,
    notification: str,
) -> None:
    window, harness, entry = run_attachment
    fixed = window.controller
    calls = install_admission_observer(window, catalog_source)
    if previous_start:
        assert isinstance(window.start("previous"), int)
        harness.complete_segment()
        assert not harness.gate.snapshot()
        calls.clear()
    old = fixed.binding
    if old is not None:
        old.state.write({("t1", "guess_us"): 87.0}, old.env)
    old_input = None if old is None else old.state.input.data
    cancel_on_notification, close_results, observed, errors = admission_canceller(
        window, action, old, fixed
    )

    def cancel_on_register(payload: GateChangedPayload) -> None:
        if payload.active:
            cancel_on_notification()

    detach: Callable[[], None]
    if notification == "progress":
        detach = harness.progress.attach_by_owner("t1-workflow", cancel_on_notification)
    else:
        subscription = harness.bus.subscribe(GateChangedPayload, cancel_on_register)
        detach = subscription.unsubscribe
    try:
        token = window.start("notification")
    finally:
        detach()
    assert isinstance(token, int)
    assert observed == [("notification", token, True, True)]
    assert len(errors) == 1 and "accepted run" in errors[0]
    if old is not None:
        assert old.state.input.data == old_input
    pending = fixed.binding
    assert (
        pending is not None and pending.state.input.data == pending.accepted_input.data
    )
    if action == "close":
        assert close_results == [False]
    controller = window.controller
    assert controller is fixed and controller.operation_id == token
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.execute()
    assert calls == []
    assert controller.status() is None and harness.gate.snapshot()
    harness.background.deliver()
    assert controller.operation_id is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not (entry.result_path / "workflows/notification").exists()
    assert not (entry.database_path / "workflows/notification").exists()
    assert window.close()


def test_replacement_accepted_progress_uses_current_host_and_form_capture(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
) -> None:
    window, harness, _entry = run_attachment
    attempts: list[float] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> int:
        return 0

    @workflow(
        "accepted-replacement",
        plan=CoherencePlan,
        tunables=CoherenceTunables,
        init=init,
    )
    def step(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: int,
    ) -> Step[int, float]:
        del plan
        attempts.append(tun.t1.guess_us)
        if state == 0:
            env.pbar("work", 2).refresh()
            harness.owner.pump_all()
            yield from ()
            return Next(tun.t1.guess_us, 1)
        return Done()

    catalog_source.install(step, window)
    assert isinstance(window.start("old"), int)
    harness.complete_segment()
    old = window.controller.binding
    assert old is not None
    old.state.write({("t1", "guess_us"): 87.0}, old.env)
    old_input, old_base = old.state.input.data, old.state.base.data
    context = harness.services.context
    context.set_parameter(context.read_context().target, "general.ext.guess_us", 41.0)
    observed: list[bool] = []
    replies: list[TunablesSnapshot | FormErrors] = []

    def apply_on_accepted_progress() -> None:
        if observed:
            return
        status = window.controller.status()
        if status is None or status.run_id != "next" or status.lifecycle != "running":
            return
        replies.append(
            window.apply_tunables(
                "next",
                {("t1", "guess_us"): {"expr": "general.ext.guess_us + 2"}},
                actor=Actor("agent", "rpc"),
            )
        )
        panel = window.tuning_panel
        observed.append(
            panel is not None
            and panel.host is window.controller.binding
            and panel.host is not old
            and panel.host.env is not old.env
        )

    detach = harness.progress.attach_by_owner("t1-workflow", apply_on_accepted_progress)
    try:
        assert isinstance(window.start("next"), int)
        harness.complete_segment()
    finally:
        detach()
    assert observed == [True] and len(replies) == 1
    reply = replies[0]
    assert isinstance(reply, TunablesSnapshot) and reply.revision == 1
    assert isinstance(reply.values, dict) and t1_input(reply.values)["guess_us"] == 43.0
    assert attempts == [20.0, 20.0, 41.0, 43.0]
    assert old.state.input.data == old_input and old.state.base.data == old_base
    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0


@pytest.mark.parametrize("previous_start", [False, True])
def test_gate_registration_reentrant_start_preserves_outer_controller(
    run_attachment: tuple[T1WorkflowWindow, StartHarness, ResultEntry],
    catalog_source: InjectedCatalog,
    previous_start: bool,
) -> None:
    window, harness, entry = run_attachment
    calls: list[str] = []

    def init(_env: InitEnv[Calibrated], _plan: CoherencePlan) -> None:
        calls.append("init")

    @workflow(
        "reentry-observer", plan=CoherencePlan, tunables=CoherenceTunables, init=init
    )
    def observe(
        env: WorkflowEnv[Calibrated],
        plan: CoherencePlan,
        tun: CoherenceTunables,
        state: None,
    ) -> Step[None, int]:
        del env, plan, tun, state
        calls.append("step")
        yield from ()
        return Done()

    catalog_source.install(observe, window)
    if previous_start:
        assert isinstance(window.start("previous"), int)
        window.stop("previous")
        harness.background.execute()
        harness.background.deliver()
        assert not harness.gate.snapshot()

    observed: list[tuple[str | None, int | None, bool]] = []
    rejections: list[InvalidRunState] = []
    preserved: list[bool] = []

    def reenter_on_register(payload: GateChangedPayload) -> None:
        if not payload.active:
            return
        outer = window.controller
        observed.append((outer.run_id, outer.operation_id, outer.has_pending_segment))
        try:
            window.start("nested")
        except InvalidRunState as exc:
            rejections.append(exc)
        finally:
            preserved.append(window.controller is outer)

    subscription = harness.bus.subscribe(GateChangedPayload, reenter_on_register)
    try:
        token = window.start("outer")
    finally:
        subscription.unsubscribe()
    assert observed == [("outer", token, True)]
    assert len(rejections) == 1 and rejections[0].action == "start"
    assert rejections[0].lifecycle == "running" and preserved == [True]
    controller = window.controller
    assert isinstance(token, int) and controller is not None
    assert (
        controller.run_id == "outer"
        and controller.operation_id == token
        and controller.status() is None
        and harness.gate.snapshot()
    )
    window.stop("outer")
    harness.background.execute()
    assert (
        calls == []
        and controller.status() is None
        and harness.gate.snapshot()
        and controller.operation_id == token
    )
    harness.background.deliver()
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    assert controller.operation_id is None and window.controller is controller
    assert not (entry.result_path / "workflows").exists()
    assert not (entry.database_path / "workflows").exists()
