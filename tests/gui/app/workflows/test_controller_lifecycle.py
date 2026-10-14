"""Controller lifecycle contracts through public native/GUI seams."""

from pathlib import Path
from threading import get_ident
from time import monotonic

import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    EnginePorts,
    InvalidRunState,
    Next,
    RunMismatch,
    RunStatus,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.operation_runner import OperationRunner

from tests.experiment.workflows._engine_fakes import (
    Plan,
    Rig,
    State,
    Tunables,
)

from ._controller_support import create_binding, create_controller, declare, identity
from ._start_fakes import StartHarness


@pytest.mark.parametrize("stop_pending", [False, True])
def test_resume_without_device_callback_retains_handle_lease_records_and_bar(
    tmp_path: Path,
    stop_pending: bool,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)

    @workflow("snapshotless-resume", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan
        env.pbar("points", 2).set_progress(state.index)
        yield from env.wait_until(env.started_at)
        if state.index == 1:
            return Done()
        state.index += 1
        controller.pause("run")
        return Next(tun.value, state)

    token = controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.complete_segment()
    engine = controller.engine
    assert engine is not None
    records = engine.records("run")
    assert len(records) == 1
    bars = harness.progress.bars_for_owner("example")
    assert len(bars) == 1 and harness.gate.snapshot()
    with pytest.raises(RunMismatch):
        controller.resume("old")
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        controller.resume("run")
    paused = controller.status()
    assert paused is not None and paused.lifecycle == "paused"
    assert controller.operation_id == token and harness.gate.snapshot()
    assert harness.progress.bars_for_owner("example") == bars
    assert harness.handles.known_outcome(token) is None
    harness.background.fail_submit = False
    assert controller.resume("run") == token
    with pytest.raises(InvalidRunState):
        controller.resume("run")
    assert engine.records("run") == records
    if stop_pending:
        controller.stop("run")
    assert harness.gate.snapshot() and harness.handles.known_outcome(token) is None
    harness.background.execute()
    harness.owner.pump_all()
    assert harness.gate.snapshot() and controller.has_pending_segment
    harness.background.deliver()
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "cancelled" if stop_pending else "finished"
    )
    assert len(rig.initialized) == 1 and engine.records("run") == records
    assert not harness.gate.snapshot() and controller.operation_id is None
    assert not harness.progress.bars_for_owner("example")


def test_native_step_failure_is_not_reported_as_finished(tmp_path: Path) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)

    @workflow("bad-step", plan=Plan, tunables=Tunables, init=rig.init)
    def fail(
        env: WorkflowEnv[int],
        plan: Plan,
        tun: Tunables,
        state: State,
    ) -> Step[State, int]:
        del env, plan, tun, state
        yield from ()
        raise ValueError("workflow failed")

    controller = create_controller(harness, rig, lambda _status, _error: None)
    token = controller.start(
        fail,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "failed"
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert "workflow failed" in (outcome.error or "")
    assert not harness.gate.snapshot()


def test_owner_view_failure_preserves_native_outcome_after_terminal_cleanup(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    callback_observations: list[tuple[OperationOutcome | None, bool, bool]] = []

    def fail_callback(status: RunStatus | None, error: Exception | None) -> None:
        callback_observations.append(
            (
                harness.handles.known_outcome(token),
                not harness.gate.snapshot(),
                controller.operation_id is None,
            )
        )
        assert status is not None and status.lifecycle == "done"
        assert error is None
        raise ValueError("view failed")

    controller = create_controller(harness, rig, fail_callback)
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.background.deliver()
    assert callback_observations == [(OperationOutcome("finished"), True, True)]
    assert controller.operation_id is None
    assert harness.handles.known_outcome(token) == OperationOutcome("finished")
    assert not harness.gate.snapshot()
    assert "view failed" in caplog.text


def test_native_execution_is_off_owner_and_terminal_delivery_returns_to_owner(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    owner_id = get_ident()
    executed: list[int] = []
    delivered: list[int] = []
    background = ThreadPoolBackgroundExecutor(harness.owner)
    runner = OperationRunner(
        harness.gate, harness.handles, harness.progress, background, harness.bus
    )

    @workflow("worker-seam", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del env, plan, tun, state
        executed.append(get_ident())
        yield from ()
        return Done()

    controller = WorkflowController(
        runner,
        "worker-seam",
        on_finished=lambda _status, _error: delivered.append(get_ident()),
    )
    try:
        token = controller.start(
            step,
            plan=Plan(),
            tunables=Tunables(),
            ports=EnginePorts(
                rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
            ),
            binding=create_binding(),
            paths=rig.paths,
            identity=identity(),
        )
        deadline = monotonic() + 5.0
        while not delivered and monotonic() < deadline:
            harness.owner.pump_once(block=True, timeout=0.05)
        assert delivered == [owner_id]
        assert len(executed) == 1 and executed[0] != owner_id
        assert harness.handles.known_outcome(token) == OperationOutcome("finished")
        assert not harness.gate.snapshot()
    finally:
        controller.stop("run")
        assert background.quiesce(timeout=5.0)


def test_pause_resume_updates_the_same_bar(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    attempts: list[int] = []

    @workflow("paused-progress", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, state
        attempts.append(tun.value)
        bar = env.pbar("work", 3)
        bar.set_progress(len(attempts))
        bar.refresh()
        if len(attempts) < 3:
            controller.pause("run")
            yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.owner.pump_all()
    status = controller.status()
    assert status is not None and status.lifecycle == "paused"
    bars = harness.progress.bars_for_operation(token)
    assert len(bars) == 1
    handle, model = bars[0]
    assert model.n == 1
    with pytest.raises(InvalidRunState):
        controller.resume("run")
    harness.background.deliver()
    assert harness.gate.snapshot() and controller.operation_id == token
    assert harness.handles.known_outcome(token) is None
    assert harness.progress.bars_for_owner("example") == bars
    engine = controller.engine
    assert engine is not None
    engine.update_tunables(
        "run",
        (TunableChange("value", 2),),
        expected_revision=0,
        actor=Actor("user", "offline"),
    )

    second = controller.resume("run")
    assert second == token
    harness.background.execute()
    harness.owner.pump_all()
    assert model.n == 2
    assert harness.progress.bars_for_owner("example") == ((handle, model),)
    harness.background.deliver()

    third = controller.resume("run")
    harness.complete_segment()
    terminal = controller.status()
    assert terminal is not None and terminal.lifecycle == "done"
    assert attempts == [1, 2, 2]
    assert len(rig.initialized) == 1
    assert not harness.progress.bars_for_owner(
        "example"
    ) and harness.handles.known_outcome(third) == OperationOutcome("finished")


@pytest.mark.parametrize("ending", ["stop", "failure"])
def test_paused_run_discards_start_progress_on_stop_or_resumed_failure(
    tmp_path: Path,
    ending: str,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    attempts = 0

    @workflow("terminal-progress", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        nonlocal attempts
        del plan, tun, state
        attempts += 1
        env.pbar("work", 2).refresh()
        if attempts == 1:
            controller.pause("run")
            yield from env.wait_until(env.started_at)
        raise ValueError("resumed workflow failed")

    controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()
    assert harness.progress.bars_for_owner("example")
    if ending == "stop":
        status = controller.stop("run")
        assert status is not None and status.lifecycle == "stopped"
        assert attempts == 1
        assert len(finished) == 2 and finished[-1] == (status, None)
        controller.stop("run")
        assert len(finished) == 2
    else:
        resumed = controller.resume("run")
        harness.background.execute()
        harness.owner.pump_all()
        harness.background.deliver()
        status = controller.status()
        assert status is not None and status.lifecycle == "failed"
        assert finished[-1] == (status, None)
        assert "resumed workflow failed" in (status.reason or "")
        outcome = harness.handles.known_outcome(resumed)
        assert outcome is not None and outcome.status == "failed"
        assert outcome.error == status.reason
    assert not harness.progress.bars_for_owner("example")
    assert not harness.gate.snapshot()
    assert controller.operation_id is None


def test_resume_admission_failure_keeps_paused_run_and_progress(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    attempts = 0

    @workflow("retry-admission", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        nonlocal attempts
        del plan, tun, state
        attempts += 1
        env.pbar("work", 2).refresh()
        if attempts == 1:
            controller.pause("run")
            yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()
    before = controller.status()
    bars = harness.progress.bars_for_owner("example")
    harness.background.fail_submit = True
    try:
        with pytest.raises(OSError, match="submit failed"):
            controller.resume("run")
        assert controller.status() == before
        assert controller.operation_id == token
        assert harness.handles.live_count() == 1
        assert harness.handles.known_outcome(token) is None
        assert harness.gate.snapshot()
        assert harness.progress.bars_for_owner("example") == bars
    finally:
        harness.background.fail_submit = False
    resumed = controller.resume("run")
    harness.background.execute()
    harness.background.deliver()
    assert harness.handles.known_outcome(resumed) == OperationOutcome("finished")


def test_stop_during_resume_submission_does_not_restart_paused_step(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    attempts: list[str] = []

    @workflow("cancel-resume", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        controller.pause("run")
        yield from env.wait_until(env.started_at)
        return Done()

    controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()

    def during_submit() -> None:
        controller.stop("run")

    harness.background.on_submit = during_submit
    resumed = controller.resume("run")
    harness.background.on_submit = None
    assert harness.gate.snapshot()
    harness.background.execute()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert attempts == ["step"]
    assert harness.handles.known_outcome(resumed) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot()


def test_resume_submission_stop_waits_for_owner_terminal_delivery(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    attempts: list[str] = []

    @workflow("register-stop", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        controller.pause("run")
        yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.complete_segment()
    paused = controller.status()
    assert paused is not None and paused.lifecycle == "paused"
    notifications: list[tuple[int, bool, bool]] = []

    def during_submit() -> None:
        controller.stop("run")
        notifications.append(
            (
                len(finished),
                bool(harness.progress.bars_for_owner("example")),
                controller.has_pending_segment,
            )
        )

    harness.background.on_submit = during_submit
    resumed = controller.resume("run")
    harness.background.on_submit = None
    assert resumed == token
    assert notifications == [(1, True, True)]
    assert controller.has_pending_segment
    assert len(finished) == 1 and harness.gate.snapshot()
    harness.background.execute()
    assert len(finished) == 1 and harness.gate.snapshot()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert finished == [(paused, None), (status, None)]
    assert attempts == ["step"]
    assert harness.handles.known_outcome(resumed) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    controller.stop("run")
    assert len(finished) == 2


def test_resume_stop_rejection_finishes_without_a_producer(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    attempts: list[str] = []

    @workflow("stop-rejection", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun, state
        attempts.append("step")
        env.pbar("work", 2).refresh()
        controller.pause("run")
        yield from env.wait_until(env.started_at)
        return Done()

    controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.complete_segment()

    def during_submit() -> None:
        controller.stop("run")

    harness.background.on_submit = during_submit
    harness.background.fail_submit = True
    with pytest.raises(OSError, match="submit failed"):
        controller.resume("run")
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert len(finished) == 2 and finished[-1] == (status, None)
    assert not harness.progress.bars_for_owner("example")
    assert not controller.has_pending_segment
    assert controller.operation_id is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot() and harness.background.work is None
    assert attempts == ["step"]
    controller.stop("run")
    assert len(finished) == 2
