"""Native startup, cancellation and lease cleanup through the public controller."""

from collections.abc import Callable
from pathlib import Path
from threading import get_ident
from time import monotonic

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    Actor,
    DeviceSnapshot,
    Done,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
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
from zcu_tools.gui.session.ports import OperationConflictError
from zcu_tools.gui.session.services.progress import (
    BoundProgressFactory,
    ProgressService,
)

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    Plan,
    Rig,
    State,
    Tunables,
)

from ._start_fakes import StartHarness


def identity() -> RunIdentity:
    return RunIdentity(
        "run",
        DeviceSnapshot({}),
        SoftwareProvenance(
            software_versions={},
            git_commit=None,
            git_dirty=None,
            qick_version=None,
            soc_fingerprint=None,
            hostname="offline",
        ),
    )


def create_controller(
    harness: StartHarness,
    rig: Rig,
    finished: Callable[[RunStatus | None, Exception | None], None],
) -> WorkflowController[int]:
    return WorkflowController(
        harness.runner,
        EnginePorts(rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices),
        "example",
        harness.progress,
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


def test_submit_is_pending_until_worker_accepts_and_owner_releases_lease(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    assert controller.operation_id == token
    assert controller.engine is controller.status() is None
    assert not rig.paths.metadata_root.exists()
    assert harness.handles.live_count() == 1
    assert len(harness.gate.snapshot()) == 1
    with pytest.raises(InvalidRunState):
        controller.start(
            declare(rig),
            plan=Plan(),
            tunables=Tunables(),
            paths=rig.paths,
            identity=identity(),
        )
    with pytest.raises(OperationConflictError):
        harness.gate.ensure_can_start("device_setup", resource_id="flux")
    harness.background.execute()
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    harness.owner.pump_all()
    assert harness.progress.bars_for_operation(token)
    assert controller.operation_id == token
    assert finished == []
    harness.background.deliver()
    assert finished == [(status, None)]
    assert controller.operation_id is None
    assert harness.handles.known_outcome(token) == OperationOutcome("finished")
    assert not harness.gate.snapshot()
    assert harness.progress.bars_for_operation(token) == ()
    assert rig.initialized
    # Runner-bound progress, not the constructor's placeholder factory.
    assert not rig.bars


def test_submit_failure_has_no_native_artifacts_or_lease(tmp_path: Path) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    harness.background.fail_submit = True
    controller = create_controller(harness, rig, lambda _status, _error: None)
    with pytest.raises(OSError, match="submit failed"):
        controller.start(
            declare(rig),
            plan=Plan(),
            tunables=Tunables(),
            paths=rig.paths,
            identity=identity(),
        )
    assert controller.status() is None
    assert controller.operation_id is None
    assert harness.handles.live_count() == 0
    assert not harness.gate.snapshot()
    assert not rig.paths.metadata_root.exists()
    assert not rig.paths.data_root.exists()


def test_stop_before_worker_acceptance_does_not_start_run(tmp_path: Path) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.handles.cancel(token)
    assert controller.stop() is None
    assert harness.gate.snapshot()
    harness.background.execute()
    assert harness.gate.snapshot()
    harness.background.deliver()
    assert finished == [(None, None)]
    assert harness.handles.known_outcome(token) == OperationOutcome("cancelled")
    assert controller.status() is None
    assert not rig.paths.metadata_root.exists()
    assert not rig.initialized
    assert not harness.gate.snapshot()


def test_native_startup_failure_reports_original_error_and_releases(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    rig.paths.metadata_root.mkdir()
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    assert harness.background.error is not None
    harness.background.deliver()
    assert finished == [(None, harness.background.error)]
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert not harness.gate.snapshot()
    assert controller.operation_id is None
    assert controller.engine is None
    assert not rig.paths.data_root.exists()


def test_stop_during_start_is_settled_before_init_and_step(tmp_path: Path) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)

    requested = False

    class StoppingClock(ManualClock):
        def now(self):
            nonlocal requested
            if not requested:
                requested = True
                controller.stop()
            return super().now()

    ports = EnginePorts(rig.plots, rig.progress, StoppingClock(), context=7)
    controller = WorkflowController(
        harness.runner,
        ports,
        "example",
        harness.progress,
        on_finished=lambda _status, _error: None,
    )
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert not rig.initialized
    assert status.committed_seq == 0
    assert harness.gate.snapshot()
    harness.background.deliver()
    assert harness.handles.known_outcome(token) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot()


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
        fail, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
    )
    harness.background.execute()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "failed"
    outcome = harness.handles.known_outcome(token)
    assert outcome is not None and outcome.status == "failed"
    assert "workflow failed" in (outcome.error or "")
    assert not harness.gate.snapshot()


def test_owner_callback_failure_cannot_leak_lease_or_pending_admission(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)

    def fail_callback(_status: RunStatus | None, _error: Exception | None) -> None:
        raise ValueError("view failed")

    controller = create_controller(harness, rig, fail_callback)
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.background.deliver()
    assert controller.operation_id is None
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", "view failed"
    )
    assert not harness.gate.snapshot()


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
        EnginePorts(rig.plots, rig.progress, rig.clock, context=7),
        "worker-seam",
        harness.progress,
        on_finished=lambda _status, _error: delivered.append(get_ident()),
    )
    try:
        token = controller.start(
            step, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
        )
        deadline = monotonic() + 5.0
        while not delivered and monotonic() < deadline:
            harness.owner.pump_once(block=True, timeout=0.05)
        assert delivered == [owner_id]
        assert len(executed) == 1 and executed[0] != owner_id
        assert harness.handles.known_outcome(token) == OperationOutcome("finished")
        assert not harness.gate.snapshot()
    finally:
        controller.stop()
        assert background.quiesce(timeout=5.0)


@pytest.fixture
def progress_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[int], list[int]]:
    """Observe the explicit D44 factory/discard contract using the real service."""
    created: list[int] = []
    discarded: list[int] = []
    make_factory = ProgressService.make_factory
    discard_operation = ProgressService.discard_operation

    def make(service: ProgressService, token: int, owner: str) -> BoundProgressFactory:
        created.append(token)
        return make_factory(service, token, owner)

    def discard(service: ProgressService, token: int) -> None:
        discarded.append(token)
        discard_operation(service, token)

    monkeypatch.setattr(ProgressService, "make_factory", make)
    monkeypatch.setattr(ProgressService, "discard_operation", discard)
    return created, discarded


def test_pause_resume_keeps_one_factory_and_updates_the_same_bar(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    attempts: list[int] = []
    owner_thread = get_ident()

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
            controller.pause()
            yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
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
        controller.resume(devices=lambda: DeviceSnapshot({}))
    harness.background.deliver()
    assert not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("example") == bars
    engine = controller.engine
    assert engine is not None
    engine.update_tunables(
        "run",
        (TunableChange("value", 2),),
        expected_revision=0,
        actor=Actor("user", "offline"),
    )

    def capture() -> DeviceSnapshot:
        assert harness.gate.snapshot()
        assert get_ident() == owner_thread
        return DeviceSnapshot({})

    second = controller.resume(devices=capture)
    assert second != token
    harness.background.execute()
    harness.owner.pump_all()
    assert model.n == 2
    assert harness.progress.bars_for_owner("example") == ((handle, model),)
    harness.background.deliver()
    assert progress_calls == ([token], [])

    third = controller.resume(devices=capture)
    harness.complete_segment()
    terminal = controller.status()
    assert terminal is not None and terminal.lifecycle == "done"
    assert attempts == [1, 2, 2]
    assert len(rig.initialized) == 1
    assert progress_calls == ([token], [token])
    assert harness.progress.bars_for_owner("example") == ()
    assert harness.handles.known_outcome(third) == OperationOutcome("finished")


@pytest.mark.parametrize("ending", ["stop", "failure"])
def test_paused_run_discards_start_progress_on_stop_or_resumed_failure(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
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
            controller.pause()
            yield from env.wait_until(env.started_at)
        raise ValueError("resumed workflow failed")

    token = controller.start(
        step, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()
    assert harness.progress.bars_for_owner("example")
    if ending == "stop":
        status = controller.stop()
        assert status is not None and status.lifecycle == "stopped"
        assert attempts == 1
        assert len(finished) == 2 and finished[-1] == (status, None)
        controller.stop()
        assert len(finished) == 2
    else:
        resumed = controller.resume(devices=lambda: DeviceSnapshot({}))
        harness.background.execute()
        harness.owner.pump_all()
        harness.background.deliver()
        status = controller.status()
        assert status is not None and status.lifecycle == "failed"
        assert isinstance(finished[-1][1], ValueError)
        assert str(finished[-1][1]) == "resumed workflow failed"
        outcome = harness.handles.known_outcome(resumed)
        assert outcome is not None and outcome.status == "failed"
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner("example")
    assert not harness.gate.snapshot()
    assert controller.operation_id is None


@pytest.mark.parametrize("failure", ["lease", "capture", "submit"])
def test_resume_admission_failure_keeps_paused_run_and_progress(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
    failure: str,
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
            controller.pause()
            yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()
    before = controller.status()
    bars = harness.progress.bars_for_owner("example")
    captured: list[str] = []

    def capture() -> DeviceSnapshot:
        assert harness.gate.snapshot()
        captured.append(failure)
        if failure == "capture":
            raise ValueError("cached snapshot unavailable")
        return DeviceSnapshot({})

    if failure == "lease":
        harness.gate.register(
            999,
            "device_setup",
            owner_id="flux",
            origin_kind=harness.bus.current_origin.kind,
            note="device setup owns hardware",
            resource_id="flux",
        )
        expected = OperationConflictError
    elif failure == "submit":
        harness.background.fail_submit = True
        expected = OSError
    else:
        expected = ValueError
    try:
        with pytest.raises(expected):
            controller.resume(devices=capture)
        assert controller.status() == before
        assert controller.operation_id is None
        assert harness.handles.live_count() == 0
        assert harness.progress.bars_for_owner("example") == bars
        assert progress_calls == ([token], [])
        assert captured == ([] if failure == "lease" else [failure])
    finally:
        if failure == "lease":
            harness.gate.release(999)
        harness.background.fail_submit = False
    resumed = controller.resume(devices=lambda: DeviceSnapshot({}))
    harness.background.execute()
    harness.background.deliver()
    assert harness.handles.known_outcome(resumed) == OperationOutcome("finished")
    assert progress_calls == ([token], [token])
    assert attempts == 2 and not harness.gate.snapshot()


def test_stop_during_resume_capture_does_not_restart_paused_step(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
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
        controller.pause()
        yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step, plan=Plan(), tunables=Tunables(), paths=rig.paths, identity=identity()
    )
    harness.background.execute()
    harness.owner.pump_all()
    harness.background.deliver()

    def capture() -> DeviceSnapshot:
        controller.stop()
        return DeviceSnapshot({})

    resumed = controller.resume(devices=capture)
    assert harness.gate.snapshot()
    harness.background.execute()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert attempts == ["step"]
    assert progress_calls == ([token], [token])
    assert harness.handles.known_outcome(resumed) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot()


def test_pause_resume_reject_pending_start_without_accepting_native_run(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    with pytest.raises(RuntimeError, match="accepted"):
        controller.pause()
    with pytest.raises(RuntimeError, match="paused"):
        controller.resume(devices=lambda: DeviceSnapshot({}))
    controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
    )
    with pytest.raises(RuntimeError, match="accepted"):
        controller.pause()
    with pytest.raises(RuntimeError, match="paused"):
        controller.resume(devices=lambda: DeviceSnapshot({}))
    assert not rig.paths.metadata_root.exists()
    controller.stop()
    harness.background.execute()
    harness.background.deliver()
    assert not harness.gate.snapshot()
