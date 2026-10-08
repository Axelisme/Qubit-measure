"""Native startup, cancellation and lease cleanup through the public controller."""

from collections.abc import Callable
from pathlib import Path
from threading import get_ident
from time import monotonic

import pytest
from zcu_tools.datafile import SoftwareProvenance
from zcu_tools.experiment.workflows import (
    DeviceSnapshot,
    Done,
    EnginePorts,
    InvalidRunState,
    RunIdentity,
    RunStatus,
    Step,
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
        harness.runner, ports, "example", on_finished=lambda _status, _error: None
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
