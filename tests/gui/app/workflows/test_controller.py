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
    RunMismatch,
    RunPaths,
    RunStatus,
    Step,
    TunableChange,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.app.workflows.tunables import TunablesHost
from zcu_tools.gui.form import FormEnv, FormModel, FormState
from zcu_tools.gui.session.adapters.thread_pool_background import (
    ThreadPoolBackgroundExecutor,
)
from zcu_tools.gui.session.events import GateChangedPayload
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


def identity(run_id: str = "run") -> RunIdentity:
    return RunIdentity(
        run_id,
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


class ValueForm(FormModel):
    """Ordinary input corresponding to the example workflow tunable."""

    value: int = 1


def create_binding(run_id: str = "run") -> TunablesHost[ValueForm, int]:
    env = FormEnv()
    return TunablesHost(FormState(ValueForm.default_input(env)), env, run_id)


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


def test_target_check_is_available_before_acceptance_without_changing_admission(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    with pytest.raises(RunMismatch) as absent:
        controller.require_run_id("run")
    assert (absent.value.expected, absent.value.actual) == ("run", None)
    assert not harness.gate.snapshot()
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
    controller.require_run_id("run")
    with pytest.raises(RunMismatch) as stale:
        controller.require_run_id("old")
    assert (stale.value.expected, stale.value.actual) == ("old", "run")
    assert controller.operation_id == token and controller.status() is None
    assert harness.gate.snapshot() and not rig.paths.metadata_root.exists()
    harness.complete_segment()
    before = controller.status()
    controller.require_run_id("run")
    assert controller.status() == before
    assert not harness.gate.snapshot()


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
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    assert controller.operation_id == token
    assert controller.run_id == "run"
    assert controller.engine is controller.status() is None
    assert not rig.paths.metadata_root.exists()
    assert harness.handles.live_count() == 1
    assert len(harness.gate.snapshot()) == 1
    with pytest.raises(InvalidRunState):
        controller.start(
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


@pytest.mark.parametrize("failure", ["lease", "factory", "submit"])
def test_submit_failure_has_no_native_artifacts_or_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)

    def reject_factory(_token: int, _owner: str) -> BoundProgressFactory:
        raise OSError("factory failed")

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
    elif failure == "factory":
        monkeypatch.setattr(harness.progress, "make_factory", reject_factory)
        expected = OSError
    else:
        harness.background.fail_submit = True
        expected = OSError
    with pytest.raises(expected):
        controller.start(
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
    assert controller.status() is None
    assert controller.run_id is None
    assert controller.operation_id is None
    assert harness.handles.live_count() == 0
    if failure == "lease":
        harness.gate.release(999)
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
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.handles.cancel(token)
    assert controller.stop("run") is None
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
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=create_binding(),
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
                controller.stop("run")
            return super().now()

    ports = EnginePorts(rig.plots, rig.progress, StoppingClock(), context=7)
    controller = WorkflowController(
        harness.runner,
        "example",
        on_finished=lambda _status, _error: None,
    )
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        ports=ports,
        binding=create_binding(),
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


def test_terminal_view_can_start_replacement_without_old_cleanup_touching_it(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    ports = EnginePorts(rig.plots, rig.progress, rig.clock, context=7)
    pending = create_binding("next")
    receipts: list[int] = []
    observed: list[
        tuple[RunStatus | None, Exception | None, OperationOutcome | None, bool, bool]
    ] = []

    def completed(status: RunStatus | None, error: Exception | None) -> None:
        if status is None or status.run_id != "run":
            return
        observed.append(
            (
                status,
                error,
                harness.handles.known_outcome(token),
                not harness.gate.snapshot(),
                controller.operation_id is None,
            )
        )
        receipts.append(
            controller.start(
                declare(rig),
                plan=Plan(),
                tunables=Tunables(),
                paths=RunPaths(tmp_path / "next-metadata", tmp_path / "next-data"),
                identity=identity("next"),
                ports=ports,
                binding=pending,
            )
        )
        raise ValueError("old view failed after replacement")

    controller = create_controller(harness, rig, completed)
    token = controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
        ports=ports,
        binding=create_binding(),
    )
    harness.complete_segment()
    assert len(observed) == len(receipts) == 1
    status, error, outcome, released, detached = observed[0]
    assert status is not None and status.lifecycle == "done"
    assert error is None and outcome == OperationOutcome("finished")
    assert released and detached
    replacement = receipts[0]
    assert replacement != token and controller.operation_id == replacement
    assert controller.binding is pending and pending.engine is None
    assert controller.run_id == "next" and controller.has_pending_segment
    assert harness.gate.snapshot() and harness.handles.live_count() == 1
    assert progress_calls == ([token, replacement], [token])
    assert "old view failed after replacement" in caplog.text
    harness.complete_segment()
    assert controller.status() is not None
    assert harness.handles.known_outcome(token) == OperationOutcome("finished")
    assert harness.handles.known_outcome(replacement) == OperationOutcome("finished")
    assert progress_calls == ([token, replacement], [token, replacement])
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0


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
        controller.resume("run", devices=lambda: DeviceSnapshot({}))
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

    def capture() -> DeviceSnapshot:
        assert harness.gate.snapshot()
        assert get_ident() == owner_thread
        return DeviceSnapshot({})

    second = controller.resume("run", devices=capture)
    assert second == token
    harness.background.execute()
    harness.owner.pump_all()
    assert model.n == 2
    assert harness.progress.bars_for_owner("example") == ((handle, model),)
    harness.background.deliver()
    assert progress_calls == ([token], [])

    third = controller.resume("run", devices=capture)
    harness.complete_segment()
    terminal = controller.status()
    assert terminal is not None and terminal.lifecycle == "done"
    assert attempts == [1, 2, 2]
    assert len(rig.initialized) == 1
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner(
        "example"
    ) and harness.handles.known_outcome(third) == OperationOutcome("finished")


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
            controller.pause("run")
            yield from env.wait_until(env.started_at)
        raise ValueError("resumed workflow failed")

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
    assert harness.progress.bars_for_owner("example")
    if ending == "stop":
        status = controller.stop("run")
        assert status is not None and status.lifecycle == "stopped"
        assert attempts == 1
        assert len(finished) == 2 and finished[-1] == (status, None)
        controller.stop("run")
        assert len(finished) == 2
    else:
        resumed = controller.resume("run", devices=lambda: DeviceSnapshot({}))
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
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner("example")
    assert not harness.gate.snapshot()
    assert controller.operation_id is None


def test_idle_tunables_failure_finishes_once_and_discards_retained_progress(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    repeated: list[RunStatus | None] = []

    def completed(status: RunStatus | None, error: Exception | None) -> None:
        finished.append((status, error))
        if status is not None and status.lifecycle == "failed":
            repeated.append(controller.stop("run"))

    controller = create_controller(harness, rig, completed)

    @workflow("idle-failure", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun, state
        env.pbar("retained", 1).refresh()
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
    engine = controller.engine
    assert engine is not None and controller.operation_id == token
    assert finished[0][0] is not None and finished[0][0].lifecycle == "paused"
    journal = rig.paths.metadata_root / "journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()
    try:
        with pytest.raises(IsADirectoryError):
            controller.apply_tunables(
                "run", {("value",): 2}, actor=Actor("agent", "test")
            )
    finally:
        journal.rmdir()
        journal.write_bytes(original)
    failed = controller.status()
    assert failed is not None and failed.lifecycle == "failed" and failed.revision == 0
    assert len(finished) == 2
    assert finished[0][0] is not None and finished[0][0].lifecycle == "paused"
    assert finished[-1][0] == failed and repeated == [failed]
    assert isinstance(finished[-1][1], IsADirectoryError)
    assert not harness.gate.snapshot()
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    assert controller.stop("run") == failed
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner("example")
    assert controller.stop("run") == failed and len(finished) == 2
    assert progress_calls == ([token], [token])
    assert harness.background.work is None and not harness.gate.snapshot()


def test_active_tunables_failure_waits_for_producer_and_preserves_native_outcome(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    errors: list[IsADirectoryError] = []
    observed: list[tuple[bool, bool, int | None, int]] = []
    binding = create_binding()
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    journal = rig.paths.metadata_root / "journal.jsonl"

    @workflow("active-failure", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int], plan: Plan, tun: Tunables, state: State
    ) -> Step[State, int]:
        del plan, tun, state
        env.pbar("retained", 1).refresh()
        original = journal.read_bytes()
        journal.unlink()
        journal.mkdir()
        try:
            try:
                controller.apply_tunables(
                    "run", {("value",): 2}, actor=Actor("agent", "test")
                )
            except IsADirectoryError as error:
                errors.append(error)
        finally:
            journal.rmdir()
            journal.write_bytes(original)
        controller.stop("run")
        observed.append(
            (
                bool(harness.gate.snapshot()),
                harness.handles.known_outcome(token) is None,
                controller.operation_id,
                len(finished),
            )
        )
        yield from env.wait_until(env.started_at)
        return Done()

    token = controller.start(
        step,
        plan=Plan(),
        tunables=Tunables(),
        ports=EnginePorts(
            rig.plots, rig.progress, rig.clock, context=7, devices=rig.devices
        ),
        binding=binding,
        paths=rig.paths,
        identity=identity(),
    )
    harness.background.execute()
    harness.owner.pump_all()
    failed = controller.status()
    assert len(errors) == 1
    assert failed is not None and failed.lifecycle == "failed" and failed.revision == 0
    assert observed == [(True, True, token, 0)]
    assert controller.operation_id == token and harness.gate.snapshot()
    assert finished == [] and progress_calls == ([token], [])
    assert harness.progress.bars_for_operation(token)
    assert binding.accepted_input.data == binding.state.base.data == {"value": 1}
    assert binding.state.input.data == {"value": 2}
    harness.background.deliver()
    assert finished == [(failed, errors[0])]
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    assert controller.operation_id is None and not harness.gate.snapshot()
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner("example")
    assert controller.stop("run") == failed and len(finished) == 1
    assert progress_calls == ([token], [token])


@pytest.mark.parametrize("failure", ["capture", "submit"])
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
    captured: list[str] = []

    def capture() -> DeviceSnapshot:
        assert harness.gate.snapshot()
        captured.append(failure)
        if failure == "capture":
            raise ValueError("cached snapshot unavailable")
        return DeviceSnapshot({})

    if failure == "submit":
        harness.background.fail_submit = True
        expected = OSError
    else:
        expected = ValueError
    try:
        with pytest.raises(expected):
            controller.resume("run", devices=capture)
        assert controller.status() == before
        assert controller.operation_id == token
        assert harness.handles.live_count() == 1
        assert harness.handles.known_outcome(token) is None
        assert harness.gate.snapshot()
        assert harness.progress.bars_for_owner("example") == bars
        assert progress_calls == ([token], [])
        assert captured == [failure]
    finally:
        harness.background.fail_submit = False
    resumed = controller.resume("run", devices=lambda: DeviceSnapshot({}))
    harness.background.execute()
    harness.background.deliver()
    assert harness.handles.known_outcome(resumed) == OperationOutcome("finished")
    assert progress_calls == ([token], [token])


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

    def capture() -> DeviceSnapshot:
        controller.stop("run")
        return DeviceSnapshot({})

    resumed = controller.resume("run", devices=capture)
    assert harness.gate.snapshot()
    harness.background.execute()
    harness.background.deliver()
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert attempts == ["step"]
    assert progress_calls == ([token], [token])
    assert harness.handles.known_outcome(resumed) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot()


def test_resume_capture_stop_waits_for_owner_terminal_delivery(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
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

    def capture() -> DeviceSnapshot:
        controller.stop("run")
        notifications.append(
            (
                len(finished),
                bool(harness.progress.bars_for_owner("example")),
                controller.has_pending_segment,
            )
        )
        return DeviceSnapshot({})

    resumed = controller.resume("run", devices=capture)
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
    assert progress_calls == ([token], [token])
    assert harness.handles.known_outcome(resumed) == OperationOutcome("cancelled")
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    controller.stop("run")
    assert len(finished) == 2


@pytest.mark.parametrize("failure", ["capture", "submit"])
def test_resume_stop_rejection_finishes_without_a_producer(
    tmp_path: Path,
    progress_calls: tuple[list[int], list[int]],
    failure: str,
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
    capture_error = ValueError("cached snapshot failed after Stop")

    def capture() -> DeviceSnapshot:
        controller.stop("run")
        if failure == "capture":
            raise capture_error
        return DeviceSnapshot({})

    harness.background.fail_submit = failure == "submit"
    with pytest.raises(ValueError if failure == "capture" else OSError) as raised:
        controller.resume("run", devices=capture)
    if failure == "capture":
        assert raised.value is capture_error
    else:
        assert str(raised.value) == "submit failed"
    status = controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert len(finished) == 2 and finished[-1] == (status, None)
    assert progress_calls == ([token], [token])
    assert not harness.progress.bars_for_owner("example")
    assert not controller.has_pending_segment
    assert controller.operation_id is None and harness.handles.live_count() == 0
    assert not harness.gate.snapshot() and harness.background.work is None
    assert attempts == ["step"]
    controller.stop("run")
    assert len(finished) == 2 and progress_calls == ([token], [token])


def test_pause_resume_reject_pending_start_without_accepting_native_run(
    tmp_path: Path,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    controller.start(
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
    with pytest.raises(RuntimeError, match="accepted"):
        controller.pause("run")
    with pytest.raises(RuntimeError, match="paused"):
        controller.resume("run", devices=lambda: DeviceSnapshot({}))
    assert not rig.paths.metadata_root.exists()
    controller.stop("run")
    harness.complete_segment()
    assert controller.run_id == "run"
    assert not harness.gate.snapshot()


@pytest.mark.parametrize("command", ["pause", "resume", "stop"])
def test_control_without_attached_target_rejects_before_side_effects(
    tmp_path: Path,
    command: str,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    captured: list[str] = []

    def capture() -> DeviceSnapshot:
        captured.append("devices")
        return DeviceSnapshot({})

    assert controller.run_id is None
    commands: dict[str, Callable[[], object]] = {
        "pause": lambda: controller.pause("old"),
        "resume": lambda: controller.resume("old", devices=capture),
        "stop": lambda: controller.stop("old"),
    }
    with pytest.raises(RunMismatch) as raised:
        commands[command]()
    assert (raised.value.expected, raised.value.actual) == ("old", None)
    assert captured == finished == []
    assert harness.handles.live_count() == 0 and not harness.gate.snapshot()
    # A rejected unattached Stop must not poison the next real Start.
    controller.start(
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
    harness.complete_segment()
    status = controller.status()
    assert status is not None and status.lifecycle == "done"


@pytest.mark.parametrize("notification", ["gate", "progress"])
def test_pending_target_is_available_to_notifications_and_rejects_old_controls(
    tmp_path: Path,
    notification: str,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    seen: list[str | None] = []
    captured: list[str] = []

    def capture() -> DeviceSnapshot:
        captured.append("devices")
        return DeviceSnapshot({})

    def notified() -> None:
        seen.append(controller.run_id)
        for invoke in (
            lambda: controller.pause("old"),
            lambda: controller.resume("old", devices=capture),
            lambda: controller.stop("old"),
        ):
            with pytest.raises(RunMismatch) as raised:
                invoke()
            assert (raised.value.expected, raised.value.actual) == ("old", "run")
        assert controller.status() is None and finished == []
        assert harness.gate.snapshot()
        assert progress_calls[1] == []

    def registered(payload: GateChangedPayload) -> None:
        if payload.active:
            notified()

    if notification == "gate":
        detach = harness.bus.subscribe(GateChangedPayload, registered).unsubscribe
    else:
        detach = harness.progress.attach_by_owner("example", notified)
    try:
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
    finally:
        detach()
    assert seen == ["run"] and captured == []
    assert not rig.paths.metadata_root.exists()
    assert not rig.paths.data_root.exists()
    assert progress_calls == ([token], [])
    harness.complete_segment()
    status = controller.status()
    assert status is not None and status.lifecycle == "done"
    assert finished == [(status, None)]
    assert rig.initialized
    assert harness.handles.known_outcome(token) == OperationOutcome("finished")
    assert controller.run_id == "run"
    assert progress_calls == ([token], [token])


@pytest.mark.parametrize("notification", ["gate", "progress"])
def test_replacement_notifications_observe_new_pending_host_without_editing_old_draft(
    tmp_path: Path, notification: str
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)
    ports = EnginePorts(rig.plots, rig.progress, rig.clock, context=7)
    old = create_binding()
    controller.start(
        declare(rig),
        plan=Plan(),
        tunables=Tunables(),
        paths=rig.paths,
        identity=identity(),
        ports=ports,
        binding=old,
    )
    harness.complete_segment()
    old.state.write({("value",): 9}, old.env)
    old_input = old.state.input.data
    pending = create_binding("next")
    observed: list[tuple[bool, int | None, bool]] = []
    errors: list[str] = []

    def observe() -> None:
        if controller.run_id != "next":
            return
        observed.append(
            (
                controller.binding is pending,
                controller.operation_id,
                controller.has_pending_segment,
            )
        )
        try:
            controller.apply_tunables(
                "next", {("value",): 3}, actor=Actor("agent", "rpc")
            )
        except RuntimeError as error:
            errors.append(str(error))
        controller.stop("next")

    def registered(payload: GateChangedPayload) -> None:
        if payload.active:
            observe()

    detach = (
        harness.bus.subscribe(GateChangedPayload, registered).unsubscribe
        if notification == "gate"
        else harness.progress.attach_by_owner("example", observe)
    )
    try:
        paths = RunPaths(tmp_path / "next-metadata", tmp_path / "next-data")
        token = controller.start(
            declare(rig),
            plan=Plan(),
            tunables=Tunables(),
            paths=paths,
            identity=identity("next"),
            ports=ports,
            binding=pending,
        )
    finally:
        detach()
    harness.complete_segment()
    assert observed == [(True, token, True)]
    assert len(errors) == 1 and "acceptance" in errors[0]
    assert old.state.input.data == old_input
    assert pending.state.input.data == {"value": 1}
    assert pending.engine is None
    assert not paths.metadata_root.exists()
    assert harness.handles.known_outcome(token) == OperationOutcome("cancelled")


@pytest.mark.parametrize("phase", ["running", "pausing", "paused"])
def test_accepted_old_controls_preserve_native_state_progress_and_capture(
    tmp_path: Path,
    phase: str,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    captured: list[str] = []
    checked: list[str] = []

    def capture() -> DeviceSnapshot:
        captured.append("devices")
        return DeviceSnapshot({})

    def reject_old_controls() -> None:
        before = controller.status()
        assert before is not None and before.lifecycle == phase
        # Lease age advances on each read; compare stable presence fields.
        gate = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
        bars = harness.progress.bars_for_owner("example")
        events = rig.events()
        for invoke in (
            lambda: controller.pause("old"),
            lambda: controller.resume("old", devices=capture),
            lambda: controller.stop("old"),
        ):
            with pytest.raises(RunMismatch) as raised:
                invoke()
            assert (raised.value.expected, raised.value.actual) == ("old", "run")
            assert controller.status() == before
            assert (
                tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
                == gate
            )
            assert harness.progress.bars_for_owner("example") == bars
            assert rig.events() == events
        assert captured == [] and progress_calls[1] == []
        checked.append(phase)

    @workflow("target-controls", plan=Plan, tunables=Tunables, init=rig.init)
    def step(
        env: WorkflowEnv[int],
        plan: Plan,
        tun: Tunables,
        state: State,
    ) -> Step[State, int]:
        del plan, tun, state
        env.pbar("work", 2).refresh()
        if phase == "pausing":
            controller.pause("run")
        if phase != "paused":
            reject_old_controls()
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
    assert harness.background.error is None
    if phase == "paused":
        reject_old_controls()
    assert checked == [phase]
    assert controller.run_id == "run"
    stopped = controller.stop("run")
    assert stopped is not None and stopped.lifecycle == "stopped"
    assert progress_calls == ([token], [token])
    assert not harness.gate.snapshot()
    assert harness.progress.bars_for_owner("example") == ()
