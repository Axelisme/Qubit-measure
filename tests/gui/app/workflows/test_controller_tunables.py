"""Controller tunables contracts through public native/GUI seams."""

from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    EnginePorts,
    RunStatus,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.session.operation_handles import OperationOutcome

from tests.experiment.workflows._engine_fakes import (
    Plan,
    Rig,
    State,
    Tunables,
)

from ._controller_support import create_binding, create_controller, identity
from ._start_fakes import StartHarness


def test_idle_tunables_failure_finishes_once_and_discards_retained_progress(
    tmp_path: Path,
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
    assert not harness.progress.bars_for_owner("example")
    assert controller.stop("run") == failed and len(finished) == 2
    assert harness.background.work is None and not harness.gate.snapshot()


def test_active_tunables_failure_waits_for_producer_and_preserves_native_outcome(
    tmp_path: Path,
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
    assert finished == []
    assert harness.progress.bars_for_operation(token)
    assert binding.accepted_input.data == binding.state.base.data == {"value": 1}
    assert binding.state.input.data == {"value": 2}
    harness.background.deliver()
    assert finished == [(failed, errors[0])]
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    assert controller.operation_id is None and not harness.gate.snapshot()
    assert not harness.progress.bars_for_owner("example")
    assert controller.stop("run") == failed and len(finished) == 1


@pytest.mark.parametrize("view_error", [False, True])
def test_resume_failed_tunables_unwind_finishes_once_without_a_producer(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    view_error: bool,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    native_errors: list[IsADirectoryError] = []
    observed: list[tuple[bool, bool, int]] = []

    def completed(status: RunStatus | None, error: Exception | None) -> None:
        finished.append((status, error))
        if view_error and status is not None and status.lifecycle == "failed":
            raise ValueError("failed view notification")

    controller = create_controller(harness, rig, completed)

    @workflow("failed-resume-admission", plan=Plan, tunables=Tunables, init=rig.init)
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
        ports=EnginePorts(rig.plots, rig.progress, rig.clock, context=7),
        binding=create_binding(),
        paths=rig.paths,
        identity=identity(),
    )
    harness.complete_segment()
    journal = rig.paths.metadata_root / "journal.jsonl"
    original = journal.read_bytes()
    journal.unlink()
    journal.mkdir()

    def during_submit() -> None:
        try:
            controller.apply_tunables(
                "run", {("value",): 2}, actor=Actor("agent", "rpc")
            )
        except IsADirectoryError as error:
            native_errors.append(error)
        controller.stop("run")
        observed.append(
            (
                bool(harness.gate.snapshot()),
                controller.has_pending_segment,
                len(finished),
            )
        )

    harness.background.on_submit = during_submit
    harness.background.fail_submit = True
    try:
        with pytest.raises(OSError, match="submit failed"):
            controller.resume("run")
    finally:
        harness.background.on_submit = None
        journal.rmdir()
        journal.write_bytes(original)
    assert len(native_errors) == 1 and observed == [(True, True, 1)]
    failed = controller.status()
    assert failed is not None and failed.lifecycle == "failed" and failed.revision == 0
    assert finished[-1] == (failed, native_errors[0]) and len(finished) == 2
    assert harness.handles.known_outcome(token) == OperationOutcome(
        "failed", failed.reason
    )
    assert harness.background.work is None and controller.operation_id is None
    assert not harness.gate.snapshot() and harness.handles.live_count() == 0
    assert not harness.progress.bars_for_owner("example")
    assert controller.stop("run") == failed and len(finished) == 2
    assert ("failed view notification" in caplog.text) == view_error
