"""Controller targets contracts through public native/GUI seams."""

from collections.abc import Callable
from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import (
    Actor,
    Done,
    EnginePorts,
    RunMismatch,
    RunPaths,
    RunStatus,
    Step,
    WorkflowEnv,
    workflow,
)
from zcu_tools.gui.session.events import GateChangedPayload
from zcu_tools.gui.session.operation_handles import OperationOutcome

from tests.experiment.workflows._engine_fakes import (
    Plan,
    Rig,
    State,
    Tunables,
)

from ._controller_support import (
    create_binding,
    create_controller,
    declare,
    identity,
)
from ._start_fakes import StartHarness


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

    assert controller.run_id is None
    commands: dict[str, Callable[[], object]] = {
        "pause": lambda: controller.pause("old"),
        "resume": lambda: controller.resume("old"),
        "stop": lambda: controller.stop("old"),
    }
    with pytest.raises(RunMismatch) as raised:
        commands[command]()
    assert (raised.value.expected, raised.value.actual) == ("old", None)
    assert finished == []
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

    def notified() -> None:
        seen.append(controller.run_id)
        for invoke in (
            lambda: controller.pause("old"),
            lambda: controller.resume("old"),
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
    assert seen == ["run"]
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
def test_accepted_old_controls_preserve_native_state_progress_and_journal(
    tmp_path: Path,
    phase: str,
    progress_calls: tuple[list[int], list[int]],
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    finished: list[tuple[RunStatus | None, Exception | None]] = []
    controller = create_controller(
        harness, rig, lambda status, error: finished.append((status, error))
    )
    checked: list[str] = []

    def reject_old_controls() -> None:
        before = controller.status()
        assert before is not None and before.lifecycle == phase
        # Lease age advances on each read; compare stable presence fields.
        gate = tuple((p.kind, p.origin_kind, p.note) for p in harness.gate.snapshot())
        bars = harness.progress.bars_for_owner("example")
        events = rig.events()
        for invoke in (
            lambda: controller.pause("old"),
            lambda: controller.resume("old"),
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
        assert progress_calls[1] == []
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
