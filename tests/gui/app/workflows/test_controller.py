"""Controller contracts through public native/GUI seams."""

from pathlib import Path

import pytest
from zcu_tools.experiment.workflows import (
    EnginePorts,
    InvalidRunState,
    RunStatus,
)
from zcu_tools.gui.app.workflows.controller import WorkflowController
from zcu_tools.gui.session.operation_handles import OperationOutcome
from zcu_tools.gui.session.ports import OperationConflictError

from tests.experiment.workflows._engine_fakes import (
    ManualClock,
    Plan,
    Rig,
    Tunables,
)

from ._controller_support import create_binding, create_controller, declare, identity
from ._start_fakes import StartHarness


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


@pytest.mark.parametrize("failure", ["lease", "submit"])
def test_submit_failure_has_no_native_artifacts_or_lease(
    tmp_path: Path,
    failure: str,
) -> None:
    harness, rig = StartHarness(), Rig(tmp_path)
    controller = create_controller(harness, rig, lambda _status, _error: None)

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
        controller.resume("run")
    assert not rig.paths.metadata_root.exists()
    controller.stop("run")
    harness.complete_segment()
    assert controller.run_id == "run"
    assert not harness.gate.snapshot()
