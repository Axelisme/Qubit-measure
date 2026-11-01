"""RunService owns failed work delivery and cleanup through the Qt executor."""

import threading

import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.services.operation_gate import OperationGate
from zcu_tools.gui.app.measure.services.run import RunService
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.services.progress import ProgressService

from tests.gui._progress_fakes import DirectProgressTransport
from tests.gui._qt_waits import qt_until

from ._run_support import make_inputs, make_plots, make_state


@pytest.mark.parametrize("requested", [False, True])
def test_core_keyboard_interrupt_fails_run_without_record_and_releases_ownership(
    qapp, requested: bool
) -> None:
    state, tab_id, adapter = make_state()
    bus = BaseEventBus()
    background = BackgroundRunner()
    handles = OperationHandles()
    gate = OperationGate(bus)
    progress = ProgressService(DirectProgressTransport())
    runner = OperationRunner(gate, handles, progress, background, bus)
    service = RunService(
        state,
        runner,
        bus,
        handles,
        gate=gate,
        devices=lambda: {},
        refresh_devices=lambda: None,
    )
    entered = threading.Event()
    release = threading.Event()
    error = KeyboardInterrupt("synthetic core interrupted")
    observed: list[bool] = []

    def core(*_args: object, context: RunContext) -> None:
        entered.set()
        release.wait()
        observed.append(context.scope.stop.is_set())
        raise error

    adapter.run.side_effect = core
    try:
        token = service.start_run(
            make_inputs(state, tab_id, adapter), plots=make_plots()
        )
        qt_until(entered.is_set, label="core reached external barrier")
        assert state.is_tab_busy(tab_id)
        assert gate.snapshot()
        if requested:
            assert service.cancel_run()
        release.set()
        qt_until(
            lambda: handles.poll(token) is not None, label="failed Run owner delivery"
        )
        outcome = handles.poll(token)
        assert outcome is not None
        assert outcome.status == "failed"
        assert outcome.error == "KeyboardInterrupt('synthetic core interrupted')"
        assert not state.is_tab_busy(tab_id)
        assert gate.snapshot() == ()
        assert progress.bars_for_owner(tab_id) == ()
        pane = state.get_tab(tab_id).run
        assert pane.result is None
        assert pane.stop_requested is requested
        assert pane.source_operation_id == token
        assert observed == [requested]
    finally:
        release.set()
        assert background.quiesce(timeout_ms=30_000)
