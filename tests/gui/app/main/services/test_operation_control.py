"""OperationControlFacet public contract tests."""

from __future__ import annotations

from zcu_tools.gui.app.main.services.operation_control import OperationControlFacet
from zcu_tools.gui.session.operation_handles import AwaitResult

from tests.gui._control_fakes import CallLog, call


class RecordingHandles:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.result = AwaitResult(reason="timeout")

    def await_outcome(self, operation_id: int, timeout: float) -> AwaitResult:
        self._log.add("handles", "await_outcome", operation_id, timeout)
        return self.result


class RecordingProgress:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.bars = (("bar", object()),)

    def bars_for_operation(self, operation_id: int) -> tuple:
        self._log.add("progress", "bars_for_operation", operation_id)
        return self.bars


def test_operation_control_routes_await_to_handles() -> None:
    log = CallLog()
    handles = RecordingHandles(log)
    facet = OperationControlFacet(handles=handles, progress=RecordingProgress(log))

    assert facet.await_operation(7, 0.5) is handles.result

    assert log.calls == [call("handles", "await_outcome", 7, 0.5)]


def test_operation_control_routes_progress_to_progress_service() -> None:
    log = CallLog()
    progress = RecordingProgress(log)
    facet = OperationControlFacet(handles=RecordingHandles(log), progress=progress)

    assert facet.get_operation_progress(9) is progress.bars

    assert log.calls == [call("progress", "bars_for_operation", 9)]
