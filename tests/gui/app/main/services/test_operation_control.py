"""OperationControlFacet public contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

from zcu_tools.gui.app.main.services.operation_control import (
    ActiveOperation,
    OperationControlFacet,
)
from zcu_tools.gui.app.main.services.run_analyze_control import ActiveTabOperation
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
    facet = OperationControlFacet(
        handles=handles,
        progress=RecordingProgress(log),
        run_analyze=cast(Any, SimpleNamespace()),
        device=cast(Any, SimpleNamespace()),
    )

    assert facet.await_operation(7, 0.5) is handles.result

    assert log.calls == [call("handles", "await_outcome", 7, 0.5)]


def test_active_operations_merge_tab_and_device_owners() -> None:
    log = CallLog()
    run = SimpleNamespace(active_tab_operations=lambda: (ActiveTabOperation(11, "gui-tab", "run"),))
    device = SimpleNamespace(
        get_active_device_operations=lambda: (SimpleNamespace(token=12),)
    )
    facet = OperationControlFacet(
        handles=RecordingHandles(log),
        progress=RecordingProgress(log),
        run_analyze=cast(Any, run),
        device=cast(Any, device),
    )

    assert facet.active_operations() == (
        ActiveOperation(11, "gui-tab", "run"),
        ActiveOperation(12, None, "device"),
    )


def test_operation_control_routes_progress_to_progress_service() -> None:
    log = CallLog()
    progress = RecordingProgress(log)
    facet = OperationControlFacet(
        handles=RecordingHandles(log),
        progress=progress,
        run_analyze=cast(Any, SimpleNamespace()),
        device=cast(Any, SimpleNamespace()),
    )

    assert facet.get_operation_progress(9) is progress.bars

    assert log.calls == [call("progress", "bars_for_operation", 9)]
