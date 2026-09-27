"""OperationControlFacet public contract tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from zcu_tools.gui.app.main.services.operation_control import (
    ActiveOperation,
    OperationControlFacet,
)
from zcu_tools.gui.app.main.services.run_analyze_control import ActiveTabOperation
from zcu_tools.gui.event_bus import EventOrigin
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.operation_handles import (
    AwaitResult,
    OperationHandles,
    OperationOutcome,
)
from zcu_tools.gui.session.pbar_host import ProgressBarModel

from tests.gui._control_fakes import CallLog, call


class RecordingHandles:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.result = AwaitResult(reason="timeout")

    def await_known_outcome(self, operation_id: int, timeout: float) -> AwaitResult:
        self._log.add("handles", "await_known_outcome", operation_id, timeout)
        return self.result

    def elapsed_seconds(self, operation_id: int) -> float:
        self._log.add("handles", "elapsed_seconds", operation_id)
        return 0.5

    def known_outcome(self, operation_id: int) -> OperationOutcome | None:
        raise AssertionError(f"unexpected cancel lookup: {operation_id}")

    def has_cancel_hook(self, operation_id: int) -> bool:
        raise AssertionError(f"unexpected cancel lookup: {operation_id}")


class RecordingProgress:
    def __init__(self, log: CallLog) -> None:
        self._log = log
        self.bars = ((1, ProgressBarModel(label="bar", total=10, start_time=0.0)),)

    def bars_for_operation(
        self, operation_id: int
    ) -> tuple[tuple[int, ProgressBarModel], ...]:
        self._log.add("progress", "bars_for_operation", operation_id)
        return self.bars


def test_operation_control_routes_await_and_elapsed_to_handles() -> None:
    log = CallLog()
    handles = RecordingHandles(log)
    facet = OperationControlFacet(
        handles=handles,
        progress=RecordingProgress(log),
        run_analyze=cast(Any, SimpleNamespace()),
        device=cast(Any, SimpleNamespace()),
    )

    assert facet.await_operation(7, 0.5) is handles.result
    assert facet.elapsed_seconds(7) == 0.5
    assert log.calls == [
        call("handles", "await_known_outcome", 7, 0.5),
        call("handles", "elapsed_seconds", 7),
    ]


def test_active_operations_merge_tab_and_device_owners() -> None:
    log = CallLog()
    run = SimpleNamespace(
        active_tab_operations=lambda: (ActiveTabOperation(11, "gui-tab", "run"),)
    )
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


def test_cancel_by_handle_uses_owner_hook_and_preserves_other_operations() -> None:
    handles = OperationHandles()
    stopped: list[int] = []
    run_token = handles.create(
        cancel_hook=lambda: stopped.append(1), origin=EventOrigin(kind="user")
    )
    post_token = handles.create(origin=EventOrigin(kind="user"))
    device_token = handles.create(
        cancel_hook=lambda: stopped.append(2), origin=EventOrigin(kind="user")
    )
    run = SimpleNamespace(
        active_tab_operations=lambda: (
            ActiveTabOperation(run_token, "gui-run", "run"),
            ActiveTabOperation(post_token, "gui-post", "analyze"),
        ),
        cancel_run=lambda: (handles.cancel(run_token), True)[1],
    )
    device = SimpleNamespace(
        get_active_device_operations=lambda: (
            SimpleNamespace(token=device_token, device_name="bias"),
        ),
        cancel_device_operation=lambda name: handles.cancel(device_token),
    )
    facet = OperationControlFacet(
        handles=handles,
        progress=RecordingProgress(CallLog()),
        run_analyze=cast(Any, run),
        device=cast(Any, device),
    )

    assert facet.cancel_operation(run_token) == "cancelling"
    assert stopped == [1]
    with pytest.raises(FailedPreconditionError) as exc_info:
        facet.cancel_operation(post_token)
    assert exc_info.value.reason_code == "not_cancellable"
    assert handles.known_outcome(post_token) is None
    assert facet.cancel_operation(device_token) == "cancelling"
    assert stopped == [1, 2]
    handles.settle(run_token, OperationOutcome("cancelled"))
    assert facet.cancel_operation(run_token) == "cancelled"
    with pytest.raises(KeyError, match="unknown"):
        facet.cancel_operation(999)


def test_cancel_does_not_report_a_failed_operation_as_finished() -> None:
    handles = OperationHandles()
    token = handles.create(origin=EventOrigin(kind="user"))
    handles.settle(token, OperationOutcome("failed", error="ramp failed"))
    facet = OperationControlFacet(
        handles=handles,
        progress=RecordingProgress(CallLog()),
        run_analyze=cast(Any, SimpleNamespace()),
        device=cast(Any, SimpleNamespace()),
    )

    with pytest.raises(FailedPreconditionError, match="ramp failed") as exc_info:
        facet.cancel_operation(token)
    assert exc_info.value.reason_code == "operation_failed"
    assert handles.known_outcome(token) == OperationOutcome(
        "failed", error="ramp failed"
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
