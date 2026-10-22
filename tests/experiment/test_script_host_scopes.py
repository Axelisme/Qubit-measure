"""Scoped cancellation through the shared public host, with synthetic work."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from zcu_tools.experiment.scope import Scope
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost

if TYPE_CHECKING:
    from zcu_tools.experiment.stop_signal import CancelResult


@pytest.mark.parametrize(
    "scope", ["acquire", "schedule", "operation", "run", "iteration"]
)
def test_cancel_scope_preserves_parent_and_next_iteration_starts_clean(
    scope: str,
) -> None:
    host = ScriptHost(Scope("workflow", name="test"))
    acquisitions: list[bool] = []
    completions: list[CallResult[bool]] = []
    acknowledgements: list[CancelResult] = []

    def first(ctx: CallContext) -> bool:
        schedule = ctx.scope.branch("schedule")
        acquire = schedule.branch("acquire")
        try:
            acknowledgements.append(host.cancel(scope=scope))
            acquisitions.append(acquire.cancelled)
        finally:
            acquire.release()
        second = schedule.branch("acquire")
        acquisitions.append(second.cancelled)
        second.release()
        schedule.release()
        return ctx.scope.cancelled

    async def script() -> None:
        with host.root.branch("iteration") as iteration:
            completions.append(await host.call(first, scope=iteration))
            completions.append(
                await host.call(lambda ctx: ctx.scope.cancelled, scope=iteration)
            )
        with host.root.branch("iteration") as iteration:
            completions.append(
                await host.call(lambda ctx: ctx.scope.cancelled, scope=iteration)
            )

    host.run(script())
    assert acknowledgements[0] is not None, "cancel must return a typed acknowledgement"
    assert asdict(acknowledgements[0]) == {
        "cancelled": True,
        "reason": "scope cancellation requested",
        "candidates": [],
        "active_path": "workflow:test/iteration#1/operation#1/schedule#1/acquire#1",
    }
    assert acquisitions == [True, scope != "acquire"]
    assert [(item.partial_result, item.stopped) for item in completions] == [
        (True, scope in {"operation", "run", "iteration", "workflow"}),
        (scope in {"iteration", "workflow"}, scope in {"iteration", "workflow"}),
        (scope == "workflow", scope == "workflow"),
    ]
    assert host.status == ("cancelled" if scope == "workflow" else "completed")
    assert host.active_path == "workflow:test"


def test_stale_segment_and_expected_refuse_current_acquire_without_stopping_parent() -> (
    None
):
    host = ScriptHost(Scope("workflow", name="test"))

    def operation(ctx: CallContext) -> None:
        schedule = ctx.scope.branch("schedule")
        old = schedule.branch("acquire")
        old.release()
        current = schedule.branch("acquire")
        try:
            path = "workflow:test/operation#1/schedule#1/acquire#2"
            for scope, expected in [("acquire#1", None), ("acquire", 1)]:
                result = host.cancel(scope=scope, expected=expected)
                assert asdict(result) == {
                    "cancelled": False,
                    "reason": "sequence mismatch",
                    "candidates": [],
                    "active_path": path,
                }
                assert not current.cancelled
            accepted = host.cancel(scope="acquire#2", expected=2)
            assert accepted.cancelled
            assert current.cancelled
            assert not schedule.cancelled
            assert not ctx.scope.cancelled
        finally:
            current.release()
            schedule.release()

    async def script() -> None:
        result = await host.call(operation)
        assert result.partial_result
        assert not result.stopped

    host.run(script())


def test_root_has_no_sequence_and_inactive_label_returns_current_path() -> None:
    host = ScriptHost(Scope("workflow", name="test"))

    async def script() -> None:
        for scope, expected in [("workflow#1", None), ("workflow", 1)]:
            result = host.cancel(scope=scope, expected=expected)
            assert asdict(result) == {
                "cancelled": False,
                "reason": "sequence mismatch",
                "candidates": [],
                "active_path": "workflow:test",
            }
        result = host.cancel(scope="acquire")
        assert asdict(result) == {
            "cancelled": False,
            "reason": "scope not active",
            "candidates": [],
            "active_path": "workflow:test",
        }

    host.run(script())
    assert host.status == "completed"


@pytest.mark.parametrize("expected", [0, -1, True, 1.5, "1"])
def test_invalid_expected_is_a_value_error(expected: int) -> None:
    host = ScriptHost(Scope("workflow", name="test"))
    with pytest.raises(ValueError, match="expected"):
        host.cancel(scope="acquire", expected=expected)


def test_scope_validation_rejects_bad_labels_sequences_and_expected_values() -> None:
    host = ScriptHost(Scope("workflow", name="test"))

    async def script() -> None:
        # First establish the new expected ABI; bad inputs must then fail as values.
        assert not host.cancel(scope="acquire", expected=1).cancelled
        for scope in [
            "",
            "acquire#0",
            "acquire#-1",
            "acquire#x",
        ]:
            with pytest.raises(ValueError, match="scope|label"):
                host.cancel(scope=scope)
        with pytest.raises(ValueError, match="sequence|expected"):
            host.cancel(scope="acquire#2", expected=1)

    host.run(script())
