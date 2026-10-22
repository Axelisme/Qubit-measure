"""Scoped cancellation through the shared public host, with synthetic work."""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING

import pytest
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost

if TYPE_CHECKING:
    from zcu_tools.experiment.stop_signal import CancelResult


@pytest.mark.parametrize(
    "scope", ["acquire", "schedule", "operation", "run", "iteration", "script"]
)
def test_cancel_scope_preserves_parent_and_commit_starts_clean_iteration(
    scope: str,
) -> None:
    host = ScriptHost()
    acquisitions: list[bool] = []
    completions: list[CallResult[bool]] = []
    acknowledgements: list[CancelResult] = []

    def first(ctx: CallContext) -> bool:
        schedule = ctx.stop_signal.child(label="schedule")
        acquire = schedule.child(label="acquire")
        try:
            acknowledgements.append(host.cancel(scope=scope))
            acquisitions.append(acquire.is_set())
        finally:
            acquire.release()
        second = schedule.child(label="acquire")
        acquisitions.append(second.is_set())
        second.release()
        schedule.release()
        return ctx.stop_signal.is_set()

    async def script() -> None:
        completions.append(await host.call(first))
        completions.append(await host.call(lambda ctx: ctx.stop_signal.is_set()))
        await host.commit()
        completions.append(await host.call(lambda ctx: ctx.stop_signal.is_set()))

    host.run(script())
    assert acknowledgements[0] is not None, "cancel must return a typed acknowledgement"
    assert asdict(acknowledgements[0]) == {
        "cancelled": True,
        "reason": "scope cancellation requested",
        "active_path": "script/iteration#1/operation#1/schedule#1/acquire#1",
    }
    assert acquisitions == [True, scope != "acquire"]
    assert [(item.partial_result, item.stopped) for item in completions] == [
        (True, scope in {"operation", "run", "iteration", "script"}),
        (scope in {"iteration", "script"}, scope in {"iteration", "script"}),
        (scope == "script", scope == "script"),
    ]
    assert host.status == ("cancelled" if scope == "script" else "completed")
    assert host.active_path == "script"


def test_stale_segment_and_expected_refuse_current_acquire_without_stopping_parent() -> (
    None
):
    host = ScriptHost()

    def operation(ctx: CallContext) -> None:
        schedule = ctx.stop_signal.child(label="schedule")
        old = schedule.child(label="acquire")
        old.release()
        current = schedule.child(label="acquire")
        try:
            path = "script/iteration#1/operation#1/schedule#1/acquire#2"
            for scope, expected in [("acquire#1", None), ("acquire", 1)]:
                result = host.cancel(scope=scope, expected=expected)
                assert asdict(result) == {
                    "cancelled": False,
                    "reason": "sequence mismatch",
                    "active_path": path,
                }
                assert not current.is_set()
            accepted = host.cancel(scope="acquire#2", expected=2)
            assert accepted.cancelled
            assert current.is_set()
            assert not schedule.is_set()
            assert not ctx.stop_signal.is_set()
        finally:
            current.release()
            schedule.release()

    async def script() -> None:
        result = await host.call(operation)
        assert result.partial_result
        assert not result.stopped

    host.run(script())


def test_root_has_no_sequence_and_inactive_label_returns_current_path() -> None:
    host = ScriptHost()

    async def script() -> None:
        for scope, expected in [("script#1", None), ("script", 1)]:
            result = host.cancel(scope=scope, expected=expected)
            assert asdict(result) == {
                "cancelled": False,
                "reason": "sequence mismatch",
                "active_path": "script/iteration#1",
            }
        result = host.cancel(scope="acquire")
        assert asdict(result) == {
            "cancelled": False,
            "reason": "scope not active",
            "active_path": "script/iteration#1",
        }

    host.run(script())
    assert host.status == "completed"


@pytest.mark.parametrize("expected", [0, -1, True, 1.5, "1"])
def test_invalid_expected_is_a_value_error(expected: int) -> None:
    host = ScriptHost()
    with pytest.raises(ValueError, match="expected"):
        host.cancel(scope="acquire", expected=expected)


def test_scope_validation_rejects_bad_labels_sequences_and_expected_values() -> None:
    host = ScriptHost()

    async def script() -> None:
        # First establish the new expected ABI; bad inputs must then fail as values.
        assert not host.cancel(scope="acquire", expected=1).cancelled
        for scope in [
            "unknown",
            "",
            "acquire#0",
            "acquire#-1",
            "acquire#x",
            "script/acquire#1",
        ]:
            with pytest.raises(ValueError, match="scope|label"):
                host.cancel(scope=scope)
        with pytest.raises(ValueError, match="sequence|expected"):
            host.cancel(scope="acquire#2", expected=1)

    host.run(script())
