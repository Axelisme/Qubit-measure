"""Explicit Host receivers and owner-loop completion recording before root cancel."""

from threading import get_ident

import pytest
from zcu_tools.experiment.scope import Scope, ScriptCancelled
from zcu_tools.experiment.script_host import CallContext, CallResult, ScriptHost


def test_call_uses_explicit_receiver_and_records_on_owner_loop() -> None:
    root = Scope("workflow", name="synthetic")
    host = ScriptHost(root)
    assert getattr(host, "root", None) is root, "host must expose its exact root"
    owner = get_ident()
    seen: list[tuple[int, int]] = []

    def record(result: CallResult[str]) -> None:
        seen.append((get_ident(), result.epoch))

    async def script() -> str:
        with root.child("calibration") as scope:
            result = await host.call(lambda ctx: ctx.scope.path, scope=scope, name="T1", on_completed=record)
            assert host.live_paths == [scope.path]
            return result.result

    assert host.run(script()) == "workflow:synthetic/calibration#1/operation:T1#1"
    assert seen == [(owner, 0)]


def test_root_cancel_records_completion_before_raising_and_does_not_set_host_error() -> None:
    root = Scope("workflow", name="synthetic")
    host = ScriptHost(root)
    assert getattr(host, "root", None) is root, "host must expose its exact root"
    seen: list[str] = []

    def work(ctx: CallContext) -> int:
        host.cancel()
        return 7

    def record(result: CallResult[int]) -> None:
        assert result.result == 7
        assert result.stopped and result.partial_result
        seen.append("recorded")

    async def script() -> None:
        try:
            await host.call(work, on_completed=record)
            seen.append("delivered")
        finally:
            seen.append("finally")

    with pytest.raises(ScriptCancelled):
        host.run(script())
    assert seen == ["recorded", "finally"]
    assert host.status == "cancelled"
    assert host.error is None


@pytest.mark.parametrize("failure", ["fn", "hook"])
def test_real_completion_errors_take_precedence_over_root_cancel(failure: str) -> None:
    root = Scope("workflow", name="synthetic")
    host = ScriptHost(root)
    assert getattr(host, "root", None) is root, "host must expose its exact root"
    error = OSError("recording failed")
    recorded: list[int] = []

    def work(ctx: CallContext) -> int:
        host.cancel()
        if failure == "fn":
            raise error
        return 7

    def record(result: CallResult[int]) -> None:
        recorded.append(result.result)
        raise error

    async def script() -> None:
        await host.call(work, on_completed=record)

    with pytest.raises(OSError) as raised:
        host.run(script())
    assert raised.value is error
    assert host.error is error
    assert recorded == ([] if failure == "fn" else [7])
