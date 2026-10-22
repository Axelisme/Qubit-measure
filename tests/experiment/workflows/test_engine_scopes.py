"""Workflow scoped cancel delivers partial facts and replaces iteration at commit."""

from pathlib import Path

import pytest
from zcu_tools.experiment.script_host import CallContext
from zcu_tools.experiment.workflows import Live, WorkflowEnv

from ._engine_fakes import Plan, Point, Rig, Tunables, completed


@pytest.mark.parametrize("scope", ["acquire", "schedule", "run", "iteration", "script"])
def test_workflow_scope_ack_status_and_partial_facts(
    tmp_path: Path, scope: str
) -> None:
    rig = Rig(tmp_path)
    outcomes: list[tuple[bool, bool]] = []
    acquisitions: list[bool] = []

    async def script(env: WorkflowEnv[int], plan: Plan, tun: Live[Tunables]) -> None:
        def acquire(ctx: CallContext) -> int:
            schedule = ctx.stop_signal.child(label="schedule")
            first = schedule.child(label="acquire")
            try:
                status = rig.engine.status()
                assert status is not None
                assert status.active_path == (
                    "script/iteration#1/operation#1/schedule#1/acquire#1"
                )
                result = rig.engine.cancel("run", scope=scope, expected=None)
                assert result.cancelled
                assert result.reason == "scope cancellation requested"
                assert result.active_path == status.active_path
                acquisitions.append(first.is_set())
            finally:
                first.release()
            second = schedule.child(label="acquire")
            acquisitions.append(second.is_set())
            second.release()
            schedule.release()
            return 11

        first = completed(await env.call(acquire))
        outcomes.append((first.partial_result, first.facts.stopped))
        second = completed(await env.call(lambda ctx: ctx.stop_signal.is_set()))
        outcomes.append((second.partial_result, second.facts.stopped))
        await env.commit(Point(value=11))
        third = completed(await env.call(lambda ctx: ctx.stop_signal.is_set()))
        outcomes.append((third.partial_result, third.facts.stopped))
        await env.commit(Point(value=12))

    rig.start(script)
    try:
        final = rig.engine.execute("run")
        assert final.lifecycle == ("cancelled" if scope == "script" else "done"), (
            final.reason
        )
        assert acquisitions == [True, scope != "acquire"]
        assert outcomes == [
            (True, scope in {"run", "iteration", "script"}),
            (scope in {"iteration", "script"}, scope in {"iteration", "script"}),
            (scope == "script", scope == "script"),
        ]
        records = rig.engine.records("run")
        assert [
            [operation.outcome for operation in record.operations] for record in records
        ] == [
            ["partial", "partial" if scope in {"iteration", "script"} else "completed"],
            ["partial" if scope == "script" else "completed"],
        ]
    finally:
        rig.close()
